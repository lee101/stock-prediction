#!/usr/bin/env python3
"""
Pretrained Model Loader - Handles loading and adapting existing model weights
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import re


class PretrainedWeightLoader:
    """Manages loading and adapting pretrained model weights"""
    
    def __init__(self, models_dir: str = "training/models"):
        self.models_dir = Path(models_dir)
        self.available_models = self._scan_models()
    
    def _scan_models(self) -> List[Dict[str, Any]]:
        """Scan available pretrained models"""
        models = []
        
        for model_path in self.models_dir.glob("*.pth"):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # Extract metadata
                model_info = {
                    'path': str(model_path),
                    'name': model_path.stem,
                    'size': model_path.stat().st_size,
                }
                
                # Try to extract model config if available
                if isinstance(checkpoint, dict):
                    if 'config' in checkpoint:
                        model_info['config'] = checkpoint['config']
                    if 'epoch' in checkpoint:
                        model_info['epoch'] = checkpoint['epoch']
                    if 'metrics' in checkpoint:
                        model_info['metrics'] = checkpoint['metrics']
                    
                    # Count parameters
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = {k: v for k, v in checkpoint.items() 
                                    if isinstance(v, torch.Tensor)}
                    
                    total_params = sum(p.numel() for p in state_dict.values())
                    model_info['total_params'] = total_params
                
                models.append(model_info)
                
            except Exception as e:
                print(f"Warning: Could not load model {model_path}: {e}")
                continue
        
        return sorted(models, key=lambda x: x.get('epoch', 0), reverse=True)
    
    def get_best_model(self, prefer_modern: bool = True) -> Optional[str]:
        """Get the best available model path"""
        if not self.available_models:
            return None
        
        # Prefer modern models if available
        if prefer_modern:
            modern_models = [m for m in self.available_models if 'modern' in m['name']]
            if modern_models:
                return modern_models[0]['path']
        
        # Otherwise return the model with highest epoch
        return self.available_models[0]['path']
    
    def load_compatible_weights(
        self, 
        model: nn.Module, 
        pretrained_path: str,
        strict: bool = False,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Load compatible weights from pretrained model"""
        
        if exclude_patterns is None:
            exclude_patterns = [
                r'.*classifier.*',  # Exclude final classification layers
                r'.*head.*',        # Exclude head layers
                r'.*output.*'       # Exclude output layers
            ]
        
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            else:
                pretrained_dict = checkpoint
            
            # Get current model state
            model_dict = model.state_dict()
            
            # Filter out excluded patterns
            filtered_dict = {}
            excluded_keys = []
            
            for key, value in pretrained_dict.items():
                should_exclude = any(re.match(pattern, key) for pattern in exclude_patterns)
                
                if should_exclude:
                    excluded_keys.append(key)
                    continue
                
                # Check if key exists in current model and shapes match
                if key in model_dict:
                    if model_dict[key].shape == value.shape:
                        filtered_dict[key] = value
                    else:
                        print(f"Shape mismatch for {key}: "
                              f"model {model_dict[key].shape} vs pretrained {value.shape}")
                else:
                    print(f"Key {key} not found in current model")
            
            # Load the filtered weights
            missing_keys, unexpected_keys = model.load_state_dict(
                filtered_dict, strict=False
            )
            
            loaded_count = len(filtered_dict)
            total_model_params = len(model_dict)
            
            print(f"Loaded {loaded_count}/{total_model_params} parameters from {pretrained_path}")
            print(f"Missing keys: {len(missing_keys)}")
            print(f"Unexpected keys: {len(unexpected_keys)}")
            print(f"Excluded keys: {len(excluded_keys)}")
            
            return {
                'loaded_params': loaded_count,
                'total_params': total_model_params,
                'missing_keys': missing_keys,
                'unexpected_keys': unexpected_keys,
                'excluded_keys': excluded_keys,
                'load_ratio': loaded_count / total_model_params
            }
            
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            return {'error': str(e)}
    
    def create_embedding_backbone(self, pretrained_path: str) -> nn.Module:
        """Create embedding backbone from pretrained model"""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Extract transformer/encoder components
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Find transformer/encoder layers
            transformer_keys = [k for k in state_dict.keys() 
                              if any(pattern in k.lower() for pattern in 
                                   ['transformer', 'encoder', 'attention'])]
            
            if not transformer_keys:
                print("No transformer layers found, creating fallback backbone")
                return self._create_fallback_backbone()
            
            # Try to reconstruct transformer architecture
            # This is simplified - you might need to adjust based on your model structure
            d_model = self._infer_model_dim(state_dict)
            nhead = self._infer_num_heads(state_dict)
            num_layers = self._infer_num_layers(state_dict)
            
            backbone = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 2,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=num_layers
            )
            
            # Load compatible weights
            self.load_compatible_weights(
                backbone, 
                pretrained_path,
                exclude_patterns=[r'.*classifier.*', r'.*head.*', r'.*output.*', r'.*action.*']
            )
            
            return backbone
            
        except Exception as e:
            print(f"Error creating backbone: {e}")
            return self._create_fallback_backbone()
    
    def _infer_model_dim(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Infer model dimension from state dict"""
        # Look for embedding or attention weights to infer dimension
        for key, tensor in state_dict.items():
            if 'embed' in key.lower() or 'in_proj' in key.lower():
                if len(tensor.shape) >= 2:
                    return tensor.shape[-1]
        return 128  # Default fallback
    
    def _infer_num_heads(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Infer number of attention heads"""
        # This is tricky to infer, use a reasonable default
        d_model = self._infer_model_dim(state_dict)
        return max(1, d_model // 32)  # Common ratio
    
    def _infer_num_layers(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """Infer number of transformer layers"""
        layer_keys = [k for k in state_dict.keys() if 'layers.' in k]
        if layer_keys:
            layer_numbers = []
            for key in layer_keys:
                match = re.search(r'layers\.(\d+)\.', key)
                if match:
                    layer_numbers.append(int(match.group(1)))
            return max(layer_numbers) + 1 if layer_numbers else 2
        return 2  # Default fallback
    
    def _create_fallback_backbone(self) -> nn.Module:
        """Create fallback backbone if loading fails"""
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
    
    def print_model_summary(self):
        """Print summary of available models"""
        print("\n" + "="*60)
        print("AVAILABLE PRETRAINED MODELS")
        print("="*60)
        
        for i, model in enumerate(self.available_models):
            print(f"\n{i+1}. {model['name']}")
            print(f"   Path: {model['path']}")
            print(f"   Size: {model['size'] / (1024*1024):.2f} MB")
            if 'total_params' in model:
                print(f"   Parameters: {model['total_params']:,}")
            if 'epoch' in model:
                print(f"   Epoch: {model['epoch']}")
            if 'metrics' in model:
                metrics = model['metrics']
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}")
        
        if self.available_models:
            best_model = self.get_best_model()
            print(f"\nRecommended model: {best_model}")
        else:
            print("\nNo models found!")
    
    def export_embedding_weights(
        self, 
        model: nn.Module, 
        output_path: str,
        include_metadata: bool = True
    ):
        """Export embedding weights for reuse"""
        
        embedding_weights = {}
        metadata = {}
        
        # Extract embedding layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                embedding_weights[name] = module.weight.detach().cpu()
                metadata[name] = {
                    'num_embeddings': module.num_embeddings,
                    'embedding_dim': module.embedding_dim,
                    'shape': list(module.weight.shape)
                }
        
        # Save weights
        save_dict = {'embeddings': embedding_weights}
        if include_metadata:
            save_dict['metadata'] = metadata
        
        torch.save(save_dict, output_path)
        print(f"Exported {len(embedding_weights)} embedding layers to {output_path}")


if __name__ == "__main__":
    # Test the loader
    loader = PretrainedWeightLoader()
    loader.print_model_summary()