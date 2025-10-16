import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger
import json
import sys
sys.path.append('training')

from trading_agent import TradingAgent


class ModelManager:
    def __init__(self, models_dir: Path = Path("models"), device: str = None):
        self.models_dir = Path(models_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models: Dict[str, TradingAgent] = {}
        self.model_configs: Dict[str, dict] = {}
        
    def load_model_for_symbol(self, symbol: str, model_path: Optional[Path] = None) -> TradingAgent:
        """Load a trained model for a specific symbol."""
        
        if model_path is None:
            # Look for symbol-specific model first
            symbol_model = self.models_dir / f"{symbol}_best_model.pth"
            if symbol_model.exists():
                model_path = symbol_model
            else:
                # Fallback to generic best model
                model_path = self.models_dir / "best_model.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model for {symbol} from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Determine input dimensions from saved state
        # This is a bit hacky but works for our current architecture
        first_layer_weight = None
        for key in checkpoint['agent_state_dict'].keys():
            if 'backbone.0.weight' in key or 'backbone.1.weight' in key:
                first_layer_weight = checkpoint['agent_state_dict'][key]
                break
        
        if first_layer_weight is not None:
            input_dim = first_layer_weight.shape[1]
        else:
            # Default fallback
            input_dim = 30 * 13  # window_size * num_features
        
        # Create model with matching architecture
        agent = TradingAgent(
            backbone_model=nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 768),
                nn.ReLU()
            ),
            hidden_dim=768,
            action_std_init=0.5
        )
        
        # Load weights
        agent.load_state_dict(checkpoint['agent_state_dict'])
        agent.to(self.device)
        agent.eval()
        
        # Store model and config
        self.models[symbol] = agent
        self.model_configs[symbol] = {
            'model_path': str(model_path),
            'input_dim': input_dim
        }
        
        return agent
    
    def load_top_k_models(self, symbol: str, k: int = 5) -> List[TradingAgent]:
        """Load top-k profitable models for ensemble predictions."""
        
        models = []
        summary_file = self.models_dir / f"{symbol}_top_k_summary.json"
        
        if not summary_file.exists():
            # Try generic top_k_summary
            summary_file = self.models_dir / "top_k_summary.json"
        
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            for model_info in summary.get('top_k_models', [])[:k]:
                model_path = self.models_dir / model_info['filename']
                if model_path.exists():
                    try:
                        agent = self.load_model_for_symbol(f"{symbol}_top_{len(models)}", model_path)
                        models.append(agent)
                        logger.info(f"Loaded top model: {model_info['filename']} (return: {model_info['total_return']:.2%})")
                    except Exception as e:
                        logger.error(f"Failed to load model {model_path}: {e}")
        
        if not models:
            # Fallback to loading the best model
            logger.warning(f"No top-k models found for {symbol}, loading best model")
            models.append(self.load_model_for_symbol(symbol))
        
        return models
    
    def predict(
        self, 
        symbol: str, 
        observation: np.ndarray,
        deterministic: bool = True,
        use_ensemble: bool = False
    ) -> Tuple[float, float]:
        """Get action prediction from model(s)."""
        
        # Ensure observation is tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        if use_ensemble and symbol in self.models:
            # Use ensemble of top-k models
            models = self.load_top_k_models(symbol)
            actions = []
            values = []
            
            with torch.no_grad():
                for model in models:
                    action, _, value = model.act(obs_tensor, deterministic=deterministic)
                    actions.append(action.cpu().numpy().flatten()[0])
                    values.append(value.cpu().numpy().flatten()[0])
            
            # Average predictions
            action = np.mean(actions)
            value = np.mean(values)
            
            logger.debug(f"Ensemble prediction for {symbol}: action={action:.3f}, value={value:.3f}")
            
        else:
            # Single model prediction
            if symbol not in self.models:
                self.load_model_for_symbol(symbol)
            
            model = self.models[symbol]
            
            with torch.no_grad():
                action, _, value = model.act(obs_tensor, deterministic=deterministic)
                action = action.cpu().numpy().flatten()[0]
                value = value.cpu().numpy().flatten()[0]
            
            logger.debug(f"Model prediction for {symbol}: action={action:.3f}, value={value:.3f}")
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        return float(action), float(value)
    
    def get_position_recommendation(
        self, 
        symbol: str, 
        observation: np.ndarray,
        max_position_size: float = 1.0,
        use_ensemble: bool = False
    ) -> Dict:
        """Get position recommendation from model."""
        
        action, value = self.predict(symbol, observation, deterministic=True, use_ensemble=use_ensemble)
        
        # Convert action to position recommendation
        # Action is in [-1, 1] where negative is short, positive is long
        position_size = abs(action) * max_position_size
        side = "buy" if action > 0 else "sell"
        
        # Confidence based on action magnitude
        confidence = abs(action)
        
        recommendation = {
            'symbol': symbol,
            'side': side,
            'position_size': position_size,
            'raw_action': action,
            'value_estimate': value,
            'confidence': confidence
        }
        
        logger.info(f"Recommendation for {symbol}: {side} with size {position_size:.2%} (confidence: {confidence:.2%})")
        
        return recommendation