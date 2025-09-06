#!/usr/bin/env python3
"""
Toto Embedding Model - Reuses pretrained weights for stock market understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

from pretrained_loader import PretrainedWeightLoader

class TotoEmbeddingModel(nn.Module):
    """
    Toto embedding model that reuses pretrained transformer weights
    and adds specialized embedding layers for stock market data
    """
    
    def __init__(
        self,
        pretrained_model_path: str,
        embedding_dim: int = 128,
        num_symbols: int = 21,  # Based on your trainingdata
        freeze_backbone: bool = True,
        symbol_embedding_dim: int = 32,
        market_context_dim: int = 16,
        input_feature_dim: int = 11,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_symbols = num_symbols
        self.freeze_backbone = freeze_backbone
        self.input_feature_dim = input_feature_dim
        
        # Load pretrained backbone using robust loader
        self.backbone = self._load_pretrained_backbone(pretrained_model_path)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Determine backbone model dimension (d_model) and add input projection
        self.backbone_dim = self._get_backbone_output_dim()
        self.input_proj = nn.Linear(self.input_feature_dim, self.backbone_dim)

        # Symbol embeddings for different stocks/crypto
        self.symbol_embeddings = nn.Embedding(num_symbols, symbol_embedding_dim)
        
        # Market regime embeddings (bull, bear, sideways, volatile)
        self.regime_embeddings = nn.Embedding(4, market_context_dim)
        
        # Time-based embeddings (hour of day, day of week, etc.)
        self.time_embeddings = nn.ModuleDict({
            'hour': nn.Embedding(24, 8),
            'day_of_week': nn.Embedding(7, 4),
            'month': nn.Embedding(12, 4),
        })
        
        # Cross-asset correlation encoder
        self.correlation_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Projection layers from backbone + context to final embedding space
        backbone_dim = self.backbone_dim
        total_context_dim = symbol_embedding_dim + market_context_dim + 16  # time embeddings total
        
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim + total_context_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Multi-asset attention for cross-pair relationships
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
    def _load_pretrained_backbone(self, model_path: str):
        """Load pretrained transformer backbone as a proper nn.Module"""
        try:
            loader = PretrainedWeightLoader(models_dir=str(Path(model_path).parent))
            backbone = loader.create_embedding_backbone(model_path)
            return backbone
        except Exception as e:
            print(f"Warning: Could not load pretrained model backbone: {e}")
            # Fallback to random initialization
            return self._create_fallback_backbone()
    
    def _create_fallback_backbone(self):
        """Create fallback backbone if pretrained loading fails"""
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
    
    def _get_backbone_output_dim(self) -> int:
        """Infer the backbone (transformer) model dimension (d_model)."""
        # If using a standard TransformerEncoder, infer from first layer
        try:
            if isinstance(self.backbone, nn.TransformerEncoder):
                layer0 = self.backbone.layers[0]
                # Prefer attention embed dim when available
                if hasattr(layer0, 'self_attn') and hasattr(layer0.self_attn, 'embed_dim'):
                    return int(layer0.self_attn.embed_dim)
                # Fallback to first linear layer input
                if hasattr(layer0, 'linear1') and hasattr(layer0.linear1, 'in_features'):
                    return int(layer0.linear1.in_features)
        except Exception:
            pass
        # Fallback
        return 128
    
    def forward(
        self,
        price_data: torch.Tensor,  # [batch, seq_len, features]
        symbol_ids: torch.Tensor,  # [batch]
        timestamps: torch.Tensor,  # [batch, 3] - hour, day_of_week, month
        market_regime: torch.Tensor,  # [batch]
        cross_asset_data: Optional[torch.Tensor] = None  # [batch, num_assets, seq_len, features]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through toto embedding model
        
        Returns:
            embeddings: Stock-specific embeddings
            cross_embeddings: Cross-asset relationship embeddings
            attention_weights: Attention weights for interpretability
        """
        batch_size = price_data.shape[0]
        
        # Get backbone embeddings
        backbone_output = self._process_backbone(price_data)
        
        # Generate contextual embeddings
        symbol_emb = self.symbol_embeddings(symbol_ids)  # [batch, symbol_dim]
        regime_emb = self.regime_embeddings(market_regime)  # [batch, regime_dim]
        
        # Time embeddings
        hour_emb = self.time_embeddings['hour'](timestamps[:, 0])
        dow_emb = self.time_embeddings['day_of_week'](timestamps[:, 1])
        month_emb = self.time_embeddings['month'](timestamps[:, 2])
        time_emb = torch.cat([hour_emb, dow_emb, month_emb], dim=-1)
        
        # Combine all context
        context = torch.cat([symbol_emb, regime_emb, time_emb], dim=-1)
        
        # Project to final embedding space
        combined = torch.cat([backbone_output, context], dim=-1)
        embeddings = self.projection(combined)
        
        # Cross-asset processing if available
        cross_embeddings = None
        attention_weights = None
        
        if cross_asset_data is not None:
            cross_embeddings, attention_weights = self._process_cross_assets(
                embeddings, cross_asset_data
            )
        
        return {
            'embeddings': embeddings,
            'cross_embeddings': cross_embeddings,
            'attention_weights': attention_weights,
            'symbol_embeddings': symbol_emb,
            'regime_embeddings': regime_emb
        }
    
    def _process_backbone(self, price_data: torch.Tensor) -> torch.Tensor:
        """Process price data through backbone"""
        # Project raw price features to backbone dim and run transformer encoder
        x = self.input_proj(price_data)  # [batch, seq, d_model]
        x = self.backbone(x)             # [batch, seq, d_model]
        return x.mean(dim=1)             # Pool over sequence dimension
    
    def _process_cross_assets(
        self, 
        base_embeddings: torch.Tensor, 
        cross_asset_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process cross-asset relationships"""
        batch_size, num_assets, seq_len, features = cross_asset_data.shape
        
        # Reshape for processing
        cross_data = cross_asset_data.view(-1, seq_len, features)
        cross_backbone = self._process_backbone(cross_data)
        cross_backbone = cross_backbone.view(batch_size, num_assets, -1)
        
        # Apply cross attention
        query = base_embeddings.unsqueeze(1)  # [batch, 1, embed_dim]
        key = value = cross_backbone  # [batch, num_assets, embed_dim]
        
        cross_embeddings, attention_weights = self.cross_attention(
            query, key, value
        )
        
        return cross_embeddings.squeeze(1), attention_weights
    
    def get_symbol_similarities(self) -> torch.Tensor:
        """Get similarity matrix between symbols"""
        embeddings = self.symbol_embeddings.weight
        similarities = torch.mm(embeddings, embeddings.t())
        return F.normalize(similarities, dim=-1)
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def save_embeddings(self, filepath: str):
        """Save learned embeddings"""
        embeddings = {
            'symbol_embeddings': self.symbol_embeddings.weight.detach().cpu(),
            'regime_embeddings': self.regime_embeddings.weight.detach().cpu(),
            'time_embeddings': {
                name: emb.weight.detach().cpu() 
                for name, emb in self.time_embeddings.items()
            }
        }
        torch.save(embeddings, filepath)


class TotoEmbeddingDataset(torch.utils.data.Dataset):
    """Dataset for training toto embeddings"""
    
    def __init__(
        self,
        data_dir: str,
        symbols: List[str],
        window_size: int = 30,
        cross_asset_window: int = 10
    ):
        self.data_dir = Path(data_dir)
        self.symbols = symbols
        self.window_size = window_size
        self.cross_asset_window = cross_asset_window
        
        # Load all data
        self.data = {}
        self.symbol_to_id = {sym: i for i, sym in enumerate(symbols)}
        
        for symbol in symbols:
            filepath = self.data_dir / f"{symbol}.csv"
            if filepath.exists():
                df = pd.read_csv(filepath, parse_dates=['timestamp'])
                df = self._add_features(df)
                self.data[symbol] = df
        
        self.samples = self._create_samples()
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical features"""
        # Price features
        df['Returns'] = df['Close'].pct_change()
        df['HL_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['OC_Ratio'] = (df['Open'] - df['Close']) / df['Close']
        
        # Moving averages
        for window in [5, 10, 20]:
            df[f'MA_{window}'] = df['Close'].rolling(window).mean()
            df[f'MA_Ratio_{window}'] = df['Close'] / df[f'MA_{window}']
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(20).std()
        
        # Time features
        df['Hour'] = df['timestamp'].dt.hour
        df['DayOfWeek'] = df['timestamp'].dt.dayofweek
        df['Month'] = df['timestamp'].dt.month
        
        # Market regime (simplified)
        df['Regime'] = 0  # Default to neutral
        vol_threshold = df['Volatility'].quantile(0.75)
        df.loc[df['Volatility'] > vol_threshold, 'Regime'] = 3  # Volatile
        
        return df.fillna(0)
    
    def _create_samples(self) -> List[Dict]:
        """Create training samples"""
        samples = []
        
        for symbol, df in self.data.items():
            for i in range(self.window_size, len(df)):
                window_data = df.iloc[i-self.window_size:i]
                current_row = df.iloc[i]
                
                sample = {
                    'symbol': symbol,
                    'symbol_id': self.symbol_to_id[symbol],
                    'price_data': window_data[['Open', 'High', 'Low', 'Close', 'Returns', 'HL_Ratio', 'OC_Ratio', 'MA_Ratio_5', 'MA_Ratio_10', 'MA_Ratio_20', 'Volatility']].values,
                    'timestamp': [current_row['Hour'], current_row['DayOfWeek'], current_row['Month']],
                    'regime': current_row['Regime'],
                    'target_return': df.iloc[i+1]['Returns'] if i+1 < len(df) else 0.0
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            'price_data': torch.tensor(sample['price_data'], dtype=torch.float32),
            'symbol_id': torch.tensor(sample['symbol_id'], dtype=torch.long),
            'timestamp': torch.tensor(sample['timestamp'], dtype=torch.long),
            'regime': torch.tensor(sample['regime'], dtype=torch.long),
            'target_return': torch.tensor(sample['target_return'], dtype=torch.float32)
        }
