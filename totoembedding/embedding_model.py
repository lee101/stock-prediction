#!/usr/bin/env python3
"""
Toto Embedding Model - Use real Toto backbone when available

Two operation modes:
- use_toto=True: Load Datadog Toto and derive embeddings from it
  - Preferred: try to obtain encoder hidden states
  - Fallback: summarize Toto forecast distributions (means/stds over horizon)
- use_toto=False: Fallback small TransformerEncoder backbone with optional weight loader
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

try:
    # Optional Toto dependencies; code guards execution if unavailable
    from toto.data.util.dataset import MaskedTimeseries
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto
    _TOTO_AVAILABLE = True
except Exception:
    _TOTO_AVAILABLE = False

from totoembedding.pretrained_loader import PretrainedWeightLoader

class TotoEmbeddingModel(nn.Module):
    """
    Toto embedding model that reuses pretrained transformer weights
    and adds specialized embedding layers for stock market data
    """
    
    def __init__(
        self,
        pretrained_model_path: Optional[str] = None,
        embedding_dim: int = 128,
        num_symbols: int = 21,  # Based on your trainingdata
        freeze_backbone: bool = True,
        symbol_embedding_dim: int = 32,
        market_context_dim: int = 16,
        input_feature_dim: int = 11,
        # Toto-specific
        use_toto: bool = True,
        toto_model_id: str = 'Datadog/Toto-Open-Base-1.0',
        toto_device: str = 'cuda',
        series_feature_index: int = 3,  # index of 'Close' in default feature order
        toto_horizon: int = 8,
        toto_num_samples: int = 256,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_symbols = num_symbols
        self.freeze_backbone = freeze_backbone
        self.input_feature_dim = input_feature_dim
        self.series_feature_index = series_feature_index
        self.use_toto = use_toto and _TOTO_AVAILABLE
        self.toto_horizon = toto_horizon
        self.toto_num_samples = toto_num_samples
        self.toto_device = toto_device

        # Initialize backbone
        self._backbone_mode = 'fallback'  # 'toto_encode' | 'toto_forecast_stats' | 'transformer' | 'fallback'
        self.toto = None
        self.toto_model = None
        self.toto_forecaster = None
        self.backbone = None
        self.input_proj = None

        if self.use_toto:
            # Try to load Toto and prefer encoder hidden states
            self._init_toto_backbone(toto_model_id)
        else:
            # Load fallback transformer backbone (optionally with weights)
            self.backbone = self._load_pretrained_backbone(pretrained_model_path)
            if freeze_backbone and hasattr(self.backbone, 'parameters'):
                for param in self.backbone.parameters():
                    param.requires_grad = False
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
        
    def _init_toto_backbone(self, model_id: str) -> None:
        """Initialize Toto model and decide on embedding strategy."""
        try:
            self.toto = Toto.from_pretrained(model_id)
            self.toto_model = self.toto.model
            try:
                self.toto_model.to(self.toto_device)
            except Exception:
                pass
            # Place the model in eval mode; let caller decide device move
            self.toto_model.eval()
            try:
                self.toto_model.compile()
            except Exception:
                pass

            # Try to create a forecaster helper for forecast-based features
            try:
                self.toto_forecaster = TotoForecaster(self.toto_model)
            except Exception:
                self.toto_forecaster = None

            # Prefer using encoder hidden states if available
            hidden_size = None
            if hasattr(self.toto_model, 'config') and hasattr(self.toto_model.config, 'hidden_size'):
                hidden_size = int(self.toto_model.config.hidden_size)

            # Probe for likely encoding methods
            if any(hasattr(self.toto_model, attr) for attr in ['encode', 'forward']):
                # Use encoder embeddings path if we can obtain hidden states
                if hidden_size is not None:
                    self.backbone_dim = hidden_size
                    self._backbone_mode = 'toto_encode'
                else:
                    # Fallback to summarized forecast stats with fixed dim
                    self.backbone_dim = 2 * self.toto_horizon
                    self._backbone_mode = 'toto_forecast_stats'
            else:
                # Use forecast statistics as Toto-derived features
                self.backbone_dim = 2 * self.toto_horizon
                self._backbone_mode = 'toto_forecast_stats'

        except Exception as e:
            print(f"Warning: Failed to initialize Toto backbone: {e}")
            # Fallback to transformer
            self.backbone = self._create_fallback_backbone()
            if self.freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            self.backbone_dim = self._get_backbone_output_dim()
            self.input_proj = nn.Linear(self.input_feature_dim, self.backbone_dim)
            self._backbone_mode = 'transformer'

    def _load_pretrained_backbone(self, model_path: Optional[str]):
        """Load pretrained transformer backbone as a proper nn.Module"""
        try:
            if model_path:
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
        
        # Time embeddings - clamp to valid ranges
        hour_emb = self.time_embeddings['hour'](timestamps[:, 0].clamp(0, 23))
        dow_emb = self.time_embeddings['day_of_week'](timestamps[:, 1].clamp(0, 6))
        month_emb = self.time_embeddings['month'](timestamps[:, 2].clamp(0, 11))
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
        """Process price data through chosen backbone and return [batch, backbone_dim]."""
        if self._backbone_mode == 'toto_encode':
            return self._encode_with_toto(price_data)
        if self._backbone_mode == 'toto_forecast_stats':
            return self._toto_forecast_stats(price_data)
        if isinstance(self.backbone, nn.TransformerEncoder) and self.input_proj is not None:
            # Project raw price features to backbone dim and run transformer encoder
            x = self.input_proj(price_data)  # [batch, seq, d_model]
            x = self.backbone(x)             # [batch, seq, d_model]
            return x.mean(dim=1)             # Pool over sequence dimension
        # Final fallback: simple mean over features and a learnable projection
        pooled = price_data.mean(dim=1)
        proj = getattr(self, '_fallback_proj', None)
        if proj is None:
            self._fallback_proj = nn.Linear(self.input_feature_dim, self.backbone_dim)
            proj = self._fallback_proj
        return proj(pooled)

    def _encode_with_toto(self, price_data: torch.Tensor) -> torch.Tensor:
        """Use Toto encoder to obtain hidden states and pool them."""
        device = self.toto_device
        bsz, seq_len, feat = price_data.shape
        # Use selected feature (e.g., Close) as univariate series expected by Toto
        series = price_data[:, :, self.series_feature_index].detach().to(torch.float32)
        outputs: List[torch.Tensor] = []
        for i in range(bsz):
            ctx = series[i]  # [seq]
            ctx = ctx.unsqueeze(0)  # [1, seq]
            # Build timestamps assuming fixed interval
            timestamp_seconds = torch.zeros(1, seq_len, device=ctx.device)
            time_interval_seconds = torch.full((1,), 60 * 15, device=ctx.device)
            mts = MaskedTimeseries(
                series=ctx.to(device),
                padding_mask=torch.full_like(ctx, True, dtype=torch.bool).to(device),
                id_mask=torch.zeros_like(ctx).to(device),
                timestamp_seconds=timestamp_seconds.to(device),
                time_interval_seconds=time_interval_seconds.to(device),
            )
            with torch.inference_mode():
                enc_hidden = None
                try:
                    if hasattr(self.toto_model, 'encode'):
                        enc_hidden = self.toto_model.encode(mts)
                    else:
                        res = self.toto_model(mts)
                        # Common attribute names to probe
                        if isinstance(res, dict):
                            enc_hidden = res.get('last_hidden_state', None) or res.get('encoder_output', None)
                        elif isinstance(res, (tuple, list)) and len(res) > 0:
                            enc_hidden = res[0]
                except Exception:
                    enc_hidden = None
                if enc_hidden is None:
                    # Fallback to forecast stats for this sample
                    outputs.append(self._toto_forecast_stats(price_data[i:i+1]).squeeze(0))
                else:
                    # enc_hidden could be [1, seq, hidden] or [seq, hidden]
                    if enc_hidden.dim() == 2:
                        pooled = enc_hidden.mean(dim=0)
                    elif enc_hidden.dim() == 3:
                        pooled = enc_hidden.mean(dim=1)
                    else:
                        pooled = enc_hidden.flatten()[: self.backbone_dim]
                    outputs.append(pooled.detach().to('cpu'))
        return torch.stack(outputs, dim=0)

    def _toto_forecast_stats(self, price_data: torch.Tensor) -> torch.Tensor:
        """Summarize Toto forecast distributions as fixed-dim features per sample."""
        if self.toto_forecaster is None:
            # As a last resort, fall back to transformer path
            if isinstance(self.backbone, nn.TransformerEncoder) and self.input_proj is not None:
                x = self.input_proj(price_data)
                x = self.backbone(x)
                return x.mean(dim=1)
            pooled = price_data.mean(dim=1)
            proj = getattr(self, '_fallback_proj', None)
            if proj is None:
                self._fallback_proj = nn.Linear(self.input_feature_dim, self.backbone_dim)
                proj = self._fallback_proj
            return proj(pooled)

        device = self.toto_device
        bsz, seq_len, feat = price_data.shape
        series = price_data[:, :, self.series_feature_index].detach().to(torch.float32)
        feats = []
        for i in range(bsz):
            ctx = series[i].unsqueeze(0)  # [1, seq]
            timestamp_seconds = torch.zeros(1, seq_len)
            time_interval_seconds = torch.full((1,), 60 * 15)
            mts = MaskedTimeseries(
                series=ctx.to(device),
                padding_mask=torch.full_like(ctx, True, dtype=torch.bool).to(device),
                id_mask=torch.zeros_like(ctx).to(device),
                timestamp_seconds=timestamp_seconds.to(device),
                time_interval_seconds=time_interval_seconds.to(device),
            )
            with torch.inference_mode():
                try:
                    forecast = self.toto_forecaster.forecast(
                        mts,
                        prediction_length=self.toto_horizon,
                        num_samples=self.toto_num_samples,
                        samples_per_batch=min(self.toto_num_samples, 256),
                    )
                    samples = getattr(forecast, 'samples', None)
                except Exception:
                    samples = None
            if samples is None:
                # If forecaster failed, back off to zeros
                feats.append(torch.zeros(self.backbone_dim))
            else:
                # Expected shapes vary; try to reduce to [horizon, samples]
                s = samples
                if isinstance(s, torch.Tensor):
                    t = s
                else:
                    try:
                        t = torch.tensor(s)
                    except Exception:
                        feats.append(torch.zeros(self.backbone_dim))
                        continue
                while t.dim() > 2:
                    t = t.squeeze(0)
                # Now t shape approximately [horizon, num_samples]
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                means = t.mean(dim=1)
                stds = t.std(dim=1)
                feat_vec = torch.cat([means, stds], dim=0)
                # Ensure fixed size 2*horizon
                if feat_vec.numel() != 2 * self.toto_horizon:
                    # Pad or truncate
                    if feat_vec.numel() < 2 * self.toto_horizon:
                        pad = torch.zeros(2 * self.toto_horizon - feat_vec.numel())
                        feat_vec = torch.cat([feat_vec, pad], dim=0)
                    else:
                        feat_vec = feat_vec[: 2 * self.toto_horizon]
                feats.append(feat_vec.detach().to('cpu'))
        return torch.stack(feats, dim=0)
    
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
        if isinstance(self.backbone, nn.Module):
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        if isinstance(self.backbone, nn.Module):
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
