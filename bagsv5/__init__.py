"""BagsV5 - State-of-the-art transformer for trading signals.

Features:
- Muon optimizer (2-3x faster convergence)
- RMSNorm (faster than LayerNorm)
- SwiGLU activation
- Rotary Position Embeddings (RoPE)
- Flash Attention
- Multi-token pre-training
"""

from bagsv5.model import BagsV5Config, BagsV5Transformer, FocalLoss
from bagsv5.dataset import (
    MultiTokenDataset,
    SingleTokenDataset,
    FeatureNormalizer,
    load_multi_token_data,
    load_ohlc_dataframe,
)
from bagsv5.train import train_v5, pretrain_multi_token, finetune_codex
from bagsv5.simulator import MarketSimulator, run_simulation, optimize_thresholds

__all__ = [
    "BagsV5Config",
    "BagsV5Transformer",
    "FocalLoss",
    "MultiTokenDataset",
    "SingleTokenDataset",
    "FeatureNormalizer",
    "load_multi_token_data",
    "load_ohlc_dataframe",
    "train_v5",
    "pretrain_multi_token",
    "finetune_codex",
    "MarketSimulator",
    "run_simulation",
    "optimize_thresholds",
]
