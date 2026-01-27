"""BagsV3LLM - Transformer-based neural trading model.

Key features:
- Modern transformer architecture with RoPE, QK norm, ReLU^2
- 256 context length for longer pattern recognition
- Chronos2 forecast integration for additional features
- Pre-training on stock data for transfer learning
- Fine-tuning on target asset (CODEX)

Architecture improvements over V2:
- Transformer vs LSTM for better parallelization and longer context
- Rotary position embeddings for relative position encoding
- Multi-head attention for capturing different pattern types
- Pre-training objective for better weight initialization
"""

from bagsv3llm.model import (
    BagsV3Config,
    BagsV3Transformer,
    BagsV3PretrainModel,
    FocalLoss,
    PretrainLoss,
)
from bagsv3llm.dataset import (
    FeatureNormalizerV3,
    load_pretraining_data,
    load_ohlc_dataframe,
    build_bar_features,
    build_aggregate_features,
    build_chronos_features,
    PretrainingDataset,
    FinetuningDataset,
)

__all__ = [
    # Model
    "BagsV3Config",
    "BagsV3Transformer",
    "BagsV3PretrainModel",
    "FocalLoss",
    "PretrainLoss",
    # Dataset
    "FeatureNormalizerV3",
    "load_pretraining_data",
    "load_ohlc_dataframe",
    "build_bar_features",
    "build_aggregate_features",
    "build_chronos_features",
    "PretrainingDataset",
    "FinetuningDataset",
]
