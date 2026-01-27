"""Bags v2 - Improved neural trading model.

Key improvements over v1:
- LSTM backbone for better temporal pattern recognition
- Attention mechanism to weight important time steps
- Focal loss for handling class imbalance
- Enhanced feature engineering
- Walk-forward validation support
"""

from bagsv2.model import BagsNeuralModelV2, BagsNeuralModelV2Simple, FocalLoss
from bagsv2.dataset import (
    FeatureNormalizerV2,
    build_window_features_v2,
    build_features_and_targets,
    fit_normalizer,
    load_ohlc_dataframe,
)

__all__ = [
    "BagsNeuralModelV2",
    "BagsNeuralModelV2Simple",
    "FocalLoss",
    "FeatureNormalizerV2",
    "build_window_features_v2",
    "build_features_and_targets",
    "fit_normalizer",
    "load_ohlc_dataframe",
]
