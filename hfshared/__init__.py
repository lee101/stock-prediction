"""Shared utilities used by both hftraining and hfinference.

Modules:
- features: common feature engineering and normalization helpers
- checkpoint: helpers for reading model metadata from checkpoints
- scaler: helpers for loading/saving processor scalers
"""

from .features import (
    standardize_column_names,
    training_feature_columns_list,
    compute_training_style_features,
    compute_compact_features,
    zscore_per_window,
    normalize_with_scaler,
    denormalize_with_scaler,
)
from .checkpoint import infer_input_dim_from_state
from .scaler import load_processor

__all__ = [
    'standardize_column_names',
    'training_feature_columns_list',
    'compute_training_style_features',
    'compute_compact_features',
    'zscore_per_window',
    'normalize_with_scaler',
    'denormalize_with_scaler',
    'infer_input_dim_from_state',
    'load_processor',
]

