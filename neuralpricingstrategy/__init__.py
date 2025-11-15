"""
Neural pricing strategy package.

This package contains the training and serving utilities for the neural
pricing model that reforecasts maxdiff buy/sell price bands using Chronos
features, portfolio context, and historical PnL relationships.
"""

from .data import (
    DEFAULT_CATEGORICAL_COLUMNS,
    DEFAULT_NUMERIC_COLUMNS,
    PricingDataset,
    build_pricing_dataset,
    load_backtest_frames,
    split_dataset_by_date,
)
from .models import PricingAdjustmentModel, PricingModelConfig
from .trainer import PricingTrainingConfig, PricingTrainingResult, train_pricing_model

__all__ = [
    "DEFAULT_CATEGORICAL_COLUMNS",
    "DEFAULT_NUMERIC_COLUMNS",
    "PricingAdjustmentModel",
    "PricingDataset",
    "PricingModelConfig",
    "PricingTrainingConfig",
    "PricingTrainingResult",
    "build_pricing_dataset",
    "load_backtest_frames",
    "split_dataset_by_date",
    "train_pricing_model",
]
