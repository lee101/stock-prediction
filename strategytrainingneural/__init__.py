"""
Neural and gradient-boosted portfolio allocators for strategytraining datasets.

This package hosts experimental research code that learns trade weights directly
from the datasets emitted by the existing strategytraining pipeline.  The
modules expose:

- feature specification utilities for converting per-strategy daily metrics
  into normalized tensors suitable for Torch / XGBoost models;
- differentiable Sortino / PnL objectives that operate on grouped daily returns;
- lightweight MLP-based weighting policies plus GPU-friendly trainers; and
- gradient-boosted baselines that optimise the same joint Sortino + PnL metric.

See ``strategytrainingneural.trainer`` for the entry point that wires those
pieces together.
"""

from .data import DailyStrategyDataset, load_daily_metrics
from .feature_builder import FeatureBuilder, FeatureSpec
from .metrics import annualised_sortino, combine_sortino_and_return
from .models import PortfolioPolicy, PolicyConfig
from .trade_windows import load_trade_window_metrics
from .current_symbols import load_current_symbols, split_by_asset_class
from .forecast_cache import ChronosForecastGenerator, ForecastGenerationConfig

__all__ = [
    "DailyStrategyDataset",
    "FeatureBuilder",
    "FeatureSpec",
    "PortfolioPolicy",
    "PolicyConfig",
    "annualised_sortino",
    "combine_sortino_and_return",
    "load_daily_metrics",
    "load_trade_window_metrics",
    "load_current_symbols",
    "split_by_asset_class",
    "ChronosForecastGenerator",
    "ForecastGenerationConfig",
]
