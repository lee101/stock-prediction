"""
GymRL package: tooling for reinforcement-learning experiments on the stock bot.

The package bundles feature engineering utilities, the Gymnasium environment,
baseline behaviour policies, and offline dataset helpers that mirror the live
trading pipeline (Toto/Kronos forecasts feeding an allocator).
"""

from .config import FeatureBuilderConfig, OfflineDatasetConfig, PortfolioEnvConfig
from .feature_pipeline import FeatureBuilder, FeatureCube
from .portfolio_env import PortfolioEnv
from .regime_filters import RegimeGuard
from .wrappers import ObservationNormalizer, NormalizerConfig
from .behaviour import topk_equal_weight, kelly_fractional, blend_policies
from .offline_dataset import build_offline_dataset, generate_behaviour_weights
from .differentiable_utils import (
    LossShutdownParams,
    LossShutdownState,
    compute_step_net_return,
    loss_shutdown_adjust,
    update_loss_shutdown_state,
)

__all__ = [
    "FeatureBuilderConfig",
    "OfflineDatasetConfig",
    "PortfolioEnvConfig",
    "FeatureBuilder",
    "FeatureCube",
    "PortfolioEnv",
    "RegimeGuard",
    "ObservationNormalizer",
    "NormalizerConfig",
    "topk_equal_weight",
    "kelly_fractional",
    "blend_policies",
    "build_offline_dataset",
    "generate_behaviour_weights",
    "LossShutdownParams",
    "LossShutdownState",
    "compute_step_net_return",
    "loss_shutdown_adjust",
    "update_loss_shutdown_state",
]
