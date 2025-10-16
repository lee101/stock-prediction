"""
GymRL package: tooling for reinforcement-learning experiments on the stock bot.

The package bundles feature engineering utilities, the Gymnasium environment,
baseline behaviour policies, and offline dataset helpers that mirror the live
trading pipeline (Toto/Chronos forecasts feeding an allocator).
"""

from .config import FeatureBuilderConfig, OfflineDatasetConfig, PortfolioEnvConfig
from .feature_pipeline import FeatureBuilder, FeatureCube
from .portfolio_env import PortfolioEnv
from .wrappers import ObservationNormalizer, NormalizerConfig
from .behaviour import topk_equal_weight, kelly_fractional, blend_policies
from .offline_dataset import build_offline_dataset, generate_behaviour_weights

__all__ = [
    "FeatureBuilderConfig",
    "OfflineDatasetConfig",
    "PortfolioEnvConfig",
    "FeatureBuilder",
    "FeatureCube",
    "PortfolioEnv",
    "ObservationNormalizer",
    "NormalizerConfig",
    "topk_equal_weight",
    "kelly_fractional",
    "blend_policies",
    "build_offline_dataset",
    "generate_behaviour_weights",
]
