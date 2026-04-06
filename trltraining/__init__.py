"""TRL training helpers for trading-plan fine-tuning and GRPO research."""

from .config import MODEL_PRESETS, RECOMMENDED_MODEL_PRESET, TRLTradingConfig
from .dataset import DatasetBundle, build_dataset_bundle
from .methods import TrainerRecommendation, recommend_trainer
from .train_grpo import build_grpo_kwargs


__all__ = [
    "MODEL_PRESETS",
    "RECOMMENDED_MODEL_PRESET",
    "DatasetBundle",
    "TRLTradingConfig",
    "TrainerRecommendation",
    "build_dataset_bundle",
    "build_grpo_kwargs",
    "recommend_trainer",
]
