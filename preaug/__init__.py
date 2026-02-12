"""Pre-augmentation and inference-time ensemble strategies."""
from .base import BaseAugmentation
from .strategies import (
    NoAugmentation,
    PercentChangeAugmentation,
    LogReturnsAugmentation,
    DifferencingAugmentation,
    RobustScalingAugmentation,
    AUGMENTATION_REGISTRY,
    get_augmentation,
)
from .inference_strategies import (
    BaseInferenceStrategy,
    SingleInference,
    TemporalDilationEnsemble,
    INFERENCE_STRATEGY_REGISTRY,
    get_inference_strategy,
)

__all__ = [
    "BaseAugmentation",
    "NoAugmentation",
    "PercentChangeAugmentation",
    "LogReturnsAugmentation",
    "DifferencingAugmentation",
    "RobustScalingAugmentation",
    "AUGMENTATION_REGISTRY",
    "get_augmentation",
    "BaseInferenceStrategy",
    "SingleInference",
    "TemporalDilationEnsemble",
    "INFERENCE_STRATEGY_REGISTRY",
    "get_inference_strategy",
]
