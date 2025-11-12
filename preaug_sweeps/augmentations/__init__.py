"""Pre-augmentation strategies for time series forecasting."""

from .base_augmentation import BaseAugmentation
from .strategies import (
    NoAugmentation,
    PercentChangeAugmentation,
    LogReturnsAugmentation,
    DifferencingAugmentation,
    DetrendingAugmentation,
    RobustScalingAugmentation,
    MinMaxStandardAugmentation,
    RollingWindowNormalization,
    AUGMENTATION_REGISTRY,
    get_augmentation,
)

__all__ = [
    "BaseAugmentation",
    "NoAugmentation",
    "PercentChangeAugmentation",
    "LogReturnsAugmentation",
    "DifferencingAugmentation",
    "DetrendingAugmentation",
    "RobustScalingAugmentation",
    "MinMaxStandardAugmentation",
    "RollingWindowNormalization",
    "AUGMENTATION_REGISTRY",
    "get_augmentation",
]
