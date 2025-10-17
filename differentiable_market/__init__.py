"""
Differentiable market training package.

This package provides an end-to-end differentiable OHLC market simulator,
policies, and training utilities for reinforcement learning based trading.
"""

from .config import DataConfig, EnvironmentConfig, TrainingConfig, EvaluationConfig

__all__ = [
    "DataConfig",
    "EnvironmentConfig",
    "TrainingConfig",
    "EvaluationConfig",
]

