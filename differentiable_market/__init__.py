"""
Differentiable market training package.

This package provides an end-to-end differentiable OHLC market simulator,
policies, and training utilities for reinforcement learning based trading.
"""

from .config import DataConfig, EnvironmentConfig, TrainingConfig, EvaluationConfig
from .trainer import DifferentiableMarketTrainer
from .policy import DirichletGRUPolicy
from .env import DifferentiableMarketEnv
from .optim import CombinedOptimizer, MuonConfig, build_muon_optimizer

__all__ = [
    "DataConfig",
    "EnvironmentConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "DifferentiableMarketTrainer",
    "DirichletGRUPolicy",
    "DifferentiableMarketEnv",
    "CombinedOptimizer",
    "MuonConfig",
    "build_muon_optimizer",
]
