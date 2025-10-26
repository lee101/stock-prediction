"""
Differentiable market training package.

This package provides an end-to-end differentiable OHLC market simulator,
policies, and training utilities for reinforcement learning based trading.
"""

from .config import DataConfig, EnvironmentConfig, TrainingConfig, EvaluationConfig
from .policy import DirichletGRUPolicy
from .trainer import DifferentiableMarketTrainer
from .env import DifferentiableMarketEnv
from .optim import CombinedOptimizer, MuonConfig, build_muon_optimizer
from .differentiable_utils import (
    TradeMemoryState,
    haar_wavelet_pyramid,
    risk_budget_mismatch,
    soft_drawdown,
    taylor_time_encoding,
    trade_memory_update,
)

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
    "taylor_time_encoding",
    "haar_wavelet_pyramid",
    "soft_drawdown",
    "risk_budget_mismatch",
    "TradeMemoryState",
    "trade_memory_update",
]
