from __future__ import annotations

"""
Differentiable utility primitives for time-series encoding, risk-aware objectives,
and trade-state recurrences used across differentiable_market experiments.
"""

from .core import (
    TradeMemoryState,
    augment_market_features,
    haar_wavelet_pyramid,
    risk_budget_mismatch,
    soft_drawdown,
    taylor_time_encoding,
    trade_memory_update,
)

__all__ = [
    "TradeMemoryState",
    "taylor_time_encoding",
    "haar_wavelet_pyramid",
    "soft_drawdown",
    "risk_budget_mismatch",
    "augment_market_features",
    "trade_memory_update",
]
