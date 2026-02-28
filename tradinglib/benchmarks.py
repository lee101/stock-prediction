from __future__ import annotations

from typing import Iterable

import numpy as np

from src.metrics_utils import compute_step_returns


def buy_and_hold_returns(prices: Iterable[float]) -> np.ndarray:
    """Returns for a buy-and-hold strategy over a price series."""
    series = np.asarray(list(prices), dtype=np.float64)
    return compute_step_returns(series)


def oracle_long_flat_returns(
    returns: Iterable[float],
    *,
    trade_cost_bps: float = 0.0,
) -> np.ndarray:
    """Upper-bound long/flat returns (no leverage), optionally with switch costs.

    Uses a simple per-switch cost in basis points applied when position changes.
    This is a coarse upper bound for sanity checking.
    """
    arr = np.asarray(list(returns), dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.float64)

    cost = float(trade_cost_bps) / 10000.0
    position = 0
    adjusted = []
    for step in arr:
        desired = 1 if step > 0 else 0
        if desired != position:
            step = step - cost
        position = desired
        adjusted.append(max(step, 0.0) if position == 1 else 0.0)
    return np.asarray(adjusted, dtype=np.float64)


def oracle_long_short_returns(returns: Iterable[float]) -> np.ndarray:
    """Upper-bound long/short returns (flip each step), ignoring costs."""
    arr = np.asarray(list(returns), dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    return np.abs(arr)


__all__ = [
    "buy_and_hold_returns",
    "oracle_long_flat_returns",
    "oracle_long_short_returns",
]
