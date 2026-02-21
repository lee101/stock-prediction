from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def _to_1d_float_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def compute_return_series(equity_curve: Iterable[float] | np.ndarray) -> np.ndarray:
    """Convert an equity curve into simple period returns."""
    equity = _to_1d_float_array(equity_curve)
    if equity.size < 2:
        return np.asarray([], dtype=np.float64)

    prev = equity[:-1]
    curr = equity[1:]
    returns = np.zeros_like(prev, dtype=np.float64)
    valid = prev != 0.0
    returns[valid] = (curr[valid] / prev[valid]) - 1.0
    return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)


def compute_max_drawdown(equity_curve: Iterable[float] | np.ndarray) -> float:
    """Return max drawdown as a fraction (0.20 == 20% drawdown)."""
    equity = _to_1d_float_array(equity_curve)
    if equity.size == 0:
        return 0.0

    peaks = np.maximum.accumulate(equity)
    drawdowns = np.zeros_like(equity, dtype=np.float64)
    valid = peaks > 0.0
    drawdowns[valid] = (peaks[valid] - equity[valid]) / peaks[valid]
    return float(np.nanmax(np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)))


def compute_pnl_smoothness(returns: Iterable[float] | np.ndarray) -> float:
    """Compute a curve jaggedness proxy: std of return deltas."""
    rets = _to_1d_float_array(returns)
    if rets.size < 2:
        return 0.0
    return float(np.std(np.diff(rets)))


def compute_pnl_smoothness_from_equity(equity_curve: Iterable[float] | np.ndarray) -> float:
    return compute_pnl_smoothness(compute_return_series(equity_curve))


def summarize_lag_results(
    lag_results: Sequence[Mapping[str, Any]],
    *,
    sortino_clip: float = 10.0,
) -> dict[str, float]:
    """Aggregate lag-sweep metrics into a single robust score.

    Per-lag Sortino can explode when downside variance is extremely close to zero.
    We clip Sortino before aggregation so one pathological lag does not dominate
    the robustness score.
    """
    if not lag_results:
        raise ValueError("lag_results must not be empty")

    sortinos_raw = _to_1d_float_array(float(row.get("sortino", 0.0) or 0.0) for row in lag_results)
    if sortino_clip > 0:
        sortinos = np.clip(sortinos_raw, -float(sortino_clip), float(sortino_clip))
    else:
        sortinos = sortinos_raw
    returns = _to_1d_float_array(float(row.get("return_pct", 0.0) or 0.0) for row in lag_results)
    drawdowns = _to_1d_float_array(float(row.get("max_drawdown_pct", 0.0) or 0.0) for row in lag_results)
    smoothness = _to_1d_float_array(float(row.get("pnl_smoothness", 0.0) or 0.0) for row in lag_results)

    sortino_mean = float(np.mean(sortinos))
    sortino_std = float(np.std(sortinos))
    sortino_p10 = float(np.percentile(sortinos, 10))
    return_mean = float(np.mean(returns))
    drawdown_mean = float(np.mean(drawdowns))
    smoothness_mean = float(np.mean(smoothness))

    # Prefer high downside-adjusted performance that remains stable under lag shifts.
    robust_score = (
        sortino_p10
        - 0.75 * sortino_std
        + 0.03 * return_mean
        - 0.08 * drawdown_mean
        - 120.0 * smoothness_mean
    )

    return {
        "lag_count": float(len(lag_results)),
        "sortino_clip": float(sortino_clip),
        "sortino_mean": sortino_mean,
        "sortino_std": sortino_std,
        "sortino_p10": sortino_p10,
        "sortino_mean_raw": float(np.mean(sortinos_raw)),
        "sortino_std_raw": float(np.std(sortinos_raw)),
        "return_mean_pct": return_mean,
        "max_drawdown_mean_pct": drawdown_mean,
        "pnl_smoothness_mean": smoothness_mean,
        "robust_score": float(robust_score),
    }
