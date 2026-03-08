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


def compute_ulcer_index(equity_curve: Iterable[float] | np.ndarray) -> float:
    """Compute ulcer index from an equity curve."""
    equity = _to_1d_float_array(equity_curve)
    if equity.size == 0:
        return 0.0

    peaks = np.maximum.accumulate(equity)
    drawdowns_pct = np.zeros_like(equity, dtype=np.float64)
    valid = peaks > 0.0
    drawdowns_pct[valid] = ((equity[valid] / peaks[valid]) - 1.0) * 100.0
    return float(np.sqrt(np.mean(np.square(drawdowns_pct))))


def compute_trade_rate(
    trade_count: float | int,
    period_count: float | int,
    *,
    min_period_count: int = 24,
) -> float:
    effective_periods = max(int(period_count), int(min_period_count), 1)
    count = max(float(trade_count), 0.0)
    return float(count / effective_periods)


def compute_market_sim_goodness_score(
    *,
    total_return: float,
    sortino: float,
    max_drawdown: float,
    pnl_smoothness: float,
    ulcer_index: float = 0.0,
    trade_rate: float = 0.0,
    period_count: int | float | None = None,
    min_period_count: int = 24,
) -> float:
    """Combine return, downside risk, smoothness, and activity into one score."""
    coverage = 1.0
    if period_count is not None and min_period_count > 0:
        coverage = min(max(float(period_count), 1.0) / float(min_period_count), 1.0)

    smoothness_score = 1.0 / (1.0 + 250.0 * max(float(pnl_smoothness), 0.0))
    raw_score = (
        100.0 * float(total_return)
        + 0.8 * float(sortino)
        + 0.5 * smoothness_score
        - 12.0 * float(max_drawdown)
        - 0.08 * float(ulcer_index)
        - 2.0 * max(float(trade_rate) - (1.0 / max(min_period_count, 1)), 0.0)
    )
    return float(raw_score * coverage)


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
    ulcer_index = _to_1d_float_array(float(row.get("ulcer_index", 0.0) or 0.0) for row in lag_results)
    trade_rate = _to_1d_float_array(float(row.get("trade_rate", 0.0) or 0.0) for row in lag_results)

    sortino_mean = float(np.mean(sortinos))
    sortino_std = float(np.std(sortinos))
    sortino_p10 = float(np.percentile(sortinos, 10))
    return_mean = float(np.mean(returns))
    drawdown_mean = float(np.mean(drawdowns))
    smoothness_mean = float(np.mean(smoothness))
    ulcer_mean = float(np.mean(ulcer_index))
    trade_rate_mean = float(np.mean(trade_rate))

    # Prefer high downside-adjusted performance that remains stable under lag shifts.
    robust_score = (
        sortino_p10
        - 0.75 * sortino_std
        + 0.03 * return_mean
        - 0.08 * drawdown_mean
        - 120.0 * smoothness_mean
        - 0.04 * ulcer_mean
        - 1.5 * trade_rate_mean
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
        "ulcer_index_mean": ulcer_mean,
        "trade_rate_mean": trade_rate_mean,
        "robust_score": float(robust_score),
    }


def summarize_scenario_results(
    scenario_results: Sequence[Mapping[str, Any]],
    *,
    sortino_clip: float = 10.0,
) -> dict[str, float]:
    """Aggregate multi-window / multi-start results into a single robustness score.

    Expected rows contain at least:
    - ``sortino``
    - ``return_pct``
    - ``max_drawdown_pct`` as a positive drawdown magnitude
    - ``pnl_smoothness``

    Optional ``trade_count`` values are summarized when present so callers can
    apply activity floors without coupling that policy into the score itself.
    """
    if not scenario_results:
        raise ValueError("scenario_results must not be empty")

    sortinos_raw = _to_1d_float_array(float(row.get("sortino", 0.0) or 0.0) for row in scenario_results)
    if sortino_clip > 0:
        sortinos = np.clip(sortinos_raw, -float(sortino_clip), float(sortino_clip))
    else:
        sortinos = sortinos_raw
    returns = _to_1d_float_array(float(row.get("return_pct", 0.0) or 0.0) for row in scenario_results)
    annualized_returns = _to_1d_float_array(
        float(row.get("annualized_return_pct", 0.0) or 0.0) for row in scenario_results
    )
    drawdowns = _to_1d_float_array(float(row.get("max_drawdown_pct", 0.0) or 0.0) for row in scenario_results)
    smoothness = _to_1d_float_array(float(row.get("pnl_smoothness", 0.0) or 0.0) for row in scenario_results)
    trade_counts = _to_1d_float_array(float(row.get("trade_count", 0.0) or 0.0) for row in scenario_results)

    sortino_mean = float(np.mean(sortinos))
    sortino_p25 = float(np.percentile(sortinos, 25))
    sortino_worst = float(np.min(sortinos))
    return_mean = float(np.mean(returns))
    return_p25 = float(np.percentile(returns, 25))
    return_worst = float(np.min(returns))
    annualized_return_mean = float(np.mean(annualized_returns))
    annualized_return_p25 = float(np.percentile(annualized_returns, 25))
    annualized_return_worst = float(np.min(annualized_returns))
    negative_return_rate = float(np.mean(returns <= 0.0))
    drawdown_mean = float(np.mean(drawdowns))
    drawdown_worst = float(np.max(drawdowns))
    smoothness_mean = float(np.mean(smoothness))
    smoothness_worst = float(np.max(smoothness))

    # Prefer configurations that remain profitable under different windows and
    # seeded positions, while penalizing jagged equity curves and deeper losses.
    robust_score = (
        1.5 * return_worst
        + 0.75 * return_p25
        + 0.35 * return_mean
        + 2.0 * sortino_p25
        + 0.5 * sortino_worst
        - 0.8 * drawdown_worst
        - 0.25 * drawdown_mean
        - 175.0 * smoothness_mean
        - 50.0 * negative_return_rate
    )

    return {
        "scenario_count": float(len(scenario_results)),
        "sortino_clip": float(sortino_clip),
        "sortino_mean": sortino_mean,
        "sortino_p25": sortino_p25,
        "sortino_worst": sortino_worst,
        "sortino_mean_raw": float(np.mean(sortinos_raw)),
        "return_mean_pct": return_mean,
        "return_p25_pct": return_p25,
        "return_worst_pct": return_worst,
        "annualized_return_mean_pct": annualized_return_mean,
        "annualized_return_p25_pct": annualized_return_p25,
        "annualized_return_worst_pct": annualized_return_worst,
        "negative_return_rate": negative_return_rate,
        "max_drawdown_mean_pct": drawdown_mean,
        "max_drawdown_worst_pct": drawdown_worst,
        "pnl_smoothness_mean": smoothness_mean,
        "pnl_smoothness_worst": smoothness_worst,
        "trade_count_mean": float(np.mean(trade_counts)) if trade_counts.size else 0.0,
        "trade_count_min": float(np.min(trade_counts)) if trade_counts.size else 0.0,
        "robust_score": float(robust_score),
    }
