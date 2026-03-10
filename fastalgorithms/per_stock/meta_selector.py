#!/usr/bin/env python3
"""Meta-selector utilities for walk-forward strategy selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_EPS = 1e-10


@dataclass
class MetaSelectorConfig:
    """Configuration for the meta-selector."""
    lookback_hours: int = 168  # 1 week default
    sit_out_if_all_negative: bool = True
    reeval_every_n_hours: int = 24
    initial_cash: float = 10_000.0
    min_window_for_sortino: int = 24  # need at least 24 bars to compute sortino
    selection_method: str = "winner"
    softmax_temperature: float = 0.25
    top_k: int = 0
    min_score: float = 0.0
    periodic_reeval_for_active: bool = False


@dataclass
class MetaSimResult:
    """Results from meta-selector simulation."""
    equity_curve: np.ndarray
    timestamps: np.ndarray
    switch_log: List[Tuple]  # (bar_idx, symbol, sortinos_dict)
    bars_in_cash: int
    total_bars: int
    per_stock_equity: Dict[str, np.ndarray]

    @property
    def total_return(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        return (self.equity_curve[-1] - self.equity_curve[0]) / (self.equity_curve[0] + _EPS)

    @property
    def max_drawdown(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        peak = np.maximum.accumulate(self.equity_curve)
        dd = (peak - self.equity_curve) / (peak + _EPS)
        return float(np.max(dd))

    @property
    def sortino(self) -> float:
        return compute_sortino(self.equity_curve)

    @property
    def num_switches(self) -> int:
        return len(self.switch_log)


def trailing_sortino(equity_arr: np.ndarray, end_idx: int, window: int,
                     min_bars: int = 24) -> float:
    """Compute trailing Sortino ratio from equity curve.

    Adapted from sim_meta_switcher.py:145-155.

    Args:
        equity_arr: Full equity curve array.
        end_idx: Last index to include (inclusive).
        window: Number of bars to look back.
        min_bars: Minimum bars needed; returns 0.0 if fewer.

    Returns:
        Trailing Sortino ratio. 0.0 if insufficient data.
    """
    start = max(0, end_idx - window)
    if end_idx - start < min_bars:
        return 0.0
    sub = equity_arr[start:end_idx + 1]
    rets = np.diff(sub) / (np.abs(sub[:-1]) + _EPS)
    if len(rets) < 2:
        return 0.0
    neg = rets[rets < 0]
    dd = np.std(neg) if len(neg) > 1 else _EPS
    return float(np.mean(rets)) / (dd + _EPS)


def compute_sortino(equity_arr: np.ndarray) -> float:
    """Compute Sortino ratio for full equity curve."""
    if len(equity_arr) < 2:
        return 0.0
    rets = np.diff(equity_arr) / (np.abs(equity_arr[:-1]) + _EPS)
    if len(rets) < 2:
        return 0.0
    neg = rets[rets < 0]
    dd = np.std(neg) if len(neg) > 1 else _EPS
    return float(np.mean(rets)) / (dd + _EPS)


def align_equity_curves(
    per_stock_equity: Dict[str, pd.DataFrame],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Align all stock equity curves to common timestamps.

    Args:
        per_stock_equity: {symbol: DataFrame with columns [timestamp, equity, in_position]}

    Returns:
        (common_timestamps, {symbol: equity_array}, {symbol: in_position_array})
    """
    # Find common timestamps across all stocks
    common_index: Optional[pd.Index] = None
    for df in per_stock_equity.values():
        timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        index = pd.DatetimeIndex(timestamps).dropna().sort_values().unique()
        common_index = index if common_index is None else common_index.intersection(index)

    if common_index is None or common_index.empty:
        raise ValueError("No common timestamps across stocks")

    common_ts_arr = common_index.to_numpy()
    equities = {}
    positions = {}

    for symbol, df in per_stock_equity.items():
        df_sorted = df.copy()
        df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"], utc=True, errors="coerce")
        df_sorted = df_sorted.dropna(subset=["timestamp"]).set_index("timestamp").loc[common_index].reset_index()
        equities[symbol] = df_sorted["equity"].values.astype(np.float64)
        positions[symbol] = df_sorted["in_position"].values.astype(bool)

    return common_ts_arr, equities, positions


def _dominant_symbol(weights: Dict[str, float]) -> str:
    if not weights:
        return "cash"
    return max(weights.items(), key=lambda item: (item[1], item[0]))[0]


def _allocation_changed(
    prev_weights: Dict[str, float],
    new_weights: Dict[str, float],
    *,
    tolerance: float = 1e-6,
) -> bool:
    symbols = set(prev_weights) | set(new_weights)
    return any(abs(prev_weights.get(symbol, 0.0) - new_weights.get(symbol, 0.0)) > tolerance for symbol in symbols)


def _select_allocations(sortinos: Dict[str, float], config: MetaSelectorConfig) -> Dict[str, float]:
    ranked = sorted(sortinos.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return {}

    if config.selection_method == "winner":
        best_symbol, best_score = ranked[0]
        if config.sit_out_if_all_negative and best_score <= float(config.min_score):
            return {}
        return {best_symbol: 1.0}

    if config.selection_method != "softmax":
        raise ValueError(
            f"Unsupported selection_method '{config.selection_method}'. Expected 'winner' or 'softmax'."
        )

    threshold = float(config.min_score)
    eligible = [(symbol, score) for symbol, score in ranked if score > threshold]
    if not eligible:
        if config.sit_out_if_all_negative:
            return {}
        eligible = ranked[:1]

    top_k = int(config.top_k)
    if top_k > 0:
        eligible = eligible[:top_k]

    scores = np.asarray([score for _, score in eligible], dtype=np.float64)
    temperature = max(float(config.softmax_temperature), 1e-6)
    logits = (scores - np.max(scores)) / temperature
    logits = np.clip(logits, -60.0, 60.0)
    weights = np.exp(logits)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        winner, _ = eligible[0]
        return {winner: 1.0}

    normalized = weights / total
    return {
        symbol: float(weight)
        for (symbol, _), weight in zip(eligible, normalized)
        if float(weight) > 1e-8
    }


def run_meta_simulation(
    per_stock_equity: Dict[str, pd.DataFrame],
    config: MetaSelectorConfig,
) -> MetaSimResult:
    """Simulate the meta-selector strategy.

    At each bar:
    1. Compute trailing Sortino for each stock
    2. If active stock exits position -> re-evaluate
    3. If in cash and periodic reeval -> re-evaluate
    4. Pick stock with highest trailing Sortino
    5. If sit_out_if_all_negative and all Sortinos <= 0 -> hold cash
    6. Apply relative return of selected stock to meta equity

    Args:
        per_stock_equity: {symbol: DataFrame[timestamp, equity, in_position]}
        config: Meta-selector configuration.

    Returns:
        MetaSimResult with combined equity curve and diagnostics.
    """
    common_ts, equities, positions = align_equity_curves(per_stock_equity)
    symbols = list(equities.keys())
    n = len(common_ts)

    # Compute per-bar relative returns for each stock
    relatives = {}
    for sym in symbols:
        eq = equities[sym]
        rel = np.ones(n)
        for i in range(1, n):
            rel[i] = eq[i] / (eq[i - 1] + _EPS) if eq[i - 1] > 0 else 1.0
        relatives[sym] = rel

    meta_equity = np.zeros(n)
    meta_equity[0] = config.initial_cash
    switch_log: List[Tuple] = []
    bars_in_cash = 0
    active_weights: Dict[str, float] = {}
    active_symbol = "cash"

    for i in range(1, n):
        should_reeval = not active_weights
        if config.reeval_every_n_hours > 0 and (i % config.reeval_every_n_hours) == 0:
            if config.periodic_reeval_for_active or not active_weights:
                should_reeval = True

        if active_symbol != "cash" and active_symbol in positions and i >= 2:
            was_in = positions[active_symbol][i - 2]
            now_in = positions[active_symbol][i - 1]
            if was_in and not now_in:
                should_reeval = True

        if should_reeval:
            sortinos = {
                sym: trailing_sortino(
                    equities[sym],
                    i - 1,
                    config.lookback_hours,
                    config.min_window_for_sortino,
                )
                for sym in symbols
            }
            new_weights = _select_allocations(sortinos, config)
            new_symbol = _dominant_symbol(new_weights)
            if _allocation_changed(active_weights, new_weights):
                switch_log.append((i, new_symbol, dict(sortinos), dict(new_weights)))
            active_weights = new_weights
            active_symbol = new_symbol

        if not active_weights:
            meta_equity[i] = meta_equity[i - 1]
            bars_in_cash += 1
            continue

        invested = 0.0
        relative_return = 0.0
        any_position = False
        for sym, weight in active_weights.items():
            invested += float(weight)
            relative_return += float(weight) * float(relatives[sym][i])
            any_position = any_position or bool(positions[sym][i - 1])

        cash_weight = max(0.0, 1.0 - invested)
        meta_equity[i] = meta_equity[i - 1] * (relative_return + cash_weight)
        if not any_position:
            bars_in_cash += 1

    return MetaSimResult(
        equity_curve=meta_equity,
        timestamps=common_ts,
        switch_log=switch_log,
        bars_in_cash=bars_in_cash,
        total_bars=n,
        per_stock_equity=equities,
    )


def run_meta_lookback_sweep(
    per_stock_equity: Dict[str, pd.DataFrame],
    lookback_hours_list: List[int] = [24, 48, 72, 168, 336, 720],
    sit_out_if_all_negative: bool = True,
    initial_cash: float = 10_000.0,
) -> pd.DataFrame:
    """Sweep lookback window and return comparison table."""
    rows = []
    for lb in lookback_hours_list:
        config = MetaSelectorConfig(
            lookback_hours=lb,
            sit_out_if_all_negative=sit_out_if_all_negative,
            initial_cash=initial_cash,
        )
        result = run_meta_simulation(per_stock_equity, config)
        rows.append({
            "lookback_hours": lb,
            "total_return_pct": result.total_return * 100,
            "sortino": result.sortino,
            "max_drawdown_pct": result.max_drawdown * 100,
            "num_switches": result.num_switches,
            "bars_in_cash": result.bars_in_cash,
            "final_equity": result.equity_curve[-1],
        })
    return pd.DataFrame(rows)
