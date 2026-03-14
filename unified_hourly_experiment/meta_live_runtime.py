"""Runtime helpers for live per-symbol meta strategy selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from unified_hourly_experiment.meta_selector import select_daily_winners


def choose_latest_winner(
    daily_returns_by_strategy: Mapping[str, pd.Series],
    *,
    lookback_days: int,
    metric: str,
    fallback_strategy: str,
    tie_break_order: Sequence[str] | None = None,
    sit_out_threshold: float | None = None,
    selection_mode: str = "winner",
    switch_margin: float = 0.0,
    min_score_gap: float = 0.0,
    recency_halflife_days: float | None = None,
) -> str | None:
    """Pick winner for the most recent day available in returns history."""
    winners = select_daily_winners(
        daily_returns_by_strategy,
        lookback_days=lookback_days,
        metric=metric,
        fallback_strategy=fallback_strategy,
        tie_break_order=tie_break_order,
        require_full_window=True,
        sit_out_threshold=sit_out_threshold,
        selection_mode=selection_mode,
        switch_margin=switch_margin,
        min_score_gap=min_score_gap,
        recency_halflife_days=recency_halflife_days,
    )
    if winners.empty:
        return fallback_strategy
    winner = winners.iloc[-1]
    if winner is None:
        return None
    if isinstance(winner, float) and np.isnan(winner):
        return None
    return str(winner)


def compute_symbol_edge(
    *,
    symbol: str,
    action: Mapping[str, Any],
    fee_rate: float,
    short_only_symbols: Sequence[str],
    entry_reference_price: float | None = None,
) -> float:
    """Compute edge score used for live entry filtering."""
    symbol_u = str(symbol).upper()
    is_short = symbol_u in {str(s).upper() for s in short_only_symbols}

    buy_price = _to_float(action.get("buy_price"))
    sell_price = _to_float(action.get("sell_price"))
    pred_high = _to_float(action.get("predicted_high"))
    pred_low = _to_float(action.get("predicted_low"))
    reference_price = _to_float(entry_reference_price)
    if is_short:
        effective_entry_price = reference_price if reference_price > 0 else sell_price
        if effective_entry_price <= 0:
            return float("-inf")
        return (effective_entry_price - pred_low) / effective_entry_price - fee_rate
    effective_entry_price = reference_price if reference_price > 0 else buy_price
    if effective_entry_price <= 0:
        return float("-inf")
    return (pred_high - effective_entry_price) / effective_entry_price - fee_rate


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
