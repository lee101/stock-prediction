"""Utility helpers extracted from `trade_stock_e2e` for gating and trend lookups."""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

from src.trade_stock_env_utils import TRUTHY_ENV_VALUES, _load_trend_summary
from src.trade_stock_utils import compute_spread_bps, expected_cost_bps

__all__ = [
    "coerce_positive_int",
    "should_skip_closed_equity",
    "get_trend_stat",
    "is_kronos_only_mode",
    "is_tradeable",
    "pass_edge_threshold",
    "resolve_signal_sign",
    "CONSENSUS_MIN_MOVE_PCT",
    "DISABLE_TRADE_GATES",
]

_TRUTHY = TRUTHY_ENV_VALUES


def coerce_positive_int(raw_value: Optional[str], default: int) -> int:
    """Best-effort coercion of environment values into non-negative integers."""
    if raw_value is None:
        return default
    try:
        parsed = int(str(raw_value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


DISABLE_TRADE_GATES = os.getenv("MARKETSIM_DISABLE_GATES", "0").strip().lower() in _TRUTHY


def should_skip_closed_equity() -> bool:
    """Determine if closed equity positions should be skipped by default."""
    env_value = os.getenv("MARKETSIM_SKIP_CLOSED_EQUITY")
    if env_value is not None:
        return env_value.strip().lower() in _TRUTHY
    return True


def get_trend_stat(symbol: str, key: str) -> Optional[float]:
    """Look up a trend summary metric for the provided symbol."""
    summary: Dict[str, Dict[str, object]] = _load_trend_summary()
    if not summary:
        return None
    symbol_info = summary.get((symbol or "").upper())
    if not symbol_info:
        return None
    value = symbol_info.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


CONSENSUS_MIN_MOVE_PCT = float(os.getenv("CONSENSUS_MIN_MOVE_PCT", "0.001"))


def is_kronos_only_mode() -> bool:
    """Whether we are forcing Kronos-only trading mode based on environment flags."""
    return os.getenv("MARKETSIM_FORCE_KRONOS", "0").strip().lower() in _TRUTHY


def is_tradeable(
    symbol: str,
    bid: Optional[float],
    ask: Optional[float],
    *,
    avg_dollar_vol: Optional[float] = None,
    atr_pct: Optional[float] = None,
) -> Tuple[bool, str]:
    """Basic market microstructure gate checking spread, optional volume and ATR."""
    spread_bps = compute_spread_bps(bid, ask)
    if DISABLE_TRADE_GATES:
        return True, f"Gates disabled (spread {spread_bps:.1f}bps)"
    if math.isinf(spread_bps):
        return False, "Missing bid/ask quote"
    atr_note = f", ATR {atr_pct:.2f}%" if atr_pct is not None else ""
    return True, f"Spread {spread_bps:.1f}bps OK (gates relaxed{atr_note})"


def pass_edge_threshold(symbol: str, expected_move_pct: float) -> Tuple[bool, str]:
    """Check whether the expected edge clears dynamic thresholds and trading costs."""
    move_bps = abs(expected_move_pct) * 1e4
    if DISABLE_TRADE_GATES:
        return True, f"Edge gating disabled ({move_bps:.1f}bps)"
    kronos_only = is_kronos_only_mode()
    base_min = 40.0 if symbol.endswith("USD") else 15.0
    if kronos_only:
        base_min *= 0.6
    min_abs_move_bps = base_min
    buffer = 10.0 if not kronos_only else 5.0
    need = max(expected_cost_bps(symbol) + buffer, min_abs_move_bps)
    if move_bps < need:
        return False, f"Edge {move_bps:.1f}bps < need {need:.1f}bps"
    return True, f"Edge {move_bps:.1f}bps \u2265 need {need:.1f}bps"


def resolve_signal_sign(move_pct: float) -> int:
    """Translate a consensus move percentage into a trading direction."""
    threshold = CONSENSUS_MIN_MOVE_PCT
    if is_kronos_only_mode():
        threshold *= 0.25
    if abs(move_pct) < threshold:
        return 0
    return 1 if move_pct > 0 else -1
