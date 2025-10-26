"""Shared helpers for simulator automation limits and thresholds."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

EntryLimitKey = Tuple[Optional[str], Optional[str]]


def parse_trade_limit_map(raw: Optional[str], *, verbose: bool = True) -> Dict[str, float]:
    """Parse map strings like 'NVDA@ci_guard:10,AAPL:22' into symbol→limit."""
    overrides: Dict[str, float] = {}
    if not raw:
        return overrides
    for item in raw.split(","):
        entry = item.strip()
        if not entry:
            continue
        if ":" not in entry:
            if verbose:
                print(f"[warn] Ignoring malformed max-trades entry (missing ':'): {entry}")
            continue
        key, value_str = entry.split(":", 1)
        key = key.strip()
        value_str = value_str.strip()
        if not key or not value_str:
            if verbose:
                print(f"[warn] Ignoring malformed max-trades entry: {entry}")
            continue
        try:
            value = float(value_str)
        except ValueError:
            if verbose:
                print(f"[warn] Ignoring max-trades entry with non-numeric value: {entry}")
            continue
        symbol_part = key.split("@", 1)[0].strip()
        if not symbol_part:
            if verbose:
                print(f"[warn] Ignoring max-trades entry without symbol: {entry}")
            continue
        if symbol_part.upper() != symbol_part:
            if verbose:
                print(f"[info] Skipping max-trades entry that does not resemble a symbol: {entry}")
            continue
        overrides[symbol_part] = value
    return overrides


def parse_entry_limit_map(raw: Optional[str]) -> Dict[EntryLimitKey, int]:
    """Parse MARKETSIM_SYMBOL_MAX_ENTRIES_MAP style strings into (symbol,strategy)→limit."""
    parsed: Dict[EntryLimitKey, int] = {}
    if not raw:
        return parsed
    for item in raw.split(","):
        entry = item.strip()
        if not entry or ":" not in entry:
            continue
        key_raw, value_raw = entry.split(":", 1)
        key_raw = key_raw.strip()
        value_raw = value_raw.strip()
        if not key_raw or not value_raw:
            continue
        symbol_key: Optional[str] = None
        strategy_key: Optional[str] = None
        if "@" in key_raw:
            sym_raw, strat_raw = key_raw.split("@", 1)
            symbol_key = sym_raw.strip().lower() or None
            strategy_key = strat_raw.strip().lower() or None
        else:
            symbol_key = key_raw.strip().lower() or None
        try:
            parsed[(symbol_key, strategy_key)] = int(float(value_raw))
        except ValueError:
            continue
    return parsed


def resolve_entry_limit(
    parsed: Dict[EntryLimitKey, int], symbol: Optional[str], strategy: Optional[str] = None
) -> Optional[int]:
    """Resolve entry limit using the same precedence as trade_stock_e2e."""
    if not parsed:
        return None
    symbol_key = symbol.lower() if symbol else None
    strategy_key = strategy.lower() if strategy else None
    for candidate in (
        (symbol_key, strategy_key),
        (symbol_key, None),
        (None, strategy_key),
        (None, None),
    ):
        if candidate in parsed:
            return parsed[candidate]
    return None


def entry_limit_to_trade_limit(entry_limit: Optional[int]) -> Optional[float]:
    """Convert a per-run entry limit to an approximate trade-count cap."""
    if entry_limit is None:
        return None
    return float(max(entry_limit, 0) * 2)


DEFAULT_MIN_SMA = -1200.0
DEFAULT_MAX_STD = 1400.0
DEFAULT_MAX_FEE_BPS = 25.0
DEFAULT_MAX_AVG_SLIP = 100.0


def apply_trend_threshold_defaults(
    min_sma: Optional[float], max_std: Optional[float]
) -> tuple[float, float]:
    """Fallback to repo-wide defaults when thresholds are unspecified."""
    return (
        DEFAULT_MIN_SMA if min_sma is None else min_sma,
        DEFAULT_MAX_STD if max_std is None else max_std,
    )


def apply_fee_slip_defaults(
    max_fee_bps: Optional[float], max_avg_slip: Optional[float]
) -> tuple[float, float]:
    """Fallback to repo-wide fee/slip defaults when thresholds are unspecified."""
    return (
        DEFAULT_MAX_FEE_BPS if max_fee_bps is None else max_fee_bps,
        DEFAULT_MAX_AVG_SLIP if max_avg_slip is None else max_avg_slip,
    )


__all__ = [
    "parse_trade_limit_map",
    "parse_entry_limit_map",
    "resolve_entry_limit",
    "entry_limit_to_trade_limit",
    "apply_trend_threshold_defaults",
    "apply_fee_slip_defaults",
    "DEFAULT_MIN_SMA",
    "DEFAULT_MAX_STD",
    "DEFAULT_MAX_FEE_BPS",
    "DEFAULT_MAX_AVG_SLIP",
]
