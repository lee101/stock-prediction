"""Cooldown management utilities for trading."""

from datetime import datetime
from typing import Dict, Optional

from src.trade_stock_state_utils import parse_timestamp


# Global cooldown state
_COOLDOWN_STATE: Dict[str, Dict] = {}


def record_loss_timestamp(symbol: str, closed_at: Optional[str], *, logger=None) -> None:
    """Record a loss timestamp to trigger cooldown for a symbol.

    Args:
        symbol: Symbol to record cooldown for
        closed_at: ISO format timestamp string when position was closed
        logger: Optional logger for warnings
    """
    if not closed_at:
        return
    ts = parse_timestamp(closed_at, logger=logger)
    if ts:
        _COOLDOWN_STATE[symbol] = {"last_stop_time": ts}


def clear_cooldown(symbol: str) -> None:
    """Clear cooldown state for a symbol.

    Args:
        symbol: Symbol to clear cooldown for
    """
    _COOLDOWN_STATE.pop(symbol, None)


def can_trade_now(
    symbol: str,
    now: datetime,
    min_cooldown_minutes: int,
    symbol_min_cooldown_fn=None,
) -> bool:
    """Check if enough time has passed since last stop to allow trading.

    Args:
        symbol: Symbol to check
        now: Current datetime
        min_cooldown_minutes: Default minimum cooldown in minutes
        symbol_min_cooldown_fn: Optional function to get symbol-specific cooldown

    Returns:
        True if trading is allowed, False if in cooldown period
    """
    # Allow symbol-specific override
    if symbol_min_cooldown_fn is not None:
        override_minutes = symbol_min_cooldown_fn(symbol)
        if override_minutes is not None and override_minutes >= 0:
            min_cooldown_minutes = float(override_minutes)

    state = _COOLDOWN_STATE.get(symbol)
    if not state:
        return True

    last_stop = state.get("last_stop_time")
    if isinstance(last_stop, datetime):
        delta = now - last_stop
        if delta.total_seconds() < min_cooldown_minutes * 60:
            return False
    return True


def get_cooldown_state() -> Dict[str, Dict]:
    """Get the current cooldown state (for testing/debugging).

    Returns:
        Copy of the cooldown state dictionary
    """
    return _COOLDOWN_STATE.copy()
