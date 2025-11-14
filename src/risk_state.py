from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Optional

from jsonshelve import FlatShelf

from src.date_utils import NEW_YORK
from stock.state import ensure_state_dir, get_state_file, resolve_state_suffix


logger = logging.getLogger(__name__)

STATE_SUFFIX = resolve_state_suffix()
RISK_STATE_FILE = get_state_file("global_risk_state", STATE_SUFFIX)
_STORE_KEY = "__global__"

_risk_store: Optional[FlatShelf] = None


@dataclass(frozen=True)
class ProbeState:
    force_probe: bool
    reason: Optional[str]
    probe_date: Optional[date]
    state: Dict[str, object]


def _to_new_york_date(timestamp: Optional[datetime] = None) -> date:
    base = timestamp or datetime.now(timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return base.astimezone(NEW_YORK).date()


def _next_trading_day(start: date) -> date:
    candidate = start + timedelta(days=1)
    while candidate.weekday() >= 5:  # Skip Saturday/Sunday
        candidate += timedelta(days=1)
    return candidate


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        year, month, day = map(int, value.split("-"))
        return date(year, month, day)
    except Exception:
        logger.warning("Unable to parse probe date %r from risk state", value)
    return None


def _ensure_store() -> Optional[FlatShelf]:
    global _risk_store
    if _risk_store is not None:
        return _risk_store
    try:
        ensure_state_dir()
        _risk_store = FlatShelf(str(RISK_STATE_FILE))
    except Exception as exc:
        logger.error("Failed to initialise global risk store: %s", exc)
        return None
    return _risk_store


def _load_state() -> Dict[str, object]:
    store = _ensure_store()
    if store is None:
        return {}
    try:
        store.load()
    except Exception as exc:
        logger.error("Failed loading global risk store: %s", exc)
        return {}
    entry = store.get(_STORE_KEY, {})
    if not isinstance(entry, dict):
        return {}
    return dict(entry)


def _save_state(state: Dict[str, object]) -> None:
    store = _ensure_store()
    if store is None:
        return
    try:
        store.load()
    except Exception as exc:
        logger.error("Failed refreshing global risk store before save: %s", exc)
        return
    store[_STORE_KEY] = dict(state)


def record_day_pl(day_pl: Optional[float], observed_at: Optional[datetime] = None) -> Dict[str, object]:
    """Persist the most recent account day PnL and derive probe scheduling.

    Args:
        day_pl: Realised/unrealised day PnL reported by the broker. When None, state is unchanged.
        observed_at: Timestamp for when the snapshot was taken. Defaults to now (UTC).
    """
    if day_pl is None:
        return _load_state()

    state = _load_state()
    observed_ts = observed_at or datetime.now(timezone.utc)
    observed_date = _to_new_york_date(observed_ts)
    state["last_day_pl"] = float(day_pl)
    state["last_day_date"] = observed_date.isoformat()
    state["updated_at"] = observed_ts.replace(tzinfo=timezone.utc).isoformat()

    if float(day_pl) < 0.0:
        probe_date = _next_trading_day(observed_date)
        state["probe_only_date"] = probe_date.isoformat()
        state["probe_reason"] = f"Previous day loss {day_pl:.2f}"
    else:
        probe_str = state.get("probe_only_date")
        probe_date = _parse_date(probe_str)
        if probe_date is not None and probe_date <= observed_date:
            state.pop("probe_only_date", None)
            state.pop("probe_reason", None)

    _save_state(state)
    return state


def resolve_probe_state(now: Optional[datetime] = None) -> ProbeState:
    """Return the active probe requirement derived from account-level losses."""
    state = _load_state()
    probe_str = state.get("probe_only_date")
    probe_date = _parse_date(probe_str)
    current_date = _to_new_york_date(now)

    if probe_date is None:
        if probe_str:
            state.pop("probe_only_date", None)
            state.pop("probe_reason", None)
            _save_state(state)
        return ProbeState(False, None, None, state)

    if current_date == probe_date:
        return ProbeState(True, state.get("probe_reason"), probe_date, state)

    if current_date > probe_date:
        state.pop("probe_only_date", None)
        state.pop("probe_reason", None)
        _save_state(state)

    return ProbeState(False, None, probe_date, state)


__all__ = [
    "ProbeState",
    "record_day_pl",
    "resolve_probe_state",
]
