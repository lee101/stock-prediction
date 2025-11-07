from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Mapping, Optional

import pytz

from jsonshelve import FlatShelf
from stock.data_utils import ensure_lower_bound
from stock.state_utils import STATE_KEY_SEPARATOR

StoreLoader = Callable[[], Optional[FlatShelf]]
LoggerLike = Optional[logging.Logger]


def normalize_side_for_key(side: str) -> str:
    normalized = str(side or "").lower()
    if "short" in normalized or "sell" in normalized:
        return "sell"
    return "buy"


def state_key(symbol: str, side: str, strategy: Optional[str] = None, *, separator: str = STATE_KEY_SEPARATOR) -> str:
    """Generate a state key for symbol, side, and optionally strategy.

    If strategy is provided, key format is: symbol|side|strategy
    Otherwise: symbol|side (for backwards compatibility)
    """
    base_key = f"{symbol}{separator}{normalize_side_for_key(side)}"
    if strategy:
        return f"{base_key}{separator}{strategy}"
    return base_key


def parse_timestamp(ts: Optional[str], *, logger: LoggerLike = None) -> Optional[datetime]:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            if logger is not None:
                logger.warning("Unable to parse timestamp %r from trade outcomes store", ts)
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_store_entry(
    store_loader: StoreLoader,
    symbol: str,
    side: str,
    *,
    strategy: Optional[str] = None,
    store_name: str,
    logger: LoggerLike = None,
) -> Dict[str, Any]:
    store = store_loader()
    if store is None:
        return {}
    try:
        store.load()
    except Exception as exc:
        if logger is not None:
            logger.error(f"Failed loading {store_name} store: {exc}")
        return {}
    return store.get(state_key(symbol, side, strategy), {})


def save_store_entry(
    store_loader: StoreLoader,
    symbol: str,
    side: str,
    state: Mapping[str, Any],
    *,
    strategy: Optional[str] = None,
    store_name: str,
    logger: LoggerLike = None,
) -> None:
    store = store_loader()
    if store is None:
        return
    try:
        store.load()
    except Exception as exc:
        if logger is not None:
            logger.error(f"Failed refreshing {store_name} store before save: {exc}")
        return
    store[state_key(symbol, side, strategy)] = dict(state)


def update_learning_state(
    store_loader: StoreLoader,
    symbol: str,
    side: str,
    updates: Mapping[str, Any],
    *,
    strategy: Optional[str] = None,
    logger: LoggerLike = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    current = dict(
        load_store_entry(
            store_loader,
            symbol,
            side,
            strategy=strategy,
            store_name="trade learning",
            logger=logger,
        )
    )
    changed = False
    for key, value in updates.items():
        if current.get(key) != value:
            current[key] = value
            changed = True
    if changed:
        stamp = (now or datetime.now(timezone.utc)).isoformat()
        current["updated_at"] = stamp
        save_store_entry(
            store_loader,
            symbol,
            side,
            current,
            strategy=strategy,
            store_name="trade learning",
            logger=logger,
        )
    return current


def mark_probe_pending(
    store_loader: StoreLoader,
    symbol: str,
    side: str,
    *,
    strategy: Optional[str] = None,
    logger: LoggerLike = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    return update_learning_state(
        store_loader,
        symbol,
        side,
        {
            "pending_probe": True,
            "probe_active": False,
            "last_probe_successful": False,
        },
        strategy=strategy,
        logger=logger,
        now=now,
    )


def mark_probe_active(
    store_loader: StoreLoader,
    symbol: str,
    side: str,
    qty: float,
    *,
    strategy: Optional[str] = None,
    logger: LoggerLike = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    return update_learning_state(
        store_loader,
        symbol,
        side,
        {
            "pending_probe": False,
            "probe_active": True,
            "last_probe_qty": qty,
            "probe_started_at": stamp,
        },
        strategy=strategy,
        logger=logger,
        now=now,
    )


def mark_probe_completed(
    store_loader: StoreLoader,
    symbol: str,
    side: str,
    successful: bool,
    *,
    strategy: Optional[str] = None,
    logger: LoggerLike = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    return update_learning_state(
        store_loader,
        symbol,
        side,
        {
            "pending_probe": not successful,
            "probe_active": False,
            "last_probe_completed_at": stamp,
            "last_probe_successful": successful,
        },
        strategy=strategy,
        logger=logger,
        now=now,
    )


def mark_probe_transitioned(
    store_loader: StoreLoader,
    symbol: str,
    side: str,
    qty: float,
    *,
    strategy: Optional[str] = None,
    logger: LoggerLike = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    return update_learning_state(
        store_loader,
        symbol,
        side,
        {
            "pending_probe": False,
            "probe_active": False,
            "last_probe_successful": False,
            "probe_transitioned_at": stamp,
            "last_probe_transition_qty": qty,
        },
        strategy=strategy,
        logger=logger,
        now=now,
    )


def describe_probe_state(
    learning_state: Optional[Mapping[str, Any]],
    *,
    now: Optional[datetime] = None,
    probe_max_duration: timedelta,
    timezone_name: str = "US/Eastern",
) -> Dict[str, Optional[Any]]:
    if learning_state is None:
        learning_state = {}
    now = now or datetime.now(timezone.utc)
    probe_active = bool(learning_state.get("probe_active"))
    probe_started_at = parse_timestamp(learning_state.get("probe_started_at"))
    summary: Dict[str, Optional[Any]] = {
        "probe_active": probe_active,
        "probe_started_at": probe_started_at.isoformat() if probe_started_at else None,
        "probe_age_seconds": None,
        "probe_expires_at": None,
        "probe_expired": False,
        "probe_transition_ready": False,
    }
    if not probe_active or probe_started_at is None:
        return summary

    probe_age = now - probe_started_at
    summary["probe_age_seconds"] = ensure_lower_bound(probe_age.total_seconds(), 0.0)
    expires_at = probe_started_at + probe_max_duration
    summary["probe_expires_at"] = expires_at.isoformat()
    summary["probe_expired"] = now >= expires_at

    est = pytz.timezone(timezone_name)
    now_est = now.astimezone(est)
    started_est = probe_started_at.astimezone(est)
    summary["probe_transition_ready"] = now_est.date() > started_est.date()
    return summary


def update_active_trade_record(
    store_loader: StoreLoader,
    symbol: str,
    side: str,
    *,
    mode: str,
    qty: float,
    strategy: Optional[str] = None,
    opened_at_sim: Optional[str] = None,
    logger: LoggerLike = None,
    now: Optional[datetime] = None,
) -> None:
    record: Dict[str, Any] = {
        "mode": mode,
        "qty": qty,
        "opened_at": (now or datetime.now(timezone.utc)).isoformat(),
    }
    if opened_at_sim:
        record["opened_at_sim"] = opened_at_sim
    if strategy:
        record["entry_strategy"] = strategy
    save_store_entry(
        store_loader,
        symbol,
        side,
        record,
        store_name="active trades",
        logger=logger,
    )


def tag_active_trade_strategy(
    store_loader: StoreLoader,
    symbol: str,
    side: str,
    strategy: Optional[str],
    *,
    logger: LoggerLike = None,
) -> None:
    if not strategy:
        return
    record = dict(
        load_store_entry(
            store_loader,
            symbol,
            side,
            store_name="active trades",
            logger=logger,
        )
    )
    if not record:
        return
    if record.get("entry_strategy") == strategy:
        return
    record["entry_strategy"] = strategy
    save_store_entry(
        store_loader,
        symbol,
        side,
        record,
        store_name="active trades",
        logger=logger,
    )


def get_active_trade_record(
    store_loader: StoreLoader,
    symbol: str,
    side: str,
    *,
    logger: LoggerLike = None,
) -> Dict[str, Any]:
    return dict(
        load_store_entry(
            store_loader,
            symbol,
            side,
            store_name="active trades",
            logger=logger,
        )
    )


def pop_active_trade_record(
    store_loader: StoreLoader,
    symbol: str,
    side: str,
    *,
    strategy: Optional[str] = None,
    logger: LoggerLike = None,
) -> Dict[str, Any]:
    store = store_loader()
    if store is None:
        return {}
    try:
        store.load()
    except Exception as exc:
        if logger is not None:
            logger.error(f"Failed loading active trades store for pop: {exc}")
        return {}
    key = state_key(symbol, side, strategy)
    record = store.data.pop(key, None) if hasattr(store, "data") else store.pop(key, None)
    if record is None:
        record = {}
    return dict(record)


__all__ = [
    "describe_probe_state",
    "get_active_trade_record",
    "load_store_entry",
    "mark_probe_active",
    "mark_probe_completed",
    "mark_probe_pending",
    "mark_probe_transitioned",
    "normalize_side_for_key",
    "parse_timestamp",
    "pop_active_trade_record",
    "save_store_entry",
    "state_key",
    "tag_active_trade_strategy",
    "update_active_trade_record",
    "update_learning_state",
]
