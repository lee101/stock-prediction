#!/usr/bin/env python3
"""Unified hourly trading bot for stocks (Alpaca) with exit orders, directional constraints, multi-position."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

import pandas as pd
import torch
from loguru import logger

from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer
from binanceneural.config import DatasetConfig
from binanceneural.model import build_policy, policy_config_from_payload
from binanceneural.inference import generate_latest_action
from src.trade_directions import (
    DEFAULT_ALPACA_LIVE8_STOCKS,
    DEFAULT_LONG_ONLY_STOCKS,
    DEFAULT_SHORT_ONLY_STOCKS,
    is_long_only_symbol,
    is_short_only_symbol,
    resolve_trade_directions,
)
from src.hourly_order_reconcile import order_created_at, orders_to_cancel_for_live_symbol
from src.torch_load_utils import torch_load_compat
from src.symbol_utils import is_crypto_symbol
from src.hourly_trader_utils import (
    EntryAllocationCandidate,
    OrderIntent,
    allocate_concentrated_entry_budget,
    entry_intensity_fraction,
    normalize_entry_allocator_mode,
)

LONG_ONLY = set(DEFAULT_LONG_ONLY_STOCKS)
SHORT_ONLY = set(DEFAULT_SHORT_ONLY_STOCKS)

STATE_DIR = Path(os.environ.get("STATE_DIR", "strategy_state")).expanduser()
STATE_FILE = STATE_DIR / "stock_portfolio_state.json"
TRADE_LOG = STATE_DIR / "stock_trade_log.jsonl"
EVENT_LOG = STATE_DIR / "stock_event_log.jsonl"
MAX_HOLD_HOURS = 6
MAX_POSITIONS = 10
TRADE_AMOUNT_SCALE = 100.0
MIN_BUY_AMOUNT = 0.0
ENTRY_INTENSITY_POWER = 1.0
ENTRY_MIN_INTENSITY_FRACTION = 0.0
LONG_INTENSITY_MULTIPLIER = 1.0
SHORT_INTENSITY_MULTIPLIER = 1.0
ENTRY_ALLOCATOR_MODE = "concentrated"
ENTRY_ALLOCATOR_EDGE_POWER = 2.0
ENTRY_ALLOCATOR_MAX_SINGLE_POSITION_FRACTION = 0.6
ENTRY_ALLOCATOR_RESERVE_FRACTION = 0.1
BROKER_EVENT_LOOKBACK_HOURS = 48.0
BROKER_EVENT_KEY_LIMIT = 512
ENTRY_REPLACE_CANCEL_WAIT_SECONDS = 0.75
ENTRY_REPLACE_CANCEL_POLL_SECONDS = 0.25


@dataclass(frozen=True)
class EntryOrderReconcileResult:
    matching_order: Optional[object]
    replacement_blocked: bool = False


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    if hasattr(value, "__dict__"):
        try:
            return {str(k): _json_safe(v) for k, v in vars(value).items()}
        except TypeError:
            pass
    return str(value)


def _write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(_json_safe(payload), default=str) + "\n")
    except Exception:
        pass


def _serialize_order(order: object) -> dict[str, Any]:
    fields = (
        "id",
        "client_order_id",
        "symbol",
        "side",
        "status",
        "type",
        "time_in_force",
        "qty",
        "filled_qty",
        "filled_avg_price",
        "limit_price",
        "stop_price",
        "submitted_at",
        "created_at",
        "updated_at",
        "filled_at",
        "expired_at",
        "canceled_at",
    )
    return {
        field: _json_safe(getattr(order, field, None))
        for field in fields
        if getattr(order, field, None) is not None
    }


def _serialize_position(symbol: str, position: object) -> dict[str, Any]:
    if isinstance(position, dict):
        payload = dict(position)
    else:
        payload = {
            "qty": getattr(position, "qty", None),
            "price": getattr(position, "price", None),
            "current_price": getattr(position, "current_price", None),
            "avg_entry_price": getattr(position, "avg_entry_price", None),
            "market_value": getattr(position, "market_value", None),
            "side": getattr(position, "side", None),
        }
    payload["symbol"] = symbol
    return _json_safe(payload)


def _record_field(record: object, field: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(field, default)
    return getattr(record, field, default)


def _normalize_enum_value(value: object) -> str:
    if value is None:
        return ""
    raw_value = getattr(value, "value", value)
    return str(raw_value).strip().lower()


def _coerce_utc_datetime(value: object) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = pd.Timestamp(value).to_pydatetime()
        except Exception:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _serialize_activity(activity: object) -> dict[str, Any]:
    if isinstance(activity, dict):
        return _json_safe(dict(activity))
    if hasattr(activity, "__dict__"):
        try:
            return _json_safe(vars(activity))
        except TypeError:
            pass
    return {"value": _json_safe(activity)}


def _activity_timestamp(activity: object) -> Optional[datetime]:
    for field in ("transaction_time", "date", "created_at"):
        timestamp = _coerce_utc_datetime(_record_field(activity, field))
        if timestamp is not None:
            return timestamp
    return None


def _order_event_timestamp(order: object) -> Optional[datetime]:
    for field in ("updated_at", "filled_at", "canceled_at", "expired_at", "created_at", "submitted_at"):
        timestamp = _coerce_utc_datetime(_record_field(order, field))
        if timestamp is not None:
            return timestamp
    return None


def _order_event_key(order: object) -> str:
    return "|".join(
        [
            str(_record_field(order, "id", "")),
            _normalize_enum_value(_record_field(order, "status")),
            str(_record_field(order, "filled_qty", "")),
            str(_record_field(order, "filled_avg_price", "")),
            (_order_event_timestamp(order) or datetime.min.replace(tzinfo=timezone.utc)).isoformat(),
        ]
    )


def _activity_event_key(activity: object) -> str:
    return "|".join(
        [
            str(_record_field(activity, "id", _record_field(activity, "activity_id", ""))),
            str(_record_field(activity, "activity_type", _record_field(activity, "type", ""))),
            str(_record_field(activity, "symbol", "")),
            str(_record_field(activity, "side", "")),
            str(_record_field(activity, "qty", "")),
            (_activity_timestamp(activity) or datetime.min.replace(tzinfo=timezone.utc)).isoformat(),
        ]
    )


def _trim_recent_keys(keys: list[str], *, limit: int = BROKER_EVENT_KEY_LIMIT) -> list[str]:
    if len(keys) <= limit:
        return keys
    return keys[-limit:]


def _iter_activity_dates(start: datetime, end: datetime) -> list[str]:
    current = start.date()
    end_date = end.date()
    dates: list[str] = []
    while current <= end_date:
        dates.append(current.isoformat())
        current = current + timedelta(days=1)
    return dates


def _serialize_orders_by_symbol(orders_by_symbol: dict[str, list]) -> dict[str, list[dict[str, Any]]]:
    return {
        str(symbol): [_serialize_order(order) for order in orders]
        for symbol, orders in orders_by_symbol.items()
    }


def _normalize_live_symbol(symbol: object) -> str:
    return str(symbol or "").replace("/", "").replace("-", "").replace("_", "").strip().upper()


def _tracked_state_summary(state: dict[str, Any]) -> dict[str, Any]:
    tracked = state.get("positions", {})
    broker_cursor = state.get("broker_event_cursor", {})
    return {
        "tracked_symbols": sorted(str(symbol) for symbol in tracked.keys()),
        "tracked_count": int(len(tracked)),
        "pending_close": sorted(str(symbol) for symbol in state.get("pending_close", [])),
        "broker_closed_orders_after": broker_cursor.get("closed_orders_after"),
        "broker_fill_activities_after": broker_cursor.get("fill_activities_after"),
    }


def log_event(event_type: str, /, **fields: Any) -> None:
    payload = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "event_type": str(event_type),
        "pid": int(os.getpid()),
        **{key: _json_safe(value) for key, value in fields.items()},
    }
    _write_jsonl(EVENT_LOG, payload)


def log_trade(event: dict):
    payload = dict(event)
    payload["logged_at"] = datetime.now(timezone.utc).isoformat()
    _write_jsonl(TRADE_LOG, payload)
    log_event(
        "trade_log_append",
        trade_event=payload.get("event"),
        payload={key: value for key, value in payload.items() if key != "logged_at"},
    )


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
            log_event("state_loaded", path=str(STATE_FILE), **_tracked_state_summary(state))
            return state
        except Exception as exc:
            log_event("state_load_failed", path=str(STATE_FILE), error=str(exc))
    state = {"positions": {}}
    log_event("state_defaulted", path=str(STATE_FILE), **_tracked_state_summary(state))
    return state


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    log_event("state_saved", path=str(STATE_FILE), **_tracked_state_summary(state))


def load_model(checkpoint_dir: Path, epoch: int = None):
    if epoch is not None:
        best_ckpt = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        if not best_ckpt.exists():
            raise ValueError(f"Checkpoint not found: {best_ckpt}")
    else:
        checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        best_ckpt = checkpoints[-1]
    logger.info("Loading checkpoint: {}", best_ckpt.name)

    ckpt = torch_load_compat(best_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    config_path = checkpoint_dir / "config.json"
    meta_path = checkpoint_dir / "training_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            config = json.load(f)
    elif config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = ckpt.get("config", {})

    feature_columns = config.get("feature_columns") or []
    sequence_length = config.get("sequence_length", 32)

    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    if not feature_columns:
        from binanceneural.data import build_default_feature_columns
        embed_w = state_dict.get("embed.weight") or state_dict.get("_orig_mod.embed.weight")
        if embed_w is not None and embed_w.ndim == 2:
            input_dim = int(embed_w.shape[1])
            for h_try in [[1], [1, 24]]:
                fc = build_default_feature_columns(h_try)
                if len(fc) == input_dim:
                    feature_columns = fc
                    break
        if not feature_columns:
            feature_columns = build_default_feature_columns([1])

    policy_cfg = policy_config_from_payload(config, input_dim=len(feature_columns), state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    normalizer = None
    config_path2 = checkpoint_dir / "config.json"
    if config_path2.exists():
        with open(config_path2) as f:
            cfg2 = json.load(f)
        if "normalizer" in cfg2:
            normalizer = FeatureNormalizer.from_dict(cfg2["normalizer"])
            logger.info("Loaded saved normalizer from config")

    num_outputs = config.get("num_outputs", 4)
    max_hold = config.get("max_hold_hours", MAX_HOLD_HOURS)
    logger.info("Model: {} outputs, max_hold={}h", num_outputs, max_hold)
    return model, feature_columns, sequence_length, normalizer


def is_market_open_now() -> bool:
    """Return True if NYSE is currently open, using Alpaca clock API (handles holidays).
    Falls back to manual time check if Alpaca is unreachable."""
    try:
        import alpaca_wrapper
        clock = alpaca_wrapper.get_clock()
        return bool(clock.is_open)
    except Exception:
        pass
    # Fallback: manual check (no holiday awareness)
    from zoneinfo import ZoneInfo
    ny = datetime.now(ZoneInfo("America/New_York"))
    if ny.weekday() >= 5:
        return False
    t = ny.time()
    from datetime import time as dt_time
    return dt_time(9, 30) <= t <= dt_time(16, 0)


def market_hours_between(start_utc: datetime, end_utc: datetime) -> float:
    """Count market-open hours (9:30-16:00 ET, Mon-Fri) between two UTC datetimes."""
    from zoneinfo import ZoneInfo
    from datetime import time as dt_time
    et = ZoneInfo("America/New_York")
    start_et = start_utc.astimezone(et)
    end_et = end_utc.astimezone(et)
    total_minutes = 0
    current = start_et.replace(second=0, microsecond=0)
    while current < end_et:
        if current.weekday() < 5:
            t = current.time()
            if dt_time(9, 30) <= t < dt_time(16, 0):
                step = min(60, int((end_et - current).total_seconds() / 60))
                total_minutes += max(step, 1)
                current += timedelta(minutes=max(step, 1))
                continue
            if t < dt_time(9, 30):
                current = current.replace(hour=9, minute=30, second=0)
                continue
        next_day = (current + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
        current = next_day
    return total_minutes / 60.0


def calendar_hours_between(start_utc: datetime, end_utc: datetime) -> float:
    """Count wall-clock hours between two datetimes, inclusive of weekends/holidays."""
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    else:
        start_utc = start_utc.astimezone(timezone.utc)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=timezone.utc)
    else:
        end_utc = end_utc.astimezone(timezone.utc)
    return max((end_utc - start_utc).total_seconds(), 0.0) / 3600.0


def get_alpaca_client(paper: bool = True):
    from alpaca.trading.client import TradingClient
    key_id = ALP_KEY_ID if paper else ALP_KEY_ID_PROD
    secret = ALP_SECRET_KEY if paper else ALP_SECRET_KEY_PROD
    return TradingClient(key_id, secret, paper=paper)


def get_current_positions(api) -> Dict[str, dict]:
    positions = {}
    try:
        for pos in api.get_all_positions():
            symbol = _normalize_live_symbol(getattr(pos, "symbol", ""))
            if not symbol:
                continue
            avg_entry_price = _positive_float(getattr(pos, "avg_entry_price", None))
            positions[symbol] = {
                "qty": float(pos.qty),
                "price": float(pos.current_price),
                "avg_entry_price": float(avg_entry_price) if avg_entry_price is not None else None,
            }
        log_event(
            "positions_snapshot",
            positions=[_serialize_position(symbol, payload) for symbol, payload in positions.items()],
            position_count=len(positions),
        )
    except Exception as e:
        logger.error("Failed to get positions: {}", e)
        log_event("positions_snapshot_failed", error=str(e))
    return positions


def get_account_info(api) -> dict:
    try:
        account = api.get_account()
        info = {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
        }
        log_event("account_snapshot", account=info)
        return info
    except Exception as e:
        logger.error("Failed to get account: {}", e)
        log_event("account_snapshot_failed", error=str(e))
        return {"equity": 0, "buying_power": 0, "cash": 0}


def get_open_orders(api) -> Dict[str, list]:
    """Get open orders grouped by symbol."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    orders_by_symbol = {}
    try:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = api.get_orders(request)
        for o in orders:
            sym = _normalize_live_symbol(getattr(o, "symbol", ""))
            if not sym:
                continue
            if sym not in orders_by_symbol:
                orders_by_symbol[sym] = []
            orders_by_symbol[sym].append(o)
        log_event(
            "open_orders_snapshot",
            order_count=sum(len(items) for items in orders_by_symbol.values()),
            orders_by_symbol=_serialize_orders_by_symbol(orders_by_symbol),
        )
    except Exception as e:
        logger.error("Failed to get orders: {}", e)
        log_event("open_orders_snapshot_failed", error=str(e))
    return orders_by_symbol


def _normalize_order_side(side: object) -> str:
    return _normalize_enum_value(side)


def _order_age_hours(order: object, *, now: Optional[datetime] = None) -> Optional[float]:
    created_at = order_created_at(order)
    if created_at is None:
        return None
    reference = now or datetime.now(timezone.utc)
    return max(0.0, (reference - created_at).total_seconds() / 3600.0)


def _order_qty(order: object) -> float:
    try:
        return float(getattr(order, "qty", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _order_limit_price(order: object) -> Optional[float]:
    raw = getattr(order, "limit_price", None)
    if raw in (None, ""):
        return None
    try:
        price = float(raw)
    except (TypeError, ValueError):
        return None
    return price if price > 0.0 else None


def _positive_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0:
        return None
    return parsed


def _position_avg_entry_price(position: object) -> Optional[float]:
    if isinstance(position, dict):
        return _positive_float(position.get("avg_entry_price"))
    return _positive_float(getattr(position, "avg_entry_price", None))


def _min_exit_price_for_position(*, qty: float, entry_price: float, fee_rate: float) -> float:
    if qty < 0:
        return float(entry_price) * max(0.0, 1.0 - max(float(fee_rate), 0.0))
    return float(entry_price) * (1.0 + max(float(fee_rate), 0.0))


def resolve_live_entry_reference_price(
    symbol: str,
    *,
    default_price: float,
    is_short: bool,
    use_market_orders: bool,
) -> tuple[float, str]:
    reference_price = _positive_float(default_price)
    if reference_price is None:
        reference_price = 0.0
    if not use_market_orders:
        return float(reference_price), "signal_price"

    try:
        import alpaca_wrapper

        quote = alpaca_wrapper.latest_data(symbol)
        ask_price = _positive_float(getattr(quote, "ask_price", None))
        bid_price = _positive_float(getattr(quote, "bid_price", None))
        if is_short:
            if bid_price is not None:
                return float(bid_price), "quote_bid"
            if ask_price is not None:
                return float(ask_price), "quote_ask_fallback"
        else:
            if ask_price is not None:
                return float(ask_price), "quote_ask"
            if bid_price is not None:
                return float(bid_price), "quote_bid_fallback"
    except Exception as exc:
        log_event(
            "market_entry_reference_failed",
            symbol=symbol,
            is_short=bool(is_short),
            error=str(exc),
        )

    return float(reference_price), "signal_price"


def _cancel_specific_orders(
    api,
    *,
    symbol: str,
    orders_by_symbol: Dict[str, list],
    orders_with_reasons: List[tuple[object, str]],
) -> None:
    if not orders_with_reasons:
        return

    cancelled_ids: set[str] = set()
    for order, reason in orders_with_reasons:
        order_id = str(getattr(order, "id", ""))
        if order_id in cancelled_ids:
            continue
        try:
            log_event("order_cancel_requested", symbol=symbol, reason=reason, order=_serialize_order(order))
            api.cancel_order_by_id(order.id)
            logger.info("{}: cancelled order {} ({})", symbol, order.id, reason)
            log_event("order_cancel_succeeded", symbol=symbol, order_id=order_id, reason=reason)
            cancelled_ids.add(order_id)
        except Exception as e:
            logger.error("{}: cancel failed - {}", symbol, e)
            log_event("order_cancel_failed", symbol=symbol, reason=reason, order=_serialize_order(order), error=str(e))

    if not cancelled_ids:
        return

    remaining = [
        order
        for order in orders_by_symbol.get(symbol, [])
        if str(getattr(order, "id", "")) not in cancelled_ids
    ]
    if remaining:
        orders_by_symbol[symbol] = remaining
    else:
        orders_by_symbol.pop(symbol, None)


def _wait_for_entry_order_cancel_ack(
    api,
    *,
    symbol: str,
    wait_seconds: float = ENTRY_REPLACE_CANCEL_WAIT_SECONDS,
    poll_seconds: float = ENTRY_REPLACE_CANCEL_POLL_SECONDS,
) -> bool:
    wait_seconds = max(float(wait_seconds), 0.0)
    poll_seconds = max(float(poll_seconds), 0.05)
    if wait_seconds <= 0.0:
        return False

    deadline = time.monotonic() + wait_seconds
    while True:
        refreshed_orders = get_open_orders(api).get(symbol, [])
        if not refreshed_orders:
            log_event("entry_cancel_wait_cleared", symbol=symbol, wait_seconds=float(wait_seconds))
            return True

        if time.monotonic() >= deadline:
            log_event(
                "entry_cancel_wait_timeout",
                symbol=symbol,
                wait_seconds=float(wait_seconds),
                open_orders=[_serialize_order(order) for order in refreshed_orders],
            )
            return False

        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _reconcile_position_orders(
    api,
    *,
    symbol: str,
    orders_by_symbol: Dict[str, list],
    position_qty: float,
    exit_price: Optional[float],
) -> list:
    symbol_orders = list(orders_by_symbol.get(symbol, []))
    if not symbol_orders:
        return []

    intents: list[OrderIntent] = []
    if float(position_qty) > 0.0 and exit_price and float(exit_price) > 0.0:
        intents.append(OrderIntent(side="sell", qty=float(abs(position_qty)), limit_price=float(exit_price), kind="exit"))
    elif float(position_qty) < 0.0 and exit_price and float(exit_price) > 0.0:
        intents.append(OrderIntent(side="buy", qty=float(abs(position_qty)), limit_price=float(exit_price), kind="exit"))

    _cancel_specific_orders(
        api,
        symbol=symbol,
        orders_by_symbol=orders_by_symbol,
        orders_with_reasons=orders_to_cancel_for_live_symbol(
            symbol_orders,
            position_qty=float(position_qty),
            intents=intents,
        ),
    )
    return list(orders_by_symbol.get(symbol, []))


def _reconcile_entry_orders(
    api,
    *,
    symbol: str,
    orders_by_symbol: Dict[str, list],
    desired_side: str,
    desired_qty: float,
    desired_limit_price: Optional[float],
    entry_order_ttl_hours: float,
) -> EntryOrderReconcileResult:
    symbol_orders = list(orders_by_symbol.get(symbol, []))
    if not symbol_orders:
        return EntryOrderReconcileResult(matching_order=None, replacement_blocked=False)

    desired_side_norm = _normalize_order_side(desired_side)
    now = datetime.now(timezone.utc)
    kept_order = None
    orders_to_cancel: list[tuple[object, str]] = []

    def _sort_key(order: object) -> tuple[bool, float]:
        created_at = order_created_at(order)
        if created_at is None:
            return (False, 0.0)
        return (True, created_at.timestamp())

    for order in sorted(symbol_orders, key=_sort_key, reverse=True):
        order_side = _normalize_order_side(getattr(order, "side", None))
        if order_side != desired_side_norm:
            orders_to_cancel.append((order, "entry_side_mismatch"))
            continue

        age_hours = _order_age_hours(order, now=now)
        if (
            float(entry_order_ttl_hours) > 0.0
            and age_hours is not None
            and age_hours >= float(entry_order_ttl_hours)
        ):
            orders_to_cancel.append((order, "entry_order_ttl_expired"))
            continue

        qty_diff_pct = abs(_order_qty(order) - float(desired_qty)) / float(desired_qty) if float(desired_qty) > 0 else float("inf")
        qty_similar = qty_diff_pct < 0.05

        if desired_limit_price is None:
            price_similar = True
        else:
            existing_limit_price = _order_limit_price(order)
            if existing_limit_price is None:
                price_similar = False
            else:
                price_diff_pct = abs(existing_limit_price - float(desired_limit_price)) / float(desired_limit_price)
                price_similar = price_diff_pct < 0.0003

        if not (qty_similar and price_similar):
            orders_to_cancel.append((order, "entry_order_mismatch"))
            continue

        if kept_order is None:
            kept_order = order
            continue
        orders_to_cancel.append((order, "duplicate_matching_entry_order"))

    _cancel_specific_orders(
        api,
        symbol=symbol,
        orders_by_symbol=orders_by_symbol,
        orders_with_reasons=orders_to_cancel,
    )

    if kept_order is None:
        if orders_to_cancel and _wait_for_entry_order_cancel_ack(api, symbol=symbol):
            return EntryOrderReconcileResult(
                matching_order=None,
                replacement_blocked=False,
            )
        return EntryOrderReconcileResult(
            matching_order=None,
            replacement_blocked=bool(orders_to_cancel),
        )
    kept_id = str(getattr(kept_order, "id", ""))
    for order in orders_by_symbol.get(symbol, []):
        if str(getattr(order, "id", "")) == kept_id:
            return EntryOrderReconcileResult(
                matching_order=order,
                replacement_blocked=bool(orders_to_cancel),
            )
    return EntryOrderReconcileResult(
        matching_order=None,
        replacement_blocked=bool(orders_to_cancel),
    )


def poll_broker_events(
    api,
    state: dict[str, Any],
    *,
    reason: str,
    now: Optional[datetime] = None,
    lookback_hours: float = BROKER_EVENT_LOOKBACK_HOURS,
) -> None:
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    import alpaca_wrapper

    now = now or datetime.now(timezone.utc)
    cursor = state.setdefault("broker_event_cursor", {})
    order_after = _coerce_utc_datetime(cursor.get("closed_orders_after"))
    activity_after = _coerce_utc_datetime(cursor.get("fill_activities_after"))
    default_after = now - timedelta(hours=max(float(lookback_hours), 1.0))

    if order_after is None:
        order_after = default_after
    if activity_after is None:
        activity_after = default_after

    seen_order_keys = list(cursor.get("recent_order_event_keys", []))
    seen_activity_keys = list(cursor.get("recent_activity_event_keys", []))
    seen_order_set = set(seen_order_keys)
    seen_activity_set = set(seen_activity_keys)

    order_event_count = 0
    activity_event_count = 0
    latest_order_ts = order_after
    latest_activity_ts = activity_after

    log_event(
        "broker_event_poll_start",
        reason=reason,
        closed_orders_after=order_after.isoformat(),
        fill_activities_after=activity_after.isoformat(),
    )

    order_request_after = max(order_after - timedelta(seconds=1), default_after)
    try:
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=order_request_after,
            until=now,
            direction="asc",
            limit=500,
        )
        orders = list(api.get_orders(request) or [])
        orders.sort(key=lambda order: _order_event_timestamp(order) or datetime.min.replace(tzinfo=timezone.utc))
        for order in orders:
            event_key = _order_event_key(order)
            if event_key in seen_order_set:
                continue
            event_ts = _order_event_timestamp(order)
            if event_ts is not None and event_ts < order_request_after:
                continue
            status = _normalize_enum_value(_record_field(order, "status"))
            log_event(
                "broker_closed_order",
                reason=reason,
                event_ts=event_ts.isoformat() if event_ts is not None else None,
                order_status=status,
                order=_serialize_order(order),
            )
            seen_order_set.add(event_key)
            seen_order_keys.append(event_key)
            order_event_count += 1
            if event_ts is not None and event_ts > latest_order_ts:
                latest_order_ts = event_ts
    except Exception as exc:
        log_event(
            "broker_closed_orders_poll_failed",
            reason=reason,
            closed_orders_after=order_request_after.isoformat(),
            error=str(exc),
        )

    try:
        for activity_date in _iter_activity_dates(activity_after, now):
            activities = alpaca_wrapper.get_account_activities(
                api,
                activity_types=["FILL"],
                date=activity_date,
                direction="asc",
                page_size=100,
            ) or []
            for activity in activities:
                event_ts = _activity_timestamp(activity)
                if event_ts is not None and event_ts < activity_after:
                    continue
                event_key = _activity_event_key(activity)
                if event_key in seen_activity_set:
                    continue
                log_event(
                    "broker_fill_activity",
                    reason=reason,
                    event_ts=event_ts.isoformat() if event_ts is not None else None,
                    activity=_serialize_activity(activity),
                )
                seen_activity_set.add(event_key)
                seen_activity_keys.append(event_key)
                activity_event_count += 1
                if event_ts is not None and event_ts > latest_activity_ts:
                    latest_activity_ts = event_ts
    except Exception as exc:
        log_event(
            "broker_fill_activities_poll_failed",
            reason=reason,
            fill_activities_after=activity_after.isoformat(),
            error=str(exc),
        )

    cursor["closed_orders_after"] = latest_order_ts.isoformat()
    cursor["fill_activities_after"] = latest_activity_ts.isoformat()
    cursor["recent_order_event_keys"] = _trim_recent_keys(seen_order_keys)
    cursor["recent_activity_event_keys"] = _trim_recent_keys(seen_activity_keys)

    log_event(
        "broker_event_poll_complete",
        reason=reason,
        closed_order_events=int(order_event_count),
        fill_activity_events=int(activity_event_count),
        closed_orders_after=cursor["closed_orders_after"],
        fill_activities_after=cursor["fill_activities_after"],
    )


def _has_open_exit_order(symbol_orders: List[object], *, qty: float) -> bool:
    exit_side = "sell" if qty > 0 else "buy"
    return any(_normalize_order_side(getattr(order, "side", None)) == exit_side for order in symbol_orders)


def _is_fractional_quantity(qty: float) -> bool:
    try:
        abs_qty = abs(float(qty))
    except (TypeError, ValueError):
        return False
    if abs_qty <= 0:
        return False
    return not abs_qty.is_integer()


def _equity_notional(qty: float, price: float) -> float:
    try:
        return abs(float(qty)) * float(price)
    except (TypeError, ValueError):
        return 0.0


def _order_time_in_force_for_qty(symbol: str, qty: float, TimeInForce):
    if is_crypto_symbol(symbol):
        return TimeInForce.GTC
    return TimeInForce.DAY if _is_fractional_quantity(qty) else TimeInForce.GTC


def _is_substantial_live_position(symbol: str, qty: float, current_price: float) -> bool:
    abs_qty = abs(float(qty))
    if abs_qty <= 0:
        return False
    if is_crypto_symbol(symbol):
        return abs_qty >= 1e-8 and _equity_notional(abs_qty, float(current_price)) >= 1.0
    return abs_qty >= 1.0 or _equity_notional(abs_qty, float(current_price)) >= 1.0


def cancel_symbol_orders(api, symbol: str, orders_by_symbol: Dict):
    for order in orders_by_symbol.get(symbol, []):
        try:
            log_event("order_cancel_requested", symbol=symbol, order=_serialize_order(order))
            api.cancel_order_by_id(order.id)
            logger.info("{}: cancelled order {}", symbol, order.id)
            log_event("order_cancel_succeeded", symbol=symbol, order_id=str(order.id))
        except Exception as e:
            logger.error("{}: cancel failed - {}", symbol, e)
            log_event("order_cancel_failed", symbol=symbol, order=_serialize_order(order), error=str(e))


def place_exit_order(api, symbol: str, qty: float, sell_price: float, side: str = "sell"):
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    order_side = OrderSide.SELL if side == "sell" else OrderSide.BUY
    abs_qty = abs(qty)
    if abs_qty <= 0:
        log_event("exit_order_skipped", symbol=symbol, qty=abs_qty, side=side, reason="non_positive_qty")
        return None
    if not is_crypto_symbol(symbol) and _equity_notional(abs_qty, sell_price) < 1.0:
        log_event(
            "exit_order_skipped",
            symbol=symbol,
            qty=abs_qty,
            side=side,
            reason="notional_below_one",
            limit_price=round(sell_price, 2),
        )
        return None
    order_qty = float(abs_qty) if _is_fractional_quantity(abs_qty) else int(abs_qty)
    time_in_force = _order_time_in_force_for_qty(symbol, order_qty, TimeInForce)

    try:
        order = LimitOrderRequest(
            symbol=symbol,
            qty=order_qty,
            side=order_side,
            limit_price=round(sell_price, 2),
            time_in_force=time_in_force,
        )
        log_event(
            "exit_order_submit_requested",
            symbol=symbol,
            qty=order_qty,
            side=side,
            limit_price=round(sell_price, 2),
            time_in_force="day" if time_in_force == TimeInForce.DAY else "gtc",
        )
        result = api.submit_order(order)
        logger.info("{}: exit order placed - {} {} @ ${:.2f} (id={})",
                    symbol, side, order_qty, sell_price, result.id)
        log_event(
            "exit_order_submit_succeeded",
            symbol=symbol,
            qty=order_qty,
            side=side,
            limit_price=round(sell_price, 2),
            order_id=str(result.id),
        )
        return str(result.id)
    except Exception as e:
        logger.error("{}: exit order failed - {}", symbol, e)
        log_event(
            "exit_order_submit_failed",
            symbol=symbol,
            qty=order_qty,
            side=side,
            limit_price=round(sell_price, 2),
            error=str(e),
        )
        return None


def force_close_position(api, symbol: str, qty: float, current_price: float = 0):
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    abs_qty = abs(float(qty))
    if abs_qty <= 0:
        log_event("force_close_skipped", symbol=symbol, qty=float(qty), reason="non_positive_qty")
        return
    side = OrderSide.SELL if qty > 0 else OrderSide.BUY
    if current_price <= 0:
        logger.error("{}: no price for limit close, skipping", symbol)
        log_event("force_close_skipped", symbol=symbol, qty=float(qty), reason="missing_current_price")
        return
    if not is_crypto_symbol(symbol) and _equity_notional(abs_qty, current_price) < 1.0:
        log_event(
            "force_close_skipped",
            symbol=symbol,
            qty=float(qty),
            reason="notional_below_one",
            current_price=float(current_price),
        )
        return
    try:
        if side == OrderSide.SELL:
            price = round(current_price * 0.997, 2)
        else:
            price = round(current_price * 1.003, 2)
        order_qty = float(abs_qty) if _is_fractional_quantity(abs_qty) else int(abs_qty)
        time_in_force = _order_time_in_force_for_qty(symbol, order_qty, TimeInForce)
        order = LimitOrderRequest(
            symbol=symbol,
            qty=order_qty,
            side=side,
            limit_price=price,
            time_in_force=time_in_force,
        )
        log_event(
            "force_close_submit_requested",
            symbol=symbol,
            qty=order_qty,
            side=_normalize_order_side(side),
            limit_price=price,
            current_price=float(current_price),
            time_in_force="day" if time_in_force == TimeInForce.DAY else "gtc",
        )
        result = api.submit_order(order)
        logger.info("{}: force-close limit {} shares @ ${:.2f} (cur=${:.2f})", symbol, order_qty, price, current_price)
        log_trade({"event": "force_close", "symbol": symbol, "qty": order_qty,
                   "price": price, "cur_price": current_price, "order_id": str(result.id)})
        log_event(
            "force_close_submit_succeeded",
            symbol=symbol,
            qty=order_qty,
            side=_normalize_order_side(side),
            limit_price=price,
            current_price=float(current_price),
            order_id=str(result.id),
        )
    except Exception as e:
        logger.error("{}: force-close failed - {}", symbol, e)
        log_event(
            "force_close_submit_failed",
            symbol=symbol,
            qty=order_qty,
            side=_normalize_order_side(side),
            current_price=float(current_price),
            error=str(e),
        )


def manage_positions(api, state: dict, max_hold_hours: int = MAX_HOLD_HOURS,
                     active_symbols: set = None):
    """Check existing positions: enforce hold timeout, ensure exit orders exist.
    Force-close positions not in active_symbols or without exit prices."""
    now = datetime.now(timezone.utc)
    positions = get_current_positions(api)
    orders_by_symbol = get_open_orders(api)
    tracked = state.get("positions", {})
    removed = []
    log_event(
        "manage_positions_start",
        active_symbols=sorted(str(symbol) for symbol in (active_symbols or set())),
        max_hold_hours=float(max_hold_hours),
        tracked_positions=_json_safe(tracked),
        live_positions=[_serialize_position(symbol, payload) for symbol, payload in positions.items()],
        open_orders_by_symbol=_serialize_orders_by_symbol(orders_by_symbol),
    )

    for symbol, info in list(tracked.items()):
        pos_data = positions.get(symbol, {})
        qty = pos_data.get("qty", 0) if isinstance(pos_data, dict) else 0
        cur_price = pos_data.get("price", 0) if isinstance(pos_data, dict) else 0
        log_event(
            "manage_position_evaluated",
            symbol=symbol,
            tracked_info=info,
            live_position=_serialize_position(symbol, pos_data),
            open_orders=[_serialize_order(order) for order in orders_by_symbol.get(symbol, [])],
        )

        # For crypto, fractional qty < 1 can still be a real position (e.g. 0.5 ETH = $1000).
        # Use notional check for crypto the same way we do for stocks.
        abs_qty_check = abs(float(qty))
        if is_crypto_symbol(symbol):
            position_is_zero = abs_qty_check < 1e-8 or (float(cur_price) > 0 and _equity_notional(abs_qty_check, float(cur_price)) < 1.0)
        else:
            position_is_zero = abs_qty_check < 1
        if position_is_zero:
            pending_entry_qty = abs(float(info.get("qty", 0.0) or 0.0))
            pending_entry_price = info.get("entry_price")
            pending_entry_side = "sell" if str(info.get("side", "")).lower() == "short" else "buy"
            if (
                pending_entry_price
                and _is_substantial_live_position(symbol, pending_entry_qty, float(pending_entry_price))
                and orders_by_symbol.get(symbol)
            ):
                reconcile_result = _reconcile_entry_orders(
                    api,
                    symbol=symbol,
                    orders_by_symbol=orders_by_symbol,
                    desired_side=pending_entry_side,
                    desired_qty=float(pending_entry_qty),
                    desired_limit_price=round(float(pending_entry_price), 2),
                    entry_order_ttl_hours=0.0,
                )
                matching_entry_order = reconcile_result.matching_order
                if matching_entry_order is not None:
                    info["entry_order_id"] = str(getattr(matching_entry_order, "id", info.get("entry_order_id") or ""))
                    log_event(
                        "manage_position_action",
                        symbol=symbol,
                        action="pending_entry_present",
                        entry_order_id=info.get("entry_order_id"),
                    )
                    continue
            logger.info("{}: position closed, cleaning state", symbol)
            log_trade({"event": "exit_filled", "symbol": symbol,
                       "entry_price": info.get("entry_price"),
                       "exit_price": info.get("exit_price")})
            log_event("manage_position_action", symbol=symbol, action="cleanup_closed_position")
            cancel_symbol_orders(api, symbol, orders_by_symbol)
            removed.append(symbol)
            continue

        broker_avg_entry_price = _position_avg_entry_price(pos_data)
        if broker_avg_entry_price is not None:
            tracked_entry_price = _positive_float(info.get("entry_price"))
            if tracked_entry_price is None or abs(tracked_entry_price - broker_avg_entry_price) >= 0.01:
                info["entry_price"] = float(broker_avg_entry_price)
                log_event(
                    "manage_position_action",
                    symbol=symbol,
                    action="entry_price_reconciled",
                    old_entry_price=float(tracked_entry_price) if tracked_entry_price is not None else None,
                    broker_avg_entry_price=float(broker_avg_entry_price),
                )

            tracked_exit_price = _positive_float(info.get("exit_price"))
            fee_rate = float(info.get("fee_rate", 0.0) or 0.0)
            min_exit_price = _min_exit_price_for_position(
                qty=float(qty),
                entry_price=float(broker_avg_entry_price),
                fee_rate=fee_rate,
            )
            should_raise_exit = tracked_exit_price is None
            if tracked_exit_price is not None and qty > 0:
                should_raise_exit = tracked_exit_price < min_exit_price
            elif tracked_exit_price is not None and qty < 0:
                should_raise_exit = tracked_exit_price > min_exit_price
            if should_raise_exit:
                old_exit_price = float(tracked_exit_price) if tracked_exit_price is not None else None
                info["exit_price"] = float(min_exit_price)
                log_event(
                    "manage_position_action",
                    symbol=symbol,
                    action="exit_price_reconciled_to_entry_basis",
                    old_exit_price=old_exit_price,
                    new_exit_price=float(min_exit_price),
                    broker_avg_entry_price=float(broker_avg_entry_price),
                    fee_rate=float(fee_rate),
                )

        if is_short_only_symbol(symbol) and qty > 0:
            logger.info("{}: SHORT_ONLY stock held long, force closing {} shares", symbol, qty)
            log_event("manage_position_action", symbol=symbol, action="force_close", reason="short_only_long_position", qty=float(qty))
            cancel_symbol_orders(api, symbol, orders_by_symbol)
            force_close_position(api, symbol, qty, cur_price)
            removed.append(symbol)
            continue

        if active_symbols and symbol not in active_symbols:
            logger.info("{}: not in active symbol set, force closing {} shares", symbol, qty)
            log_event("manage_position_action", symbol=symbol, action="force_close", reason="not_in_active_symbols", qty=float(qty))
            cancel_symbol_orders(api, symbol, orders_by_symbol)
            force_close_position(api, symbol, qty, cur_price)
            removed.append(symbol)
            continue

        entry_time = datetime.fromisoformat(info["entry_time"])
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        use_calendar_hours = is_crypto_symbol(symbol)
        hours_held = (
            calendar_hours_between(entry_time, now)
            if use_calendar_hours
            else market_hours_between(entry_time, now)
        )
        pos_hold_limit = info.get("hold_hours", max_hold_hours)

        if hours_held >= pos_hold_limit:
            hours_label = "calendar" if use_calendar_hours else "market"
            logger.info(
                "{}: hold timeout ({:.1f} {} hrs >= {:.1f}h), force closing",
                symbol,
                hours_held,
                hours_label,
                pos_hold_limit,
            )
            log_event(
                "manage_position_action",
                symbol=symbol,
                action="force_close",
                reason="hold_timeout",
                hours_held=float(hours_held),
                hold_limit=float(pos_hold_limit),
                hold_clock=hours_label,
            )
            cancel_symbol_orders(api, symbol, orders_by_symbol)
            time.sleep(0.75)  # Wait for cancel to propagate before submitting new order
            force_close_position(api, symbol, qty, cur_price)
            removed.append(symbol)
            continue

        if not info.get("exit_price"):
            logger.info("{}: no exit price set, force closing {} shares", symbol, qty)
            log_event("manage_position_action", symbol=symbol, action="force_close", reason="missing_exit_price", qty=float(qty))
            cancel_symbol_orders(api, symbol, orders_by_symbol)
            time.sleep(0.75)  # Wait for cancel to propagate
            force_close_position(api, symbol, qty, cur_price)
            removed.append(symbol)
            continue

        symbol_orders = _reconcile_position_orders(
            api,
            symbol=symbol,
            orders_by_symbol=orders_by_symbol,
            position_qty=float(qty),
            exit_price=info.get("exit_price"),
        )
        has_exit = _has_open_exit_order(symbol_orders, qty=qty)
        if not has_exit:
            if symbol_orders:
                logger.info(
                    "{}: cancelling {} non-exit open orders before placing protective exit",
                    symbol,
                    len(symbol_orders),
                )
                log_event(
                    "manage_position_action",
                    symbol=symbol,
                    action="cancel_open_orders_before_exit",
                    reason="missing_protective_exit",
                    open_order_count=len(symbol_orders),
                )
                cancel_symbol_orders(api, symbol, orders_by_symbol)
                time.sleep(0.75)  # Wait for cancel to propagate before placing exit
            exit_side = "sell" if qty > 0 else "buy"
            oid = place_exit_order(api, symbol, qty, info["exit_price"], side=exit_side)
            if oid:
                info["exit_order_id"] = oid
                log_event(
                    "manage_position_action",
                    symbol=symbol,
                    action="protective_exit_submitted",
                    exit_order_id=str(oid),
                    exit_side=exit_side,
                    exit_price=float(info["exit_price"]),
                )
            else:
                logger.warning("{}: failed to place exit order, force closing", symbol)
                log_event("manage_position_action", symbol=symbol, action="force_close", reason="exit_submit_failed")
                force_close_position(api, symbol, qty, cur_price)
                removed.append(symbol)
                continue
        else:
            log_event("manage_position_action", symbol=symbol, action="protective_exit_present")

    for s in removed:
        tracked.pop(s, None)

    pending_close = set(state.get("pending_close", []))
    newly_pending_close: set[str] = set()
    for s in removed:
        pending_close.add(s)
        newly_pending_close.add(s)
    state["pending_close"] = list(pending_close)

    for symbol, pos_data in positions.items():
        qty = pos_data.get("qty", 0) if isinstance(pos_data, dict) else 0
        cur_price = pos_data.get("price", 0) if isinstance(pos_data, dict) else 0
        abs_qty = abs(float(qty))
        should_force_close_untracked = _is_substantial_live_position(symbol, abs_qty, cur_price)
        if symbol not in tracked and should_force_close_untracked and symbol not in pending_close:
            if active_symbols and symbol not in active_symbols:
                logger.info("{}: untracked position not in active set ({} shares), force closing", symbol, qty)
                log_event("manage_position_action", symbol=symbol, action="force_close_untracked", reason="untracked_not_active", qty=float(qty))
                force_close_position(api, symbol, qty, cur_price)
                pending_close.add(symbol)
                newly_pending_close.add(symbol)
                continue
            logger.info("{}: untracked position found ({} shares), force closing (no exit price)", symbol, qty)
            log_event("manage_position_action", symbol=symbol, action="force_close_untracked", reason="untracked_position", qty=float(qty))
            force_close_position(api, symbol, qty, cur_price)
            pending_close.add(symbol)
            newly_pending_close.add(symbol)

    still_pending = []
    for s in pending_close:
        if s in positions:
            pq = positions[s].get("qty", 0) if isinstance(positions[s], dict) else float(positions[s])
            pq_cur_price = positions[s].get("price", 0) if isinstance(positions[s], dict) else 0
            abs_pq = abs(float(pq))
            is_substantial = _is_substantial_live_position(s, abs_pq, float(pq_cur_price))
            if is_substantial:
                still_pending.append(s)
                # If untracked and no exit order exists, the previous close attempt expired or failed.
                # Retry force_close so the position doesn't sit abandoned indefinitely.
                if s not in tracked and s not in newly_pending_close:
                    existing_orders = orders_by_symbol.get(s, [])
                    exit_side_retry = "sell" if float(pq) > 0 else "buy"
                    has_any_close_order = any(
                        _normalize_order_side(getattr(o, "side", None)) == exit_side_retry
                        for o in existing_orders
                    )
                    if not has_any_close_order:
                        logger.warning(
                            "{}: pending_close has no exit order (qty={:.4f}, val=${:.2f}), retrying force close",
                            s, float(pq), _equity_notional(abs_pq, float(pq_cur_price)),
                        )
                        log_event(
                            "manage_position_action",
                            symbol=s,
                            action="force_close",
                            reason="pending_close_retry",
                            qty=float(pq),
                        )
                        force_close_position(api, s, float(pq), float(pq_cur_price))
    state["pending_close"] = still_pending

    state["positions"] = tracked
    log_event(
        "manage_positions_complete",
        tracked_positions=_json_safe(tracked),
        pending_close=sorted(str(symbol) for symbol in state.get("pending_close", [])),
        removed_symbols=sorted(str(symbol) for symbol in removed),
    )
    return len(tracked)


def generate_signal_for_symbol(
    symbol: str,
    model: torch.nn.Module,
    feature_columns: list,
    sequence_length: int,
    data_root: Path,
    cache_root: Path,
    device: torch.device,
    saved_normalizer: Optional[FeatureNormalizer] = None,
) -> Optional[Dict]:
    horizons = [1]
    if any("h24" in c for c in feature_columns):
        horizons = [1, 24]
    data_config = DatasetConfig(
        symbol=symbol,
        data_root=str(data_root),
        forecast_cache_root=str(cache_root),
        forecast_horizons=horizons,
        sequence_length=sequence_length,
        min_history_hours=100,
        validation_days=30,
        cache_only=True,
    )
    try:
        data_module = BinanceHourlyDataModule(data_config)
    except Exception as e:
        logger.warning("Failed to load data for {}: {}", symbol, e)
        log_event("signal_frame_load_failed", symbol=symbol, error=str(e))
        return None

    frame = data_module.frame.copy()
    frame["symbol"] = symbol

    try:
        norm = saved_normalizer if saved_normalizer is not None else data_module.normalizer
        action = generate_latest_action(
            model=model,
            frame=frame,
            feature_columns=feature_columns,
            normalizer=norm,
            sequence_length=sequence_length,
            horizon=1,
            device=device,
        )
        log_event(
            "signal_generated_raw",
            symbol=symbol,
            buy_price=action.get("buy_price"),
            sell_price=action.get("sell_price"),
            buy_amount=action.get("buy_amount"),
            sell_amount=action.get("sell_amount"),
            hold_hours=action.get("hold_hours"),
        )
        return action
    except Exception as e:
        logger.error("Failed to generate action for {}: {}", symbol, e)
        log_event("signal_generation_failed", symbol=symbol, error=str(e))
        return None


def execute_trades(
    api,
    signals: Dict,
    state: dict,
    max_positions: int = MAX_POSITIONS,
    *,
    market_order_entry: bool = False,
    entry_order_ttl_hours: float = 0.0,
    fee_rate: float = 0.0,
):
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    account = get_account_info(api)
    equity = account["equity"]
    buying_power = account["buying_power"]
    use_market_orders = bool(market_order_entry) and is_market_open_now()
    log_event(
        "execute_trades_start",
        signal_symbols=sorted(str(symbol) for symbol in signals.keys()),
        max_positions=int(max_positions),
        market_order_entry=bool(market_order_entry),
        effective_market_order_entry=bool(use_market_orders),
        tracked_positions=_json_safe(state.get("positions", {})),
        account=account,
    )

    if equity <= 0:
        logger.error("No equity available")
        log_event("execute_trades_aborted", reason="non_positive_equity", account=account)
        return

    tracked = state.get("positions", {})
    positions = get_current_positions(api)
    open_orders_by_symbol = get_open_orders(api)
    current_count = len(tracked)
    initial_slots_available = max_positions - current_count
    slots_available = max(initial_slots_available, 0)
    planning_buying_power = float(buying_power)
    allocator_mode = normalize_entry_allocator_mode(ENTRY_ALLOCATOR_MODE)
    allocator_reserve_fraction = min(max(float(ENTRY_ALLOCATOR_RESERVE_FRACTION), 0.0), 1.0)

    stock_leverage = 2.0
    per_position_stock = (equity * stock_leverage) / max_positions
    sorted_signals = sorted(signals.items(), key=lambda x: x[1].get("edge", 0), reverse=True)
    entry_candidates: list[dict[str, Any]] = []

    for symbol, action in sorted_signals:
        if symbol in tracked:
            log_event("entry_skipped", symbol=symbol, reason="already_tracked")
            continue

        live_pos = positions.get(symbol, {})
        live_qty = float(live_pos.get("qty", 0.0) or 0.0)
        live_price = float(live_pos.get("price", 0.0) or 0.0)
        if _is_substantial_live_position(symbol, abs(live_qty), live_price):
            log_event(
                "entry_skipped",
                symbol=symbol,
                reason="live_position_already_open",
                live_qty=float(live_qty),
                live_price=float(live_price),
            )
            continue

        buy_price = action.get("buy_price", 0)
        sell_price = action.get("sell_price", 0)
        if buy_price <= 0:
            log_event("entry_skipped", symbol=symbol, reason="invalid_buy_price", buy_price=buy_price)
            continue

        hold_hours = action.get("hold_hours", MAX_HOLD_HOURS)
        crypto = is_crypto_symbol(symbol)
        if crypto:
            is_short = False
        else:
            directions = resolve_trade_directions(symbol, allow_short=True)
            is_short = directions.can_short and not directions.can_long

        signal_amount, intensity_frac = entry_intensity_fraction(
            action,
            is_short=is_short,
            trade_amount_scale=TRADE_AMOUNT_SCALE,
            intensity_power=ENTRY_INTENSITY_POWER,
            min_intensity_fraction=ENTRY_MIN_INTENSITY_FRACTION,
            side_multiplier=SHORT_INTENSITY_MULTIPLIER if is_short else LONG_INTENSITY_MULTIPLIER,
        )
        log_event(
            "entry_candidate",
            symbol=symbol,
            side=("short" if is_short else "long"),
            edge=action.get("edge"),
            buy_price=buy_price,
            sell_price=sell_price,
            hold_hours=hold_hours,
            signal_amount=signal_amount,
            intensity_fraction=float(intensity_frac),
        )
        if intensity_frac <= 0:
            log_event("entry_skipped", symbol=symbol, reason="non_positive_intensity", signal_amount=signal_amount)
            continue
        if MIN_BUY_AMOUNT > 0 and signal_amount < MIN_BUY_AMOUNT:
            log_event("entry_skipped", symbol=symbol, reason="below_min_buy_amount", signal_amount=signal_amount, min_buy_amount=MIN_BUY_AMOUNT)
            continue

        if is_short:
            entry_price = sell_price
            exit_price = buy_price
            entry_side = OrderSide.SELL
        else:
            entry_price = buy_price
            exit_price = sell_price
            entry_side = OrderSide.BUY

        market_entry_reference_price = _positive_float(action.get("market_entry_reference_price"))
        if use_market_orders and market_entry_reference_price is not None:
            entry_price = float(market_entry_reference_price)
        entry_reference_source = str(action.get("market_entry_reference_source", "signal_price"))

        if entry_price <= 0:
            log_event("entry_skipped", symbol=symbol, reason="invalid_entry_price", entry_price=entry_price)
            continue
        if len(entry_candidates) >= slots_available:
            log_event("entry_skipped", symbol=symbol, reason="no_slots_remaining")
            continue

        per_position = per_position_stock if not crypto else (equity / max_positions)
        candidate = {
            "symbol": symbol,
            "action": action,
            "entry_side": entry_side,
            "entry_side_norm": "sell" if is_short else "buy",
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "hold_hours": float(hold_hours),
            "signal_amount": float(signal_amount),
            "intensity_fraction": float(intensity_frac),
            "side_str": "short" if is_short else "long",
            "order_type": "market" if use_market_orders else "limit",
            "entry_reference_source": entry_reference_source,
            "fee_rate": float(fee_rate),
            "slot_budget": float(per_position),
        }

        if allocator_mode == "legacy":
            position_alloc = per_position * intensity_frac
            target_value = min(position_alloc, planning_buying_power * 0.9)
            target_qty = int(target_value / entry_price)
            if target_qty <= 0:
                log_event(
                    "entry_skipped",
                    symbol=symbol,
                    reason="target_qty_below_one",
                    target_value=float(target_value),
                    entry_price=float(entry_price),
                )
                continue
            if target_value > planning_buying_power:
                logger.warning(
                    "{}: insufficient buying power (${:.0f} needed, ${:.0f} avail)",
                    symbol,
                    target_value,
                    planning_buying_power,
                )
                log_event(
                    "entry_skipped",
                    symbol=symbol,
                    reason="insufficient_buying_power",
                    target_value=float(target_value),
                    buying_power=float(planning_buying_power),
                )
                continue
            candidate["position_alloc"] = float(position_alloc)
            candidate["target_value"] = float(target_value)
            candidate["target_qty"] = int(target_qty)
            planning_buying_power = max(0.0, planning_buying_power - float(target_value))
        entry_candidates.append(candidate)

    if allocator_mode == "concentrated" and entry_candidates:
        deployable_budget = max(0.0, planning_buying_power) * (1.0 - allocator_reserve_fraction)
        target_values = allocate_concentrated_entry_budget(
            [
                EntryAllocationCandidate(
                    symbol=str(candidate["symbol"]),
                    edge=float(candidate["action"].get("edge", 0.0) or 0.0),
                    intensity_fraction=float(candidate["intensity_fraction"]),
                    slot_budget=float(candidate["slot_budget"]),
                )
                for candidate in entry_candidates
            ],
            max_positions=max_positions,
            deployable_budget=deployable_budget,
            edge_power=float(ENTRY_ALLOCATOR_EDGE_POWER),
            max_single_position_fraction=float(ENTRY_ALLOCATOR_MAX_SINGLE_POSITION_FRACTION),
        )
        concentrated_candidates: list[dict[str, Any]] = []
        for candidate, target_value in zip(entry_candidates, target_values, strict=False):
            candidate["position_alloc"] = float(target_value)
            candidate["target_value"] = float(target_value)
            target_qty = int(float(target_value) / float(candidate["entry_price"]))
            if target_qty <= 0:
                log_event(
                    "entry_skipped",
                    symbol=candidate["symbol"],
                    reason="target_qty_below_one",
                    target_value=float(target_value),
                    entry_price=float(candidate["entry_price"]),
                )
                continue
            candidate["target_qty"] = int(target_qty)
            concentrated_candidates.append(candidate)
        entry_candidates = concentrated_candidates

    if initial_slots_available <= 0:
        logger.info("Max positions reached ({}/{})", current_count, max_positions)
        log_event("execute_trades_aborted", reason="no_slots_available", current_count=int(current_count), max_positions=int(max_positions))

    selected_by_symbol = {candidate["symbol"]: candidate for candidate in entry_candidates}
    reconcile_symbols = set(open_orders_by_symbol) | set(selected_by_symbol) | set(tracked)
    for symbol in sorted(reconcile_symbols):
        live_pos = positions.get(symbol, {})
        live_qty = float(live_pos.get("qty", 0.0) or 0.0)
        live_price = float(live_pos.get("price", 0.0) or 0.0)
        if _is_substantial_live_position(symbol, abs(live_qty), live_price):
            if symbol in tracked:
                _reconcile_position_orders(
                    api,
                    symbol=symbol,
                    orders_by_symbol=open_orders_by_symbol,
                    position_qty=float(live_qty),
                    exit_price=tracked.get(symbol, {}).get("exit_price"),
                )
            continue

        candidate = selected_by_symbol.get(symbol)
        if candidate is None:
            _reconcile_position_orders(
                api,
                symbol=symbol,
                orders_by_symbol=open_orders_by_symbol,
                position_qty=0.0,
                exit_price=None,
            )
            continue

        reconcile_result = _reconcile_entry_orders(
            api,
            symbol=symbol,
            orders_by_symbol=open_orders_by_symbol,
            desired_side=candidate["entry_side_norm"],
            desired_qty=float(candidate["target_qty"]),
            desired_limit_price=None if candidate["order_type"] == "market" else round(float(candidate["entry_price"]), 2),
            entry_order_ttl_hours=float(entry_order_ttl_hours),
        )
        candidate["matching_order"] = reconcile_result.matching_order
        candidate["entry_replacement_blocked"] = bool(reconcile_result.replacement_blocked)

    for candidate in entry_candidates:
        symbol = candidate["symbol"]
        matching_order = candidate.get("matching_order")
        if matching_order is not None:
            tracked_at = order_created_at(matching_order) or datetime.now(timezone.utc)
            tracked[symbol] = {
                "qty": candidate["target_qty"] if candidate["side_str"] == "long" else -candidate["target_qty"],
                "side": candidate["side_str"],
                "entry_price": candidate["entry_price"],
                "entry_time": tracked_at.isoformat(),
                "entry_order_id": str(getattr(matching_order, "id", "")),
                "exit_price": candidate["exit_price"],
                "exit_order_id": None,
                "hold_hours": candidate["hold_hours"],
                "max_hold_until": (tracked_at + timedelta(hours=float(candidate["hold_hours"]))).isoformat(),
                "fee_rate": float(candidate.get("fee_rate", 0.0)),
            }
            log_event("position_tracking_created", symbol=symbol, tracked_position=tracked[symbol])
            log_event("entry_skipped", symbol=symbol, reason="existing_open_entry_order", existing_order=_serialize_order(matching_order))
            continue
        if bool(candidate.get("entry_replacement_blocked")):
            log_event(
                "entry_skipped",
                symbol=symbol,
                reason="waiting_for_entry_order_cancel",
            )
            continue

        try:
            if candidate["order_type"] == "market":
                from alpaca.trading.requests import MarketOrderRequest

                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=int(candidate["target_qty"]),
                    side=candidate["entry_side"],
                    time_in_force=TimeInForce.DAY,
                )
            else:
                order = LimitOrderRequest(
                    symbol=symbol,
                    qty=int(candidate["target_qty"]),
                    side=candidate["entry_side"],
                    limit_price=round(float(candidate["entry_price"]), 2),
                    time_in_force=TimeInForce.DAY,
                )
            log_event(
                "entry_order_submit_requested",
                symbol=symbol,
                side=candidate["side_str"],
                qty=int(candidate["target_qty"]),
                limit_price=round(float(candidate["entry_price"]), 2),
                exit_price=float(candidate["exit_price"]),
                edge=candidate["action"].get("edge"),
                hold_hours=float(candidate["hold_hours"]),
                signal_amount=float(candidate["signal_amount"]),
                intensity_fraction=float(candidate["intensity_fraction"]),
                position_alloc=float(candidate["position_alloc"]),
                target_value=float(candidate["target_value"]),
                order_type=candidate["order_type"],
                entry_reference_source=candidate.get("entry_reference_source"),
            )
            result = api.submit_order(order)
            logger.info(
                "{}: {} {} entry {} shares @ ${:.2f} (${:.0f}, hold={:.1f}h)",
                symbol,
                candidate["order_type"],
                candidate["side_str"],
                candidate["target_qty"],
                candidate["entry_price"],
                candidate["position_alloc"],
                candidate["hold_hours"],
            )
            log_trade({"event": "entry", "symbol": symbol, "side": candidate["side_str"],
                       "qty": candidate["target_qty"], "price": candidate["entry_price"], "exit_price": candidate["exit_price"],
                       "edge": candidate["action"].get("edge"), "intensity": candidate["intensity_fraction"],
                       "signal_amount": candidate["signal_amount"],
                       "alloc": candidate["position_alloc"], "order_id": str(result.id), "order_type": candidate["order_type"]})
            log_event(
                "entry_order_submit_succeeded",
                symbol=symbol,
                side=candidate["side_str"],
                qty=int(candidate["target_qty"]),
                limit_price=round(float(candidate["entry_price"]), 2),
                exit_price=float(candidate["exit_price"]),
                order_id=str(result.id),
                order_type=candidate["order_type"],
            )

            now = datetime.now(timezone.utc)
            tracked[symbol] = {
                "qty": candidate["target_qty"] if candidate["side_str"] == "long" else -candidate["target_qty"],
                "side": candidate["side_str"],
                "entry_price": candidate["entry_price"],
                "entry_time": now.isoformat(),
                "entry_order_id": str(result.id),
                "exit_price": candidate["exit_price"],
                "exit_order_id": None,
                "hold_hours": candidate["hold_hours"],
                "max_hold_until": (now + timedelta(hours=float(candidate["hold_hours"]))).isoformat(),
                "fee_rate": float(candidate.get("fee_rate", 0.0)),
            }
            log_event("position_tracking_created", symbol=symbol, tracked_position=tracked[symbol])

            buying_power -= float(candidate["target_value"])
            slots_available -= 1
        except Exception as e:
            logger.error("{}: order failed - {}", symbol, e)
            log_event(
                "entry_order_submit_failed",
                symbol=symbol,
                side=candidate["side_str"],
                qty=int(candidate["target_qty"]),
                limit_price=round(float(candidate["entry_price"]), 2),
                error=str(e),
            )

    state["positions"] = tracked
    log_event(
        "execute_trades_complete",
        tracked_positions=_json_safe(tracked),
        remaining_buying_power=float(buying_power),
        slots_available=int(slots_available),
    )


def run_cycle(
    model, feature_columns, sequence_length, device,
    stocks, args, api, state, max_positions=MAX_POSITIONS, max_hold_hours=MAX_HOLD_HOURS,
    normalizer=None,
):
    active_set = set(stocks)
    market_open = is_market_open_now() or args.ignore_market_hours
    log_event(
        "cycle_start",
        stocks=sorted(str(symbol) for symbol in stocks),
        market_open=bool(market_open),
        dry_run=bool(args.dry_run),
        ignore_market_hours=bool(args.ignore_market_hours),
        max_positions=int(max_positions),
        max_hold_hours=float(max_hold_hours),
        tracked_positions=_json_safe(state.get("positions", {})),
    )
    if api is not None:
        poll_broker_events(api, state, reason="pre_cycle")

    num_pos = manage_positions(api, state, max_hold_hours=max_hold_hours, active_symbols=active_set)
    logger.info("Active positions: {} | Market: {}", num_pos, "OPEN" if market_open else "CLOSED")

    signals = {}
    for symbol in stocks:
        action = generate_signal_for_symbol(
            symbol, model, feature_columns, sequence_length,
            args.stock_data_root, args.stock_cache_root, device,
            saved_normalizer=normalizer,
        )
        if action:
            pred_high = action.get("predicted_high", 0)
            pred_low = action.get("predicted_low", 0)
            buy_price = action.get("buy_price", 0)
            sell_price = action.get("sell_price", 0)
            use_market_orders = bool(getattr(args, "market_order_entry", False)) and bool(market_open)
            is_short_symbol = is_short_only_symbol(symbol)
            default_entry_price = sell_price if is_short_symbol else buy_price
            entry_reference_price, entry_reference_source = resolve_live_entry_reference_price(
                symbol,
                default_price=float(default_entry_price or 0.0),
                is_short=is_short_symbol,
                use_market_orders=use_market_orders,
            )
            action["market_entry_reference_price"] = float(entry_reference_price)
            action["market_entry_reference_source"] = str(entry_reference_source)

            if is_short_symbol:
                edge = (
                    (entry_reference_price - pred_low) / entry_reference_price - args.fee_rate
                    if entry_reference_price > 0
                    else 0
                )
            else:
                edge = (
                    (pred_high - entry_reference_price) / entry_reference_price - args.fee_rate
                    if entry_reference_price > 0
                    else 0
                )

            if edge >= args.min_edge:
                action["edge"] = edge
                signals[symbol] = action
                side = "short" if is_short_symbol else "long"
                signal_amount, intensity = entry_intensity_fraction(
                    action,
                    is_short=is_short_symbol,
                    trade_amount_scale=TRADE_AMOUNT_SCALE,
                    intensity_power=ENTRY_INTENSITY_POWER,
                    min_intensity_fraction=ENTRY_MIN_INTENSITY_FRACTION,
                    side_multiplier=(
                        SHORT_INTENSITY_MULTIPLIER if is_short_symbol else LONG_INTENSITY_MULTIPLIER
                    ),
                )
                logger.info("{}: {} buy={:.2f} sell={:.2f} ref={:.2f} edge={:.4f} hold={:.1f}h amt={:.3f} int={:.3f}",
                           symbol, side, buy_price, sell_price, entry_reference_price, edge,
                           action.get("hold_hours", 0), signal_amount, intensity)
                log_event(
                    "signal_accepted",
                    symbol=symbol,
                    side=side,
                    buy_price=float(buy_price),
                    sell_price=float(sell_price),
                    market_entry_reference_price=float(entry_reference_price),
                    market_entry_reference_source=str(entry_reference_source),
                    edge=float(edge),
                    hold_hours=float(action.get("hold_hours", 0)),
                    signal_amount=float(signal_amount),
                    intensity=float(intensity),
                )
            else:
                logger.debug("{}: edge={:.4f} below {:.4f}", symbol, edge, args.min_edge)
                log_event(
                    "signal_rejected",
                    symbol=symbol,
                    reason="edge_below_threshold",
                    edge=float(edge),
                    min_edge=float(args.min_edge),
                    buy_price=float(buy_price),
                    sell_price=float(sell_price),
                    market_entry_reference_price=float(entry_reference_price),
                    market_entry_reference_source=str(entry_reference_source),
                )
        else:
            log_event("signal_rejected", symbol=symbol, reason="no_action")

    execution_mode = "none"
    if market_open and not args.dry_run and signals:
        execute_trades(
            api,
            signals,
            state,
            max_positions=max_positions,
            market_order_entry=bool(getattr(args, "market_order_entry", False)),
            entry_order_ttl_hours=float(getattr(args, "entry_order_ttl_hours", 0.0) or 0.0),
            fee_rate=float(getattr(args, "fee_rate", 0.0) or 0.0),
        )
        execution_mode = "live_execute"
    elif not market_open and signals:
        logger.info("Market closed - {} signals ready, will trade when open", len(signals))
        execution_mode = "market_closed"
    elif not signals:
        logger.info("No signals above threshold")
        execution_mode = "no_signals"
    else:
        execution_mode = "dry_run"

    if api is not None:
        poll_broker_events(api, state, reason="post_cycle")
    save_state(state)
    log_event(
        "cycle_complete",
        execution_mode=execution_mode,
        signal_symbols=sorted(str(symbol) for symbol in signals.keys()),
        tracked_positions=_json_safe(state.get("positions", {})),
    )


def main():
    global TRADE_AMOUNT_SCALE, MIN_BUY_AMOUNT
    global ENTRY_INTENSITY_POWER, ENTRY_MIN_INTENSITY_FRACTION
    global LONG_INTENSITY_MULTIPLIER, SHORT_INTENSITY_MULTIPLIER
    global ENTRY_ALLOCATOR_MODE, ENTRY_ALLOCATOR_EDGE_POWER
    global ENTRY_ALLOCATOR_MAX_SINGLE_POSITION_FRACTION, ENTRY_ALLOCATOR_RESERVE_FRACTION
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--stock-symbols", default=",".join(DEFAULT_ALPACA_LIVE8_STOCKS))
    parser.add_argument("--crypto-symbols", default="")
    parser.add_argument("--stock-data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--crypto-data-root", type=Path, default=Path("trainingdatahourly/crypto"))
    parser.add_argument("--stock-cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--crypto-cache-root", type=Path, default=Path("binanceneural/forecast_cache"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--min-edge", type=float, default=0.012)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--allocation-per-symbol", type=float, default=1000.0)
    parser.add_argument("--ignore-market-hours", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument(
        "--market-order-entry",
        action="store_true",
        help="Use market orders for live entries during regular market hours.",
    )
    parser.add_argument(
        "--entry-order-ttl-hours",
        type=float,
        default=6.0,
        help="Cancel and refresh open entry orders older than this many hours (0 disables).",
    )
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS)
    parser.add_argument("--max-hold-hours", type=int, default=MAX_HOLD_HOURS)
    parser.add_argument("--trade-amount-scale", type=float, default=TRADE_AMOUNT_SCALE)
    parser.add_argument("--min-buy-amount", type=float, default=MIN_BUY_AMOUNT)
    parser.add_argument("--entry-intensity-power", type=float, default=ENTRY_INTENSITY_POWER)
    parser.add_argument(
        "--entry-min-intensity-fraction",
        type=float,
        default=ENTRY_MIN_INTENSITY_FRACTION,
        help="Floor intensity fraction applied to non-zero signals (0-1).",
    )
    parser.add_argument(
        "--long-intensity-multiplier",
        type=float,
        default=LONG_INTENSITY_MULTIPLIER,
        help="Multiply long-side signal intensity (post power transform).",
    )
    parser.add_argument(
        "--short-intensity-multiplier",
        type=float,
        default=SHORT_INTENSITY_MULTIPLIER,
        help="Multiply short-side signal intensity (post power transform).",
    )
    parser.add_argument(
        "--entry-allocator-mode",
        type=str,
        default=ENTRY_ALLOCATOR_MODE,
        choices=["legacy", "concentrated"],
        help="How to allocate budget across selected entries.",
    )
    parser.add_argument(
        "--entry-allocator-edge-power",
        type=float,
        default=ENTRY_ALLOCATOR_EDGE_POWER,
        help="Exponent applied to edge when concentrating spare entry budget.",
    )
    parser.add_argument(
        "--entry-allocator-max-single-position-fraction",
        type=float,
        default=ENTRY_ALLOCATOR_MAX_SINGLE_POSITION_FRACTION,
        help="Cap concentrated sizing as a fraction of deployable budget per symbol.",
    )
    parser.add_argument(
        "--entry-allocator-reserve-fraction",
        type=float,
        default=ENTRY_ALLOCATOR_RESERVE_FRACTION,
        help="Fraction of buying power held back from new entries.",
    )
    args = parser.parse_args()
    if args.trade_amount_scale <= 0:
        raise ValueError("--trade-amount-scale must be > 0")
    if args.min_buy_amount < 0:
        raise ValueError("--min-buy-amount must be >= 0")
    if args.entry_intensity_power < 0:
        raise ValueError("--entry-intensity-power must be >= 0")
    if args.entry_min_intensity_fraction < 0:
        raise ValueError("--entry-min-intensity-fraction must be >= 0")
    if args.long_intensity_multiplier < 0:
        raise ValueError("--long-intensity-multiplier must be >= 0")
    if args.short_intensity_multiplier < 0:
        raise ValueError("--short-intensity-multiplier must be >= 0")
    if args.entry_allocator_edge_power < 0:
        raise ValueError("--entry-allocator-edge-power must be >= 0")
    if not 0 <= float(args.entry_allocator_max_single_position_fraction) <= 1:
        raise ValueError("--entry-allocator-max-single-position-fraction must be in [0, 1]")
    if not 0 <= float(args.entry_allocator_reserve_fraction) <= 1:
        raise ValueError("--entry-allocator-reserve-fraction must be in [0, 1]")
    TRADE_AMOUNT_SCALE = float(args.trade_amount_scale)
    MIN_BUY_AMOUNT = float(args.min_buy_amount)
    ENTRY_INTENSITY_POWER = float(args.entry_intensity_power)
    ENTRY_MIN_INTENSITY_FRACTION = float(args.entry_min_intensity_fraction)
    LONG_INTENSITY_MULTIPLIER = float(args.long_intensity_multiplier)
    SHORT_INTENSITY_MULTIPLIER = float(args.short_intensity_multiplier)
    ENTRY_ALLOCATOR_MODE = normalize_entry_allocator_mode(args.entry_allocator_mode)
    ENTRY_ALLOCATOR_EDGE_POWER = float(args.entry_allocator_edge_power)
    ENTRY_ALLOCATOR_MAX_SINGLE_POSITION_FRACTION = float(args.entry_allocator_max_single_position_fraction)
    ENTRY_ALLOCATOR_RESERVE_FRACTION = float(args.entry_allocator_reserve_fraction)

    max_pos = args.max_positions
    max_hold = args.max_hold_hours

    paper = not args.live

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feature_columns, sequence_length, normalizer = load_model(args.checkpoint_dir, epoch=args.epoch)
    model = model.to(device)

    stocks = [s.strip().upper() for s in args.stock_symbols.split(",") if s.strip()]

    logger.info("=" * 60)
    logger.info("Portfolio Stock Trading Bot")
    logger.info("=" * 60)
    logger.info("Stocks: {} (LONG_ONLY: {}, SHORT_ONLY: {})",
               len(stocks), len([s for s in stocks if is_long_only_symbol(s)]),
               len([s for s in stocks if is_short_only_symbol(s)]))
    logger.info("Max positions: {}, Hold limit: {}h", max_pos, max_hold)
    logger.info("Live entry order type: {}", "MARKET" if args.market_order_entry else "LIMIT")
    logger.info("Live entry order TTL: {}h", float(args.entry_order_ttl_hours))
    logger.info(
        "Entry allocator: mode={} edge_power={} max_single_frac={} reserve_frac={}",
        ENTRY_ALLOCATOR_MODE,
        float(ENTRY_ALLOCATOR_EDGE_POWER),
        float(ENTRY_ALLOCATOR_MAX_SINGLE_POSITION_FRACTION),
        float(ENTRY_ALLOCATOR_RESERVE_FRACTION),
    )
    logger.info("Mode: {}", "LIVE" if not paper else "PAPER")
    log_event(
        "trader_started",
        checkpoint_dir=str(args.checkpoint_dir),
        epoch=args.epoch,
        stocks=sorted(stocks),
        paper=bool(paper),
        loop=bool(args.loop),
        max_positions=int(max_pos),
        max_hold_hours=float(max_hold),
    )

    api = None if args.dry_run else get_alpaca_client(paper=paper)
    state = load_state()

    if api:
        account = get_account_info(api)
        logger.info("Equity: ${:.2f}, Buying power: ${:.2f}", account["equity"], account["buying_power"])

    run_cycle(model, feature_columns, sequence_length, device, stocks, args, api, state, max_pos, max_hold, normalizer=normalizer)

    if not args.loop:
        return

    while True:
        now = datetime.now(timezone.utc)
        next_hour = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
        wait_secs = (next_hour - now).total_seconds()
        logger.info("Sleeping {:.0f}s until next hour", wait_secs)
        time.sleep(wait_secs)

        run_cycle(model, feature_columns, sequence_length, device, stocks, args, api, state, max_pos, max_hold, normalizer=normalizer)


if __name__ == "__main__":
    main()
