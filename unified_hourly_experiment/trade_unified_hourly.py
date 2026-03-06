#!/usr/bin/env python3
"""Unified hourly trading bot for stocks (Alpaca) with exit orders, directional constraints, multi-position."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from src.torch_load_utils import torch_load_compat
from src.symbol_utils import is_crypto_symbol
from src.hourly_trader_utils import entry_intensity_fraction

LONG_ONLY = {"NVDA", "MSFT", "META", "GOOG", "NET", "PLTR", "DBX", "TSLA", "AAPL"}
SHORT_ONLY = {"YELP", "EBAY", "TRIP", "MTCH", "ANGI", "Z", "EXPE", "BKNG", "NWSA", "NYT"}

STATE_FILE = Path("strategy_state/stock_portfolio_state.json")
TRADE_LOG = Path("strategy_state/stock_trade_log.jsonl")
EVENT_LOG = Path("strategy_state/stock_event_log.jsonl")
MAX_HOLD_HOURS = 6
MAX_POSITIONS = 10
TRADE_AMOUNT_SCALE = 100.0
MIN_BUY_AMOUNT = 0.0
ENTRY_INTENSITY_POWER = 1.0
ENTRY_MIN_INTENSITY_FRACTION = 0.0
LONG_INTENSITY_MULTIPLIER = 1.0
SHORT_INTENSITY_MULTIPLIER = 1.0


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


def _serialize_orders_by_symbol(orders_by_symbol: dict[str, list]) -> dict[str, list[dict[str, Any]]]:
    return {
        str(symbol): [_serialize_order(order) for order in orders]
        for symbol, orders in orders_by_symbol.items()
    }


def _tracked_state_summary(state: dict[str, Any]) -> dict[str, Any]:
    tracked = state.get("positions", {})
    return {
        "tracked_symbols": sorted(str(symbol) for symbol in tracked.keys()),
        "tracked_count": int(len(tracked)),
        "pending_close": sorted(str(symbol) for symbol in state.get("pending_close", [])),
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


def get_alpaca_client(paper: bool = True):
    from alpaca.trading.client import TradingClient
    key_id = ALP_KEY_ID if paper else ALP_KEY_ID_PROD
    secret = ALP_SECRET_KEY if paper else ALP_SECRET_KEY_PROD
    return TradingClient(key_id, secret, paper=paper)


def get_current_positions(api) -> Dict[str, dict]:
    positions = {}
    try:
        for pos in api.get_all_positions():
            positions[pos.symbol] = {
                "qty": float(pos.qty),
                "price": float(pos.current_price),
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
            sym = o.symbol
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
    if side is None:
        return ""
    value = getattr(side, "value", side)
    return str(value).strip().lower()


def _has_open_exit_order(symbol_orders: List[object], *, qty: float) -> bool:
    exit_side = "sell" if qty > 0 else "buy"
    return any(_normalize_order_side(getattr(order, "side", None)) == exit_side for order in symbol_orders)


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
    if abs_qty < 1:
        log_event("exit_order_skipped", symbol=symbol, qty=abs_qty, side=side, reason="qty_below_one")
        return None

    try:
        order = LimitOrderRequest(
            symbol=symbol,
            qty=int(abs_qty),
            side=order_side,
            limit_price=round(sell_price, 2),
            time_in_force=TimeInForce.GTC,
        )
        log_event(
            "exit_order_submit_requested",
            symbol=symbol,
            qty=int(abs_qty),
            side=side,
            limit_price=round(sell_price, 2),
            time_in_force="gtc",
        )
        result = api.submit_order(order)
        logger.info("{}: exit order placed - {} {} @ ${:.2f} (id={})",
                    symbol, side, int(abs_qty), sell_price, result.id)
        log_event(
            "exit_order_submit_succeeded",
            symbol=symbol,
            qty=int(abs_qty),
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
            qty=int(abs_qty),
            side=side,
            limit_price=round(sell_price, 2),
            error=str(e),
        )
        return None


def force_close_position(api, symbol: str, qty: float, current_price: float = 0):
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    if abs(qty) < 1:
        log_event("force_close_skipped", symbol=symbol, qty=float(qty), reason="qty_below_one")
        return
    side = OrderSide.SELL if qty > 0 else OrderSide.BUY
    if current_price <= 0:
        logger.error("{}: no price for limit close, skipping", symbol)
        log_event("force_close_skipped", symbol=symbol, qty=float(qty), reason="missing_current_price")
        return
    try:
        if side == OrderSide.SELL:
            price = round(current_price * 0.997, 2)
        else:
            price = round(current_price * 1.003, 2)
        order = LimitOrderRequest(
            symbol=symbol,
            qty=int(abs(qty)),
            side=side,
            limit_price=price,
            time_in_force=TimeInForce.DAY,
        )
        log_event(
            "force_close_submit_requested",
            symbol=symbol,
            qty=int(abs(qty)),
            side=_normalize_order_side(side),
            limit_price=price,
            current_price=float(current_price),
        )
        result = api.submit_order(order)
        logger.info("{}: force-close limit {} shares @ ${:.2f} (cur=${:.2f})", symbol, int(abs(qty)), price, current_price)
        log_trade({"event": "force_close", "symbol": symbol, "qty": int(abs(qty)),
                   "price": price, "cur_price": current_price, "order_id": str(result.id)})
        log_event(
            "force_close_submit_succeeded",
            symbol=symbol,
            qty=int(abs(qty)),
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
            qty=int(abs(qty)),
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

        if abs(qty) < 1:
            logger.info("{}: position closed, cleaning state", symbol)
            log_trade({"event": "exit_filled", "symbol": symbol,
                       "entry_price": info.get("entry_price"),
                       "exit_price": info.get("exit_price")})
            log_event("manage_position_action", symbol=symbol, action="cleanup_closed_position")
            cancel_symbol_orders(api, symbol, orders_by_symbol)
            removed.append(symbol)
            continue

        if symbol in SHORT_ONLY and qty > 0:
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
        hours_held = market_hours_between(entry_time, now)
        pos_hold_limit = info.get("hold_hours", max_hold_hours)

        if hours_held >= pos_hold_limit:
            logger.info("{}: hold timeout ({:.1f} market hrs >= {:.1f}h), force closing", symbol, hours_held, pos_hold_limit)
            log_event(
                "manage_position_action",
                symbol=symbol,
                action="force_close",
                reason="hold_timeout",
                hours_held=float(hours_held),
                hold_limit=float(pos_hold_limit),
            )
            cancel_symbol_orders(api, symbol, orders_by_symbol)
            force_close_position(api, symbol, qty, cur_price)
            removed.append(symbol)
            continue

        if not info.get("exit_price"):
            logger.info("{}: no exit price set, force closing {} shares", symbol, qty)
            log_event("manage_position_action", symbol=symbol, action="force_close", reason="missing_exit_price", qty=float(qty))
            cancel_symbol_orders(api, symbol, orders_by_symbol)
            force_close_position(api, symbol, qty, cur_price)
            removed.append(symbol)
            continue

        symbol_orders = orders_by_symbol.get(symbol, [])
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
    for s in removed:
        pending_close.add(s)
    state["pending_close"] = list(pending_close)

    for symbol, pos_data in positions.items():
        qty = pos_data.get("qty", 0) if isinstance(pos_data, dict) else 0
        cur_price = pos_data.get("price", 0) if isinstance(pos_data, dict) else 0
        if symbol not in tracked and abs(qty) >= 1 and symbol not in pending_close:
            if active_symbols and symbol not in active_symbols:
                logger.info("{}: untracked position not in active set ({} shares), force closing", symbol, qty)
                log_event("manage_position_action", symbol=symbol, action="force_close_untracked", reason="untracked_not_active", qty=float(qty))
                force_close_position(api, symbol, qty, cur_price)
                pending_close.add(symbol)
                continue
            logger.info("{}: untracked position found ({} shares), force closing (no exit price)", symbol, qty)
            log_event("manage_position_action", symbol=symbol, action="force_close_untracked", reason="untracked_position", qty=float(qty))
            force_close_position(api, symbol, qty, cur_price)
            pending_close.add(symbol)

    still_pending = []
    for s in pending_close:
        if s in positions:
            pq = positions[s].get("qty", 0) if isinstance(positions[s], dict) else float(positions[s])
            if abs(pq) >= 1:
                still_pending.append(s)
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


def execute_trades(api, signals: Dict, state: dict, max_positions: int = MAX_POSITIONS):
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    account = get_account_info(api)
    equity = account["equity"]
    buying_power = account["buying_power"]
    log_event(
        "execute_trades_start",
        signal_symbols=sorted(str(symbol) for symbol in signals.keys()),
        max_positions=int(max_positions),
        tracked_positions=_json_safe(state.get("positions", {})),
        account=account,
    )

    if equity <= 0:
        logger.error("No equity available")
        log_event("execute_trades_aborted", reason="non_positive_equity", account=account)
        return

    tracked = state.get("positions", {})
    current_count = len(tracked)
    slots_available = max_positions - current_count

    if slots_available <= 0:
        logger.info("Max positions reached ({}/{})", current_count, max_positions)
        log_event("execute_trades_aborted", reason="no_slots_available", current_count=int(current_count), max_positions=int(max_positions))
        return

    stock_leverage = 2.0
    per_position_stock = (equity * stock_leverage) / max_positions

    sorted_signals = sorted(signals.items(), key=lambda x: x[1].get("edge", 0), reverse=True)

    for symbol, action in sorted_signals:
        if slots_available <= 0:
            log_event("entry_skipped", symbol=symbol, reason="no_slots_remaining")
            break

        if symbol in tracked:
            log_event("entry_skipped", symbol=symbol, reason="already_tracked")
            continue

        buy_price = action.get("buy_price", 0)
        sell_price = action.get("sell_price", 0)
        if buy_price <= 0:
            log_event("entry_skipped", symbol=symbol, reason="invalid_buy_price", buy_price=buy_price)
            continue

        hold_hours = action.get("hold_hours", MAX_HOLD_HOURS)

        crypto = is_crypto_symbol(symbol)
        if crypto:
            is_long = True
            is_short = False
        else:
            is_long = symbol in LONG_ONLY or symbol not in SHORT_ONLY
            is_short = symbol in SHORT_ONLY

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
            exit_side = "buy"
        else:
            entry_price = buy_price
            exit_price = sell_price
            entry_side = OrderSide.BUY
            exit_side = "sell"

        if entry_price <= 0:
            log_event("entry_skipped", symbol=symbol, reason="invalid_entry_price", entry_price=entry_price)
            continue

        per_position = per_position_stock if not crypto else (equity / max_positions)
        position_alloc = per_position * intensity_frac
        target_value = min(position_alloc, buying_power * 0.9)
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

        if target_value > buying_power:
            logger.warning("{}: insufficient buying power (${:.0f} needed, ${:.0f} avail)",
                          symbol, target_value, buying_power)
            log_event(
                "entry_skipped",
                symbol=symbol,
                reason="insufficient_buying_power",
                target_value=float(target_value),
                buying_power=float(buying_power),
            )
            continue

        try:
            order = LimitOrderRequest(
                symbol=symbol,
                qty=target_qty,
                side=entry_side,
                limit_price=round(entry_price, 2),
                time_in_force=TimeInForce.DAY,
            )
            side_str = "short" if is_short else "long"
            log_event(
                "entry_order_submit_requested",
                symbol=symbol,
                side=side_str,
                qty=int(target_qty),
                limit_price=round(entry_price, 2),
                exit_price=float(exit_price),
                edge=action.get("edge"),
                hold_hours=float(hold_hours),
                signal_amount=float(signal_amount),
                intensity_fraction=float(intensity_frac),
                position_alloc=float(position_alloc),
                target_value=float(target_value),
            )
            result = api.submit_order(order)
            logger.info("{}: {} entry {} shares @ ${:.2f} (${:.0f}, hold={:.1f}h)",
                       symbol, side_str, target_qty, entry_price, position_alloc, hold_hours)
            log_trade({"event": "entry", "symbol": symbol, "side": side_str,
                       "qty": target_qty, "price": entry_price, "exit_price": exit_price,
                       "edge": action.get("edge"), "intensity": intensity_frac,
                       "signal_amount": signal_amount,
                       "alloc": position_alloc, "order_id": str(result.id)})
            log_event(
                "entry_order_submit_succeeded",
                symbol=symbol,
                side=side_str,
                qty=int(target_qty),
                limit_price=round(entry_price, 2),
                exit_price=float(exit_price),
                order_id=str(result.id),
            )

            now = datetime.now(timezone.utc)
            tracked[symbol] = {
                "qty": target_qty if not is_short else -target_qty,
                "side": side_str,
                "entry_price": entry_price,
                "entry_time": now.isoformat(),
                "entry_order_id": str(result.id),
                "exit_price": exit_price,
                "exit_order_id": None,
                "hold_hours": hold_hours,
                "max_hold_until": (now + timedelta(hours=hold_hours)).isoformat(),
            }
            log_event("position_tracking_created", symbol=symbol, tracked_position=tracked[symbol])

            buying_power -= target_value
            slots_available -= 1

        except Exception as e:
            logger.error("{}: order failed - {}", symbol, e)
            log_event(
                "entry_order_submit_failed",
                symbol=symbol,
                side=("short" if is_short else "long"),
                qty=int(target_qty),
                limit_price=round(entry_price, 2),
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

            if symbol in SHORT_ONLY:
                edge = (sell_price - pred_low) / sell_price - args.fee_rate if sell_price > 0 else 0
            else:
                edge = (pred_high - buy_price) / buy_price - args.fee_rate if buy_price > 0 else 0

            if edge >= args.min_edge:
                action["edge"] = edge
                signals[symbol] = action
                side = "short" if symbol in SHORT_ONLY else "long"
                signal_amount, intensity = entry_intensity_fraction(
                    action,
                    is_short=(symbol in SHORT_ONLY),
                    trade_amount_scale=TRADE_AMOUNT_SCALE,
                    intensity_power=ENTRY_INTENSITY_POWER,
                    min_intensity_fraction=ENTRY_MIN_INTENSITY_FRACTION,
                    side_multiplier=(
                        SHORT_INTENSITY_MULTIPLIER if (symbol in SHORT_ONLY) else LONG_INTENSITY_MULTIPLIER
                    ),
                )
                logger.info("{}: {} buy={:.2f} sell={:.2f} edge={:.4f} hold={:.1f}h amt={:.3f} int={:.3f}",
                           symbol, side, buy_price, sell_price, edge,
                           action.get("hold_hours", 0), signal_amount, intensity)
                log_event(
                    "signal_accepted",
                    symbol=symbol,
                    side=side,
                    buy_price=float(buy_price),
                    sell_price=float(sell_price),
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
                )
        else:
            log_event("signal_rejected", symbol=symbol, reason="no_action")

    execution_mode = "none"
    if market_open and not args.dry_run and signals:
        execute_trades(api, signals, state, max_positions=max_positions)
        execution_mode = "live_execute"
    elif not market_open and signals:
        logger.info("Market closed - {} signals ready, will trade when open", len(signals))
        execution_mode = "market_closed"
    elif not signals:
        logger.info("No signals above threshold")
        execution_mode = "no_signals"
    else:
        execution_mode = "dry_run"

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--stock-symbols", default="NVDA,MSFT,META,GOOG")
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
    args = parser.parse_args()
    TRADE_AMOUNT_SCALE = float(args.trade_amount_scale)
    MIN_BUY_AMOUNT = float(args.min_buy_amount)
    ENTRY_INTENSITY_POWER = float(args.entry_intensity_power)
    ENTRY_MIN_INTENSITY_FRACTION = float(args.entry_min_intensity_fraction)
    LONG_INTENSITY_MULTIPLIER = float(args.long_intensity_multiplier)
    SHORT_INTENSITY_MULTIPLIER = float(args.short_intensity_multiplier)

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
               len(stocks), len([s for s in stocks if s in LONG_ONLY]),
               len([s for s in stocks if s in SHORT_ONLY]))
    logger.info("Max positions: {}, Hold limit: {}h", max_pos, max_hold)
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
