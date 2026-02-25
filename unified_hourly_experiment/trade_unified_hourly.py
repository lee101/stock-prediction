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
from typing import Dict, List, Optional

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

LONG_ONLY = {"NVDA", "MSFT", "META", "GOOG", "NET", "PLTR", "DBX", "TSLA", "AAPL"}
SHORT_ONLY = {"YELP", "EBAY", "TRIP", "MTCH", "ANGI", "Z", "EXPE", "BKNG", "NWSA", "NYT"}

STATE_FILE = Path("strategy_state/stock_portfolio_state.json")
TRADE_LOG = Path("strategy_state/stock_trade_log.jsonl")
MAX_HOLD_HOURS = 6
MAX_POSITIONS = 10
TRADE_AMOUNT_SCALE = 100.0
MIN_BUY_AMOUNT = 0.0


def log_trade(event: dict):
    event["logged_at"] = datetime.now(timezone.utc).isoformat()
    try:
        TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(TRADE_LOG, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        pass


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"positions": {}}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


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
    except Exception as e:
        logger.error("Failed to get positions: {}", e)
    return positions


def get_account_info(api) -> dict:
    try:
        account = api.get_account()
        return {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
        }
    except Exception as e:
        logger.error("Failed to get account: {}", e)
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
    except Exception as e:
        logger.error("Failed to get orders: {}", e)
    return orders_by_symbol


def cancel_symbol_orders(api, symbol: str, orders_by_symbol: Dict):
    for order in orders_by_symbol.get(symbol, []):
        try:
            api.cancel_order_by_id(order.id)
            logger.info("{}: cancelled order {}", symbol, order.id)
        except Exception as e:
            logger.error("{}: cancel failed - {}", symbol, e)


def place_exit_order(api, symbol: str, qty: float, sell_price: float, side: str = "sell"):
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    order_side = OrderSide.SELL if side == "sell" else OrderSide.BUY
    abs_qty = abs(qty)
    if abs_qty < 1:
        return None

    try:
        order = LimitOrderRequest(
            symbol=symbol,
            qty=int(abs_qty),
            side=order_side,
            limit_price=round(sell_price, 2),
            time_in_force=TimeInForce.GTC,
        )
        result = api.submit_order(order)
        logger.info("{}: exit order placed - {} {} @ ${:.2f} (id={})",
                    symbol, side, int(abs_qty), sell_price, result.id)
        return str(result.id)
    except Exception as e:
        logger.error("{}: exit order failed - {}", symbol, e)
        return None


def force_close_position(api, symbol: str, qty: float, current_price: float = 0):
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    if abs(qty) < 1:
        return
    side = OrderSide.SELL if qty > 0 else OrderSide.BUY
    if current_price <= 0:
        logger.error("{}: no price for limit close, skipping", symbol)
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
        result = api.submit_order(order)
        logger.info("{}: force-close limit {} shares @ ${:.2f} (cur=${:.2f})", symbol, int(abs(qty)), price, current_price)
        log_trade({"event": "force_close", "symbol": symbol, "qty": int(abs(qty)),
                   "price": price, "cur_price": current_price, "order_id": str(result.id)})
    except Exception as e:
        logger.error("{}: force-close failed - {}", symbol, e)


def manage_positions(api, state: dict, max_hold_hours: int = MAX_HOLD_HOURS,
                     active_symbols: set = None):
    """Check existing positions: enforce hold timeout, ensure exit orders exist.
    Force-close positions not in active_symbols or without exit prices."""
    now = datetime.now(timezone.utc)
    positions = get_current_positions(api)
    orders_by_symbol = get_open_orders(api)
    tracked = state.get("positions", {})
    removed = []

    for symbol, info in list(tracked.items()):
        pos_data = positions.get(symbol, {})
        qty = pos_data.get("qty", 0) if isinstance(pos_data, dict) else 0
        cur_price = pos_data.get("price", 0) if isinstance(pos_data, dict) else 0

        if abs(qty) < 1:
            logger.info("{}: position closed, cleaning state", symbol)
            log_trade({"event": "exit_filled", "symbol": symbol,
                       "entry_price": info.get("entry_price"),
                       "exit_price": info.get("exit_price")})
            cancel_symbol_orders(api, symbol, orders_by_symbol)
            removed.append(symbol)
            continue

        if symbol in SHORT_ONLY and qty > 0:
            logger.info("{}: SHORT_ONLY stock held long, force closing {} shares", symbol, qty)
            cancel_symbol_orders(api, symbol, orders_by_symbol)
            force_close_position(api, symbol, qty, cur_price)
            removed.append(symbol)
            continue

        if active_symbols and symbol not in active_symbols:
            logger.info("{}: not in active symbol set, force closing {} shares", symbol, qty)
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
            cancel_symbol_orders(api, symbol, orders_by_symbol)
            force_close_position(api, symbol, qty, cur_price)
            removed.append(symbol)
            continue

        if not info.get("exit_price"):
            logger.info("{}: no exit price set, force closing {} shares", symbol, qty)
            cancel_symbol_orders(api, symbol, orders_by_symbol)
            force_close_position(api, symbol, qty, cur_price)
            removed.append(symbol)
            continue

        has_exit = len(orders_by_symbol.get(symbol, [])) > 0
        if not has_exit:
            exit_side = "sell" if qty > 0 else "buy"
            oid = place_exit_order(api, symbol, qty, info["exit_price"], side=exit_side)
            if oid:
                info["exit_order_id"] = oid
            else:
                logger.warning("{}: failed to place exit order, force closing", symbol)
                force_close_position(api, symbol, qty, cur_price)
                removed.append(symbol)
                continue

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
                force_close_position(api, symbol, qty, cur_price)
                pending_close.add(symbol)
                continue
            logger.info("{}: untracked position found ({} shares), force closing (no exit price)", symbol, qty)
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
        return action
    except Exception as e:
        logger.error("Failed to generate action for {}: {}", symbol, e)
        return None


def execute_trades(api, signals: Dict, state: dict, max_positions: int = MAX_POSITIONS):
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    account = get_account_info(api)
    equity = account["equity"]
    buying_power = account["buying_power"]

    if equity <= 0:
        logger.error("No equity available")
        return

    tracked = state.get("positions", {})
    current_count = len(tracked)
    slots_available = max_positions - current_count

    if slots_available <= 0:
        logger.info("Max positions reached ({}/{})", current_count, max_positions)
        return

    leverage = 2.0
    per_position = (equity * leverage) / max_positions

    sorted_signals = sorted(signals.items(), key=lambda x: x[1].get("edge", 0), reverse=True)

    for symbol, action in sorted_signals:
        if slots_available <= 0:
            break

        if symbol in tracked:
            continue

        buy_price = action.get("buy_price", 0)
        sell_price = action.get("sell_price", 0)
        buy_amount = action.get("buy_amount", 0)
        intensity_frac = min(buy_amount / TRADE_AMOUNT_SCALE, 1.0)

        if buy_price <= 0 or intensity_frac <= 0:
            continue
        if MIN_BUY_AMOUNT > 0 and buy_amount < MIN_BUY_AMOUNT:
            continue

        hold_hours = action.get("hold_hours", MAX_HOLD_HOURS)

        is_long = symbol in LONG_ONLY or symbol not in SHORT_ONLY
        is_short = symbol in SHORT_ONLY

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
            continue

        position_alloc = per_position * intensity_frac
        target_value = min(position_alloc, buying_power * 0.9)
        target_qty = int(target_value / entry_price)

        if target_qty <= 0:
            continue

        if target_value > buying_power:
            logger.warning("{}: insufficient buying power (${:.0f} needed, ${:.0f} avail)",
                          symbol, target_value, buying_power)
            continue

        try:
            order = LimitOrderRequest(
                symbol=symbol,
                qty=target_qty,
                side=entry_side,
                limit_price=round(entry_price, 2),
                time_in_force=TimeInForce.DAY,
            )
            result = api.submit_order(order)
            side_str = "short" if is_short else "long"
            logger.info("{}: {} entry {} shares @ ${:.2f} (${:.0f}, hold={:.1f}h)",
                       symbol, side_str, target_qty, entry_price, position_alloc, hold_hours)
            log_trade({"event": "entry", "symbol": symbol, "side": side_str,
                       "qty": target_qty, "price": entry_price, "exit_price": exit_price,
                       "edge": action.get("edge"), "intensity": intensity_frac,
                       "alloc": position_alloc, "order_id": str(result.id)})

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

            buying_power -= target_value
            slots_available -= 1

        except Exception as e:
            logger.error("{}: order failed - {}", symbol, e)

    state["positions"] = tracked


def run_cycle(
    model, feature_columns, sequence_length, device,
    stocks, args, api, state, max_positions=MAX_POSITIONS, max_hold_hours=MAX_HOLD_HOURS,
    normalizer=None,
):
    active_set = set(stocks)
    market_open = is_market_open_now() or args.ignore_market_hours

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
                buy_amt = action.get("buy_amount", 0)
                intensity = min(buy_amt / TRADE_AMOUNT_SCALE, 1.0)
                logger.info("{}: {} buy={:.2f} sell={:.2f} edge={:.4f} hold={:.1f}h int={:.3f}",
                           symbol, side, buy_price, sell_price, edge,
                           action.get("hold_hours", 0), intensity)
            else:
                logger.debug("{}: edge={:.4f} below {:.4f}", symbol, edge, args.min_edge)

    if market_open and not args.dry_run and signals:
        execute_trades(api, signals, state, max_positions=max_positions)
    elif not market_open and signals:
        logger.info("Market closed - {} signals ready, will trade when open", len(signals))
    elif not signals:
        logger.info("No signals above threshold")

    save_state(state)


def main():
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
    args = parser.parse_args()

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
