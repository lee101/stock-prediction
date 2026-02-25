#!/usr/bin/env python3
"""Crypto hourly trading bot -- trades BTC/ETH on Alpaca ONLY when stock market is CLOSED."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, Optional
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

import torch
from loguru import logger

from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer
from binanceneural.config import DatasetConfig
from binanceneural.model import build_policy, policy_config_from_payload
from binanceneural.inference import generate_latest_action
from src.torch_load_utils import torch_load_compat
from src.date_utils import is_nyse_open_on_date

NEW_YORK = ZoneInfo("America/New_York")
BUFFER_OPEN = dt_time(8, 30)
BUFFER_CLOSE = dt_time(17, 0)
BACKOUT_START = dt_time(8, 0)

STATE_FILE = Path("strategy_state/crypto_state.json")
TRADE_LOG = Path("strategy_state/crypto_trade_log.jsonl")
MAX_HOLD_HOURS = 6
TRADE_AMOUNT_SCALE = 100.0


def is_crypto_allowed() -> bool:
    ny = datetime.now(NEW_YORK)
    if not is_nyse_open_on_date(ny):
        return True
    t = ny.time()
    return not (BUFFER_OPEN <= t < BUFFER_CLOSE)


def is_backout_window() -> bool:
    ny = datetime.now(NEW_YORK)
    if not is_nyse_open_on_date(ny):
        return False
    t = ny.time()
    return BACKOUT_START <= t < BUFFER_OPEN


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


def alpaca_symbol(symbol: str) -> str:
    """BTCUSD -> BTC/USD for Alpaca crypto."""
    if "/" in symbol:
        return symbol
    if symbol.endswith("USD"):
        return symbol[:-3] + "/USD"
    return symbol


def load_model(checkpoint_dir: Path, epoch: int = None):
    if epoch is not None:
        best_ckpt = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    else:
        checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if not checkpoints:
            raise ValueError(f"No checkpoints in {checkpoint_dir}")
        best_ckpt = checkpoints[-1]
    logger.info("Loading: {}", best_ckpt.name)

    ckpt = torch_load_compat(best_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

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
    sequence_length = config.get("sequence_length", 48)

    if not feature_columns:
        from binanceneural.data import build_default_feature_columns
        embed_w = state_dict.get("embed.weight")
        if embed_w is not None and embed_w.ndim == 2:
            input_dim = int(embed_w.shape[1])
            for h_try in [[1, 6], [1]]:
                fc = build_default_feature_columns(h_try)
                if len(fc) == input_dim:
                    feature_columns = fc
                    break
        if not feature_columns:
            feature_columns = build_default_feature_columns([1, 6])

    policy_cfg = policy_config_from_payload(config, input_dim=len(feature_columns), state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    normalizer = None
    if config_path.exists():
        with open(config_path) as f:
            cfg2 = json.load(f)
        if "normalizer" in cfg2:
            normalizer = FeatureNormalizer.from_dict(cfg2["normalizer"])

    return model, feature_columns, sequence_length, normalizer


def get_alpaca_client(paper: bool = True):
    from alpaca.trading.client import TradingClient
    key_id = ALP_KEY_ID if paper else ALP_KEY_ID_PROD
    secret = ALP_SECRET_KEY if paper else ALP_SECRET_KEY_PROD
    return TradingClient(key_id, secret, paper=paper)


def get_crypto_position(api, alp_sym: str) -> Optional[dict]:
    try:
        pos = api.get_open_position(alp_sym.replace("/", ""))
        return {"qty": float(pos.qty), "price": float(pos.current_price)}
    except Exception:
        return None


def get_account_equity(api) -> float:
    try:
        return float(api.get_account().equity)
    except Exception:
        return 0


def get_open_orders_for(api, alp_sym: str) -> list:
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    try:
        orders = api.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        return [o for o in orders if o.symbol == alp_sym.replace("/", "")]
    except Exception:
        return []


def cancel_orders(api, orders):
    for o in orders:
        try:
            api.cancel_order_by_id(o.id)
        except Exception:
            pass


def place_crypto_order(api, alp_sym: str, qty: float, price: float, side: str = "buy"):
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
    try:
        order = LimitOrderRequest(
            symbol=alp_sym,
            qty=round(qty, 6),
            side=order_side,
            limit_price=round(price, 2),
            time_in_force=TimeInForce.GTC,
        )
        result = api.submit_order(order)
        logger.info("{}: {} {:.6f} @ ${:.2f} (id={})", alp_sym, side, qty, price, result.id)
        return str(result.id)
    except Exception as e:
        logger.error("{}: order failed - {}", alp_sym, e)
        return None


def force_close_crypto(api, alp_sym: str, qty: float, cur_price: float):
    if abs(qty) < 1e-8:
        return
    side = "sell" if qty > 0 else "buy"
    slip = 0.003
    price = cur_price * (1 - slip) if side == "sell" else cur_price * (1 + slip)
    oid = place_crypto_order(api, alp_sym, abs(qty), price, side)
    if oid:
        log_trade({"event": "force_close", "symbol": alp_sym, "qty": abs(qty),
                   "price": price, "side": side})


def generate_signal(
    symbol: str, model, feature_columns, sequence_length, data_root, cache_root,
    device, normalizer=None,
) -> Optional[Dict]:
    horizons = [1, 6]
    if not any("h6" in c for c in feature_columns):
        horizons = [1]
    data_config = DatasetConfig(
        symbol=symbol, data_root=str(data_root),
        forecast_cache_root=str(cache_root),
        forecast_horizons=horizons, sequence_length=sequence_length,
        min_history_hours=100, validation_days=30, cache_only=True,
    )
    try:
        dm = BinanceHourlyDataModule(data_config)
    except Exception as e:
        logger.warning("{}: data load failed: {}", symbol, e)
        return None
    frame = dm.frame.copy()
    frame["symbol"] = symbol
    try:
        norm = normalizer if normalizer is not None else dm.normalizer
        return generate_latest_action(
            model=model, frame=frame, feature_columns=feature_columns,
            normalizer=norm, sequence_length=sequence_length, horizon=1, device=device,
        )
    except Exception as e:
        logger.error("{}: signal gen failed: {}", symbol, e)
        return None


def manage_position(api, symbol: str, state: dict, max_hold_hours: int):
    alp_sym = alpaca_symbol(symbol)
    tracked = state.get("positions", {})
    pos = get_crypto_position(api, alp_sym)

    if symbol in tracked and (pos is None or abs(pos["qty"]) < 1e-8):
        logger.info("{}: position closed", symbol)
        log_trade({"event": "exit_filled", "symbol": symbol,
                   "entry_price": tracked[symbol].get("entry_price")})
        cancel_orders(api, get_open_orders_for(api, alp_sym))
        tracked.pop(symbol, None)
        state["positions"] = tracked
        return False

    if symbol not in tracked:
        return False

    info = tracked[symbol]
    now = datetime.now(timezone.utc)
    entry_time = datetime.fromisoformat(info["entry_time"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)
    hours_held = (now - entry_time).total_seconds() / 3600

    if hours_held >= max_hold_hours:
        logger.info("{}: hold timeout ({:.1f}h >= {}h)", symbol, hours_held, max_hold_hours)
        cancel_orders(api, get_open_orders_for(api, alp_sym))
        force_close_crypto(api, alp_sym, pos["qty"], pos["price"])
        tracked.pop(symbol, None)
        state["positions"] = tracked
        return False

    orders = get_open_orders_for(api, alp_sym)
    if not orders and info.get("exit_price"):
        exit_side = "sell" if pos["qty"] > 0 else "buy"
        place_crypto_order(api, alp_sym, abs(pos["qty"]), info["exit_price"], exit_side)

    state["positions"] = tracked
    return True


def backout_all(api, state: dict, symbols: list):
    tracked = state.get("positions", {})
    for symbol in list(tracked.keys()):
        alp_sym = alpaca_symbol(symbol)
        pos = get_crypto_position(api, alp_sym)
        if pos and abs(pos["qty"]) > 1e-8:
            logger.info("{}: market backout, closing {:.6f} @ ${:.2f}", symbol, pos["qty"], pos["price"])
            cancel_orders(api, get_open_orders_for(api, alp_sym))
            force_close_crypto(api, alp_sym, pos["qty"], pos["price"])
            log_trade({"event": "market_backout", "symbol": symbol, "qty": pos["qty"],
                       "price": pos["price"]})
        tracked.pop(symbol, None)
    state["positions"] = tracked


def run_cycle(model, feature_columns, sequence_length, device, symbol, args, api, state):
    alp_sym = alpaca_symbol(symbol)
    tracked = state.get("positions", {})

    if is_backout_window():
        logger.info("Backout window -- force-closing before market")
        backout_all(api, state, [symbol])
        save_state(state)
        return

    allowed = is_crypto_allowed()

    has_pos = manage_position(api, symbol, state, args.max_hold_hours)

    if not allowed:
        logger.info("Market hours -- crypto trading paused (pos={})", has_pos)
        save_state(state)
        return

    if has_pos:
        logger.info("{}: holding position", symbol)
        save_state(state)
        return

    action = generate_signal(
        symbol, model, feature_columns, sequence_length,
        args.data_root, args.cache_root, device, normalizer=args._normalizer,
    )
    if not action:
        save_state(state)
        return

    buy_price = action.get("buy_price", 0)
    sell_price = action.get("sell_price", 0)
    buy_amount = action.get("buy_amount", 0)
    intensity = min(buy_amount / TRADE_AMOUNT_SCALE, 1.0)

    if buy_price <= 0 or intensity <= 0:
        logger.debug("{}: no signal (buy={:.2f} int={:.3f})", symbol, buy_price, intensity)
        save_state(state)
        return

    pred_high = action.get("predicted_high", 0)
    edge = (pred_high - buy_price) / buy_price - args.fee if buy_price > 0 else 0

    logger.info("{}: buy={:.2f} sell={:.2f} edge={:.4f} int={:.3f}",
               symbol, buy_price, sell_price, edge, intensity)

    if edge < args.min_edge:
        logger.info("{}: edge {:.4f} below threshold {:.4f}", symbol, edge, args.min_edge)
        save_state(state)
        return

    if args.dry_run:
        logger.info("{}: DRY RUN -- would buy", symbol)
        save_state(state)
        return

    equity = get_account_equity(api)
    if equity <= 0:
        save_state(state)
        return

    alloc = equity * intensity * args.alloc_frac
    qty = alloc / (buy_price * (1 + args.fee))
    if qty < 1e-6:
        save_state(state)
        return

    oid = place_crypto_order(api, alp_sym, qty, buy_price, "buy")
    if oid:
        now = datetime.now(timezone.utc)
        tracked[symbol] = {
            "qty": qty, "side": "long",
            "entry_price": buy_price, "exit_price": sell_price,
            "entry_time": now.isoformat(),
            "entry_order_id": oid,
        }
        log_trade({"event": "entry", "symbol": symbol, "qty": qty,
                   "price": buy_price, "exit_price": sell_price,
                   "edge": edge, "intensity": intensity})
        state["positions"] = tracked

    save_state(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/crypto"))
    parser.add_argument("--cache-root", type=Path, default=Path("cryptoalpacaexperiment/forecast_cache"))
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--max-hold-hours", type=int, default=MAX_HOLD_HOURS)
    parser.add_argument("--alloc-frac", type=float, default=0.5,
                        help="Fraction of equity per trade")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feature_columns, sequence_length, normalizer = load_model(args.checkpoint_dir, epoch=args.epoch)
    model = model.to(device)
    args._normalizer = normalizer

    symbol = args.symbol.upper()
    paper = not args.live

    logger.info("=" * 50)
    logger.info("Crypto Trading Bot: {} ({})", symbol, "LIVE" if not paper else "PAPER")
    logger.info("Max hold: {}h, Fee: {}%, Min edge: {}", args.max_hold_hours, args.fee * 100, args.min_edge)
    logger.info("=" * 50)

    api = None if args.dry_run else get_alpaca_client(paper=paper)
    state = load_state()

    if api:
        equity = get_account_equity(api)
        logger.info("Account equity: ${:.2f}", equity)

    run_cycle(model, feature_columns, sequence_length, device, symbol, args, api, state)

    if not args.loop:
        return

    while True:
        now = datetime.now(timezone.utc)
        next_hour = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
        wait = (next_hour - now).total_seconds()
        logger.info("Sleeping {:.0f}s", wait)
        time.sleep(wait)
        run_cycle(model, feature_columns, sequence_length, device, symbol, args, api, state)


if __name__ == "__main__":
    main()
