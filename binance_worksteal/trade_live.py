#!/usr/bin/env python3
"""Live work-stealing daily trading bot for Binance margin.

Runs daily at UTC midnight:
1. Fetch latest daily bars for all symbols
2. Compute dip targets and proximity scores
3. Place limit orders for best candidates
4. Manage exits (profit target, stop loss, trailing stop, max hold)
5. Handle FDUSD<->USDT swaps for BTC/ETH execution
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from loguru import logger

from binance_worksteal.strategy import (
    WorkStealConfig, compute_ref_price, compute_sma, compute_rsi,
    compute_volume_ratio, get_fee, FDUSD_SYMBOLS,
)

# Binance API
try:
    from binance.client import Client as BinanceClient
    from binance.enums import *
except ImportError:
    logger.warning("binance package not installed, using mock mode")
    BinanceClient = None

STATE_FILE = Path("binance_worksteal/live_state.json")
LOG_FILE = Path("binance_worksteal/trade_log.jsonl")


# Default config (best from sweep)
DEFAULT_CONFIG = WorkStealConfig(
    dip_pct=0.12,
    proximity_pct=0.01,
    profit_target_pct=0.10,
    stop_loss_pct=0.06,
    max_positions=3,
    max_hold_days=7,
    lookback_days=10,
    ref_price_method="high",
    sma_filter_period=10,
    trailing_stop_pct=0.03,
    max_drawdown_exit=0.25,
    enable_shorts=False,
    max_leverage=1.0,
    maker_fee=0.001,
    fdusd_fee=0.0,
    initial_cash=10000.0,
)

# Symbol -> Binance trading pair mapping
SYMBOL_PAIRS = {
    "BTCUSD": {"fdusd": "BTCFDUSD", "usdt": "BTCUSDT"},
    "ETHUSD": {"fdusd": "ETHFDUSD", "usdt": "ETHUSDT"},
    "SOLUSD": {"fdusd": "SOLFDUSD", "usdt": "SOLUSDT"},
    "BNBUSD": {"fdusd": "BNBFDUSD", "usdt": "BNBUSDT"},
}
# All other symbols use USDT pairs
DEFAULT_QUOTE = "usdt"


def get_binance_pair(symbol: str, prefer_fdusd: bool = True) -> str:
    base = symbol.replace("USD", "")
    if prefer_fdusd and symbol in FDUSD_SYMBOLS and symbol in SYMBOL_PAIRS:
        return SYMBOL_PAIRS[symbol]["fdusd"]
    return f"{base}USDT"


def fetch_daily_bars(client, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
    pair = get_binance_pair(symbol, prefer_fdusd=False)  # always use USDT for data
    try:
        klines = client.get_klines(
            symbol=pair,
            interval="1d",
            limit=lookback_days + 5,
        )
    except Exception as e:
        logger.error(f"Failed to fetch klines for {pair}: {e}")
        return pd.DataFrame()

    rows = []
    for k in klines:
        rows.append({
            "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "symbol": symbol,
        })
    return pd.DataFrame(rows)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"positions": {}, "last_exit": {}, "peak_equity": 0}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def log_trade(trade: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


def get_account_equity(client) -> float:
    try:
        info = client.get_margin_account()
        return float(info["totalNetAssetOfBtc"]) * float(
            client.get_symbol_ticker(symbol="BTCUSDT")["price"]
        )
    except Exception as e:
        logger.error(f"Failed to get equity: {e}")
        return 0


def swap_fdusd_to_usdt(client, amount: float):
    """Swap FDUSD to USDT (1:1) if needed for margin operations."""
    try:
        client.create_order(
            symbol="FDUSDUSDT", side="SELL",
            type="MARKET", quantity=f"{amount:.2f}",
        )
        logger.info(f"Swapped {amount:.2f} FDUSD -> USDT")
    except Exception as e:
        logger.warning(f"FDUSD->USDT swap failed: {e}")


def swap_usdt_to_fdusd(client, amount: float):
    """Swap USDT to FDUSD (1:1) for 0% fee trading."""
    try:
        client.create_order(
            symbol="FDUSDUSDT", side="BUY",
            type="MARKET", quantity=f"{amount:.2f}",
        )
        logger.info(f"Swapped {amount:.2f} USDT -> FDUSD")
    except Exception as e:
        logger.warning(f"USDT->FDUSD swap failed: {e}")


def place_limit_buy(client, symbol: str, price: float, quantity: float, config: WorkStealConfig):
    pair = get_binance_pair(symbol, prefer_fdusd=True)
    logger.info(f"Placing limit buy: {pair} qty={quantity:.6f} price={price:.2f}")
    try:
        order = client.create_margin_order(
            symbol=pair, side="BUY", type="LIMIT",
            timeInForce="GTC",
            quantity=f"{quantity:.6f}",
            price=f"{price:.2f}",
        )
        return order
    except Exception as e:
        logger.error(f"Limit buy failed for {pair}: {e}")
        return None


def place_limit_sell(client, symbol: str, price: float, quantity: float):
    pair = get_binance_pair(symbol, prefer_fdusd=True)
    logger.info(f"Placing limit sell: {pair} qty={quantity:.6f} price={price:.2f}")
    try:
        order = client.create_margin_order(
            symbol=pair, side="SELL", type="LIMIT",
            timeInForce="GTC",
            quantity=f"{quantity:.6f}",
            price=f"{price:.2f}",
        )
        return order
    except Exception as e:
        logger.error(f"Limit sell failed for {pair}: {e}")
        return None


def run_daily_cycle(client, symbols: List[str], config: WorkStealConfig, dry_run: bool = True):
    state = load_state()
    positions = state.get("positions", {})
    last_exit = state.get("last_exit", {})

    now = datetime.now(timezone.utc)
    logger.info(f"Daily cycle at {now.isoformat()}, {len(positions)} open positions")

    # Fetch bars for all symbols
    all_bars = {}
    for sym in symbols:
        bars = fetch_daily_bars(client, sym, config.lookback_days + 10)
        if not bars.empty and len(bars) > config.lookback_days:
            all_bars[sym] = bars
    logger.info(f"Fetched bars for {len(all_bars)}/{len(symbols)} symbols")

    equity = get_account_equity(client) if not dry_run else config.initial_cash

    # Check exits
    exits_to_process = []
    for sym, pos in list(positions.items()):
        if sym not in all_bars:
            continue
        bars = all_bars[sym]
        close = float(bars.iloc[-1]["close"])
        high = float(bars.iloc[-1]["high"])
        low = float(bars.iloc[-1]["low"])

        entry_price = pos["entry_price"]
        entry_date = pd.Timestamp(pos["entry_date"])
        peak = max(pos.get("peak_price", entry_price), high)
        pos["peak_price"] = peak

        exit_price = None
        exit_reason = ""

        # Profit target
        target = entry_price * (1 + config.profit_target_pct)
        if high >= target:
            exit_price = target
            exit_reason = "profit_target"
        # Stop loss
        elif low <= entry_price * (1 - config.stop_loss_pct):
            exit_price = entry_price * (1 - config.stop_loss_pct)
            exit_reason = "stop_loss"
        # Trailing stop
        elif config.trailing_stop_pct > 0:
            trail = peak * (1 - config.trailing_stop_pct)
            if low <= trail:
                exit_price = trail
                exit_reason = "trailing_stop"
        # Max hold
        if exit_price is None and config.max_hold_days > 0:
            held = (now - entry_date).days
            if held >= config.max_hold_days:
                exit_price = close
                exit_reason = "max_hold"

        if exit_price is not None:
            exits_to_process.append((sym, exit_price, exit_reason, pos))

    for sym, exit_price, reason, pos in exits_to_process:
        logger.info(f"EXIT {sym}: {reason} at {exit_price:.2f} (entry {pos['entry_price']:.2f})")
        if not dry_run:
            place_limit_sell(client, sym, exit_price, pos["quantity"])
        log_trade({
            "timestamp": now.isoformat(), "symbol": sym, "side": "sell",
            "price": exit_price, "quantity": pos["quantity"],
            "reason": reason, "pnl": (exit_price - pos["entry_price"]) * pos["quantity"],
            "dry_run": dry_run,
        })
        last_exit[sym] = now.isoformat()
        del positions[sym]

    # Check entries
    if len(positions) < config.max_positions:
        candidates = []
        for sym, bars in all_bars.items():
            if sym in positions:
                continue
            if sym in last_exit:
                exit_ts = pd.Timestamp(last_exit[sym])
                if (now - exit_ts).days < config.reentry_cooldown_days:
                    continue
            if len(bars) < config.lookback_days:
                continue

            close = float(bars.iloc[-1]["close"])

            # SMA filter
            if config.sma_filter_period > 0:
                sma = compute_sma(bars, config.sma_filter_period)
                if close < sma:
                    continue

            ref_high = compute_ref_price(bars, config.ref_price_method, config.lookback_days)
            buy_target = ref_high * (1 - config.dip_pct)
            dist = (close - buy_target) / ref_high

            if dist <= config.proximity_pct:
                dip_score = -dist
                fill_price = buy_target
                candidates.append((sym, dip_score, fill_price, close))

        candidates.sort(key=lambda x: x[1], reverse=True)
        slots = config.max_positions - len(positions)

        for sym, score, fill_price, close in candidates[:slots]:
            if sym in positions:
                continue
            fee_rate = get_fee(sym, config)
            alloc = equity * config.max_position_pct
            quantity = alloc / (fill_price * (1 + fee_rate))
            if quantity <= 0:
                continue

            logger.info(f"ENTRY {sym}: buy limit at {fill_price:.2f} "
                        f"(close={close:.2f}, dip_score={score:.4f}, qty={quantity:.6f})")

            if not dry_run:
                # Handle FDUSD swap if needed
                if sym in FDUSD_SYMBOLS:
                    swap_usdt_to_fdusd(client, alloc)
                place_limit_buy(client, sym, fill_price, quantity, config)

            positions[sym] = {
                "entry_price": fill_price,
                "entry_date": now.isoformat(),
                "quantity": quantity,
                "peak_price": close,
                "target_sell": fill_price * (1 + config.profit_target_pct),
                "stop_price": fill_price * (1 - config.stop_loss_pct),
            }
            log_trade({
                "timestamp": now.isoformat(), "symbol": sym, "side": "buy",
                "price": fill_price, "quantity": quantity,
                "reason": f"dip_buy(score={score:.4f})",
                "dry_run": dry_run,
            })

    # Save state
    state["positions"] = positions
    state["last_exit"] = last_exit
    state["peak_equity"] = max(state.get("peak_equity", 0), equity)
    save_state(state)

    logger.info(f"Cycle complete: {len(positions)} positions, equity=${equity:.0f}")
    for sym, pos in positions.items():
        logger.info(f"  {sym}: entry={pos['entry_price']:.2f} "
                    f"target={pos['target_sell']:.2f} stop={pos['stop_price']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--dip-pct", type=float, default=0.12)
    parser.add_argument("--profit-target", type=float, default=0.10)
    parser.add_argument("--stop-loss", type=float, default=0.06)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--sma-filter", type=int, default=10)
    parser.add_argument("--trailing-stop", type=float, default=0.03)
    args = parser.parse_args()

    if args.live:
        args.dry_run = False

    from binance_worksteal.backtest import FULL_UNIVERSE
    symbols = args.symbols or FULL_UNIVERSE

    config = WorkStealConfig(
        dip_pct=args.dip_pct,
        profit_target_pct=args.profit_target,
        stop_loss_pct=args.stop_loss,
        max_positions=args.max_positions,
        sma_filter_period=args.sma_filter,
        trailing_stop_pct=args.trailing_stop,
        max_drawdown_exit=0.25,
        proximity_pct=0.01,
        lookback_days=10,
        ref_price_method="high",
        max_hold_days=7,
    )

    # Initialize Binance client
    if not args.dry_run and BinanceClient:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from env_real import BINANCE_API_KEY, BINANCE_SECRET
            client = BinanceClient(BINANCE_API_KEY, BINANCE_SECRET)
        except ImportError:
            logger.error("env_real.py not found - need BINANCE_API_KEY and BINANCE_SECRET")
            return 1
    else:
        client = None
        logger.info("Running in DRY RUN mode")

    if args.daemon:
        logger.info("Starting daemon mode - runs daily at UTC midnight")
        while True:
            now = datetime.now(timezone.utc)
            next_run = (now + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
            if now.hour == 0 and now.minute < 10:
                run_daily_cycle(client, symbols, config, dry_run=args.dry_run)
                next_run = (now + timedelta(days=1)).replace(hour=0, minute=5, second=0)
            sleep_secs = (next_run - datetime.now(timezone.utc)).total_seconds()
            logger.info(f"Next run in {sleep_secs/3600:.1f}h at {next_run}")
            time.sleep(max(60, sleep_secs))
    else:
        run_daily_cycle(client, symbols, config, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main() or 0)
