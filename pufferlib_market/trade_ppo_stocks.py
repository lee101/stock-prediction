#!/usr/bin/env python3
"""PPO-based stock trader for Alpaca - shortable stocks."""

from __future__ import annotations
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pufferlib_market.inference import PPOTrader, compute_hourly_features
from loguru import logger

SYMBOLS = ["EBAY", "MTCH", "ANGI", "Z", "EXPE", "BKNG", "NWSA"]
POSITION_LIMITS = {
    "EBAY": 100, "MTCH": 50, "ANGI": 500, "Z": 50,
    "EXPE": 30, "BKNG": 5, "NWSA": 100
}

def fetch_hourly_data(symbol: str, hours: int = 200) -> pd.DataFrame:
    from alpaca.data import StockHistoricalDataClient, StockBarsRequest, TimeFrame
    client = StockHistoricalDataClient()
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Hour, start=start, end=end)
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()
    return df.rename(columns={"symbol": "sym"})

def get_current_prices() -> dict:
    from alpaca.data import StockHistoricalDataClient, StockLatestQuoteRequest
    client = StockHistoricalDataClient()
    request = StockLatestQuoteRequest(symbol_or_symbols=SYMBOLS)
    quotes = client.get_stock_latest_quote(request)
    return {sym: quotes[sym].ask_price for sym in SYMBOLS}

def get_current_position(api) -> tuple:
    try:
        positions = api.get_all_positions()
        for pos in positions:
            if pos.symbol in SYMBOLS:
                qty = float(pos.qty)
                return pos.symbol, "long" if qty > 0 else "short", abs(qty)
    except Exception as e:
        logger.warning(f"Error getting positions: {e}")
    return None, None, 0.0

def execute_signal(api, signal, prices: dict, allocation_usd: float = 1000.0):
    import alpaca_wrapper
    cur_sym, cur_side, cur_qty = get_current_position(api)

    if signal.action == "flat":
        if cur_sym and cur_qty > 0:
            logger.info(f"Closing: {cur_side} {cur_qty} {cur_sym}")
            alpaca_wrapper.open_market_order_violently(cur_sym, cur_qty, "sell" if cur_side == "long" else "buy")
        return

    if signal.confidence < 0.6:
        logger.info(f"Low confidence ({signal.confidence:.1%}), skip")
        return

    if cur_sym == signal.symbol and cur_side == signal.direction:
        logger.info(f"Already in {signal.direction} {signal.symbol}")
        return

    if cur_sym and cur_qty > 0:
        logger.info(f"Closing {cur_side} {cur_sym}")
        alpaca_wrapper.open_market_order_violently(cur_sym, cur_qty, "sell" if cur_side == "long" else "buy")
        time.sleep(1)

    price = prices.get(signal.symbol, 0)
    if price <= 0:
        return

    raw_qty = allocation_usd / price
    max_qty = POSITION_LIMITS.get(signal.symbol, 50)
    qty = min(raw_qty, max_qty)

    logger.info(f"Entering {signal.direction} {signal.symbol}: {qty:.2f} @ ${price:.2f}")
    alpaca_wrapper.open_market_order_violently(signal.symbol, qty, "buy" if signal.direction == "long" else "sell")

def run_once(trader: PPOTrader, api, allocation_usd: float):
    prices = get_current_prices()
    logger.info(f"Prices: {prices}")

    features = np.zeros((len(SYMBOLS), 16), dtype=np.float32)
    for i, sym in enumerate(SYMBOLS):
        try:
            df = fetch_hourly_data(sym, hours=200)
            features[i] = compute_hourly_features(df)
        except Exception as e:
            logger.error(f"Error fetching {sym}: {e}")

    cur_sym, cur_side, cur_qty = get_current_position(api)
    if cur_sym:
        sym_idx = SYMBOLS.index(cur_sym) if cur_sym in SYMBOLS else 0
        is_short = cur_side == "short"
        trader.current_position = (len(SYMBOLS) + sym_idx) if is_short else sym_idx
        trader.position_qty = cur_qty
    else:
        trader.current_position = None
        trader.position_qty = 0.0

    signal = trader.get_signal(features, prices)
    logger.info(f"Signal: {signal.action} (conf={signal.confidence:.1%})")
    execute_signal(api, signal, prices, allocation_usd)

def is_market_hours() -> bool:
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    hour_utc = now.hour + now.minute / 60
    return 14.5 <= hour_utc < 21.0

def seconds_until_next_market_hour(buffer: int = 30) -> float:
    now = datetime.now(timezone.utc)
    if is_market_hours():
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        return max(1.0, (next_hour + timedelta(seconds=buffer) - now).total_seconds())
    else:
        if now.weekday() == 4 and now.hour >= 21:
            days_until = 3
        elif now.weekday() == 5:
            days_until = 2
        elif now.weekday() == 6:
            days_until = 1
        else:
            days_until = 0 if now.hour < 14 else 1
        next_open = (now + timedelta(days=days_until)).replace(hour=14, minute=30, second=0, microsecond=0)
        return max(60.0, (next_open - now).total_seconds())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--allocation-usd", type=float, default=1000.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    if args.paper:
        os.environ["PAPER"] = "1"

    from alpaca.trading.client import TradingClient
    from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

    paper = os.environ.get("PAPER", "1") == "1" or args.paper
    key_id = ALP_KEY_ID if paper else ALP_KEY_ID_PROD
    secret = ALP_SECRET_KEY if paper else ALP_SECRET_KEY_PROD

    api = TradingClient(key_id, secret, paper=paper)
    logger.info(f"Connected to Alpaca ({'paper' if paper else 'live'})")

    trader = PPOTrader(args.checkpoint, args.device, long_only=False)
    trader.SYMBOLS = SYMBOLS

    if args.once:
        run_once(trader, api, args.allocation_usd)
        return

    logger.info("Starting hourly stock trading loop")
    while True:
        if is_market_hours():
            try:
                run_once(trader, api, args.allocation_usd)
            except Exception as e:
                logger.error(f"Error: {e}")
        else:
            logger.info("Market closed")
        wait = seconds_until_next_market_hour()
        logger.info(f"Sleeping {wait/60:.1f} min")
        time.sleep(wait)

if __name__ == "__main__":
    main()
