#!/usr/bin/env python3
"""PPO-based crypto trader for Alpaca."""

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

SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD"]
SYMBOLS_SLASH = ["BTC/USD", "ETH/USD", "SOL/USD", "LINK/USD"]


def to_slash(sym: str) -> str:
    return sym[:-3] + "/" + sym[-3:]


def from_slash(sym: str) -> str:
    return sym.replace("/", "")


def fetch_hourly_data(symbol: str, hours: int = 200) -> pd.DataFrame:
    """Fetch hourly OHLCV data from Alpaca."""
    from alpaca.data import CryptoHistoricalDataClient, CryptoBarsRequest, TimeFrame

    client = CryptoHistoricalDataClient()
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)

    sym_slash = to_slash(symbol) if "/" not in symbol else symbol

    request = CryptoBarsRequest(
        symbol_or_symbols=sym_slash,
        timeframe=TimeFrame.Hour,
        start=start,
        end=end,
    )
    bars = client.get_crypto_bars(request)
    df = bars.df.reset_index()
    df = df.rename(columns={"symbol": "sym"})
    return df


def get_current_prices() -> dict:
    """Get current prices for all symbols."""
    from alpaca.data import CryptoHistoricalDataClient, CryptoLatestQuoteRequest

    client = CryptoHistoricalDataClient()
    request = CryptoLatestQuoteRequest(symbol_or_symbols=SYMBOLS_SLASH)
    quotes = client.get_crypto_latest_quote(request)
    return {from_slash(sym): quotes[sym].ask_price for sym in SYMBOLS_SLASH}


def get_current_position(api) -> tuple:
    """Get current crypto position."""
    try:
        positions = api.get_all_positions()
        for pos in positions:
            sym = from_slash(pos.symbol) if "/" in pos.symbol else pos.symbol
            if sym in SYMBOLS:
                qty = float(pos.qty)
                side = "long" if qty > 0 else "short"
                return sym, side, abs(qty)
    except Exception as e:
        logger.warning(f"Error getting positions: {e}")
    return None, None, 0.0


def execute_signal(api, signal, prices: dict, allocation_usd: float = 1000.0):
    """Execute trading signal via Alpaca."""
    import alpaca_wrapper

    if signal.action == "flat":
        cur_sym, cur_side, cur_qty = get_current_position(api)
        if cur_sym:
            logger.info(f"Closing position: {cur_side} {cur_qty} {cur_sym}")
            alpaca_wrapper.exit_position(cur_sym)
        return

    if signal.confidence < 0.6:
        logger.info(f"Low confidence ({signal.confidence:.1%}), skipping")
        return

    cur_sym, cur_side, cur_qty = get_current_position(api)

    if cur_sym == signal.symbol and cur_side == signal.direction:
        logger.info(f"Already in {signal.direction} {signal.symbol}")
        return

    if cur_sym and cur_sym != signal.symbol:
        logger.info(f"Closing {cur_side} {cur_sym} before entering {signal.symbol}")
        alpaca_wrapper.exit_position(cur_sym)
        time.sleep(1)

    price = prices.get(signal.symbol, 0)
    if price <= 0:
        logger.error(f"Invalid price for {signal.symbol}")
        return

    qty = allocation_usd / price

    logger.info(f"Entering {signal.direction} {signal.symbol}: {qty:.6f} @ {price:.2f}")

    if signal.direction == "long":
        alpaca_wrapper.buy_crypto(signal.symbol, qty)
    else:
        alpaca_wrapper.sell_crypto(signal.symbol, qty)


def run_once(trader: PPOTrader, api, allocation_usd: float):
    """Run one trading cycle."""
    prices = get_current_prices()
    logger.info(f"Prices: {prices}")

    features = np.zeros((len(SYMBOLS), 16), dtype=np.float32)
    for i, sym in enumerate(SYMBOLS):
        try:
            df = fetch_hourly_data(sym, hours=200)
            features[i] = compute_hourly_features(df)
        except Exception as e:
            logger.error(f"Error fetching {sym}: {e}")
            features[i] = 0

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
    logger.info(f"Signal: {signal.action} (conf={signal.confidence:.1%}, val={signal.value_estimate:.2f})")

    execute_signal(api, signal, prices, allocation_usd)


def seconds_until_next_hour(buffer: int = 30) -> float:
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return max(1.0, (next_hour + timedelta(seconds=buffer) - now).total_seconds())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--allocation-usd", type=float, default=1000.0)
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--paper", action="store_true", help="Use paper trading")
    args = parser.parse_args()

    if args.paper:
        os.environ["PAPER"] = "1"

    from alpaca.trading.client import TradingClient
    from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, PAPER

    paper = os.environ.get("PAPER", "1") == "1" or args.paper
    key_id = ALP_KEY_ID if paper else ALP_KEY_ID_PROD
    secret = ALP_SECRET_KEY if paper else ALP_SECRET_KEY_PROD

    api = TradingClient(key_id, secret, paper=paper)
    logger.info(f"Connected to Alpaca ({'paper' if paper else 'live'})")

    trader = PPOTrader(args.checkpoint, args.device)

    if args.once:
        run_once(trader, api, args.allocation_usd)
        return

    logger.info("Starting hourly trading loop")
    while True:
        try:
            run_once(trader, api, args.allocation_usd)
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")

        wait = seconds_until_next_hour()
        logger.info(f"Sleeping {wait/60:.1f} min until next hour")
        time.sleep(wait)


if __name__ == "__main__":
    main()
