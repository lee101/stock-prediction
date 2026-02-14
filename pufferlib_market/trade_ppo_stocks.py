#!/usr/bin/env python3
"""PPO-based stock trader for Alpaca."""

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

from pufferlib_market.inference import PPOTrader, compute_hourly_features, Policy
from loguru import logger
import torch

SYMBOLS = ["NVDA", "MSFT", "META", "GOOG", "PLTR", "DBX", "TRIP"]


class StockPPOTrader(PPOTrader):
    """PPO trader adapted for stocks."""

    SYMBOLS = SYMBOLS

    def __init__(self, checkpoint_path: str, device: str = "cpu", long_only: bool = False):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.long_only = long_only

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "model" in ckpt:
            state_dict = ckpt["model"]
            config = ckpt.get("config", {})
        else:
            state_dict = ckpt
            config = {}

        self.num_symbols = len(self.SYMBOLS)
        self.obs_size = self.num_symbols * 16 + 5 + self.num_symbols

        # Infer num_actions from checkpoint
        actor_bias_key = [k for k in state_dict if 'actor' in k and 'bias' in k and '2' in k]
        if actor_bias_key:
            self.num_actions = state_dict[actor_bias_key[0]].shape[0]
        elif long_only:
            self.num_actions = 1 + self.num_symbols
        else:
            self.num_actions = 1 + 2 * self.num_symbols

        # Check for allocation bins
        basic_actions = 1 + 2 * self.num_symbols
        if self.num_actions > basic_actions:
            self.alloc_bins = (self.num_actions - 1) // (2 * self.num_symbols)
            self.per_symbol_actions = self.alloc_bins
            self.side_block = self.num_symbols * self.per_symbol_actions
        else:
            self.alloc_bins = 0
            self.per_symbol_actions = 1
            self.side_block = self.num_symbols

        # Infer hidden size from checkpoint
        input_proj_key = [k for k in state_dict if 'input_proj' in k and 'weight' in k]
        if input_proj_key:
            hidden = state_dict[input_proj_key[0]].shape[0]
        else:
            hidden = config.get("hidden_size", 256)
        blocks = config.get("num_blocks", 3)

        self.policy = Policy(self.obs_size, self.num_actions, hidden, blocks)
        self.policy.load_state_dict(state_dict)
        self.policy.to(self.device)
        self.policy.eval()

        self.current_position = None
        self.cash = 10000.0
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.hold_hours = 0
        self.step = 0
        self.max_steps = config.get("max_steps", 720)

        logger.info("Loaded stock model from {}", checkpoint_path)


def fetch_hourly_data(symbol: str, hours: int = 200) -> pd.DataFrame:
    """Fetch hourly OHLCV data from Alpaca (IEX feed)."""
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    from env_real import ALP_KEY_ID, ALP_SECRET_KEY

    client = StockHistoricalDataClient(ALP_KEY_ID, ALP_SECRET_KEY)
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()
    df = df.rename(columns={"symbol": "sym"})
    return df


def get_current_prices() -> dict:
    """Get current prices for all symbols (IEX feed)."""
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.data.enums import DataFeed
    from env_real import ALP_KEY_ID, ALP_SECRET_KEY

    client = StockHistoricalDataClient(ALP_KEY_ID, ALP_SECRET_KEY)
    request = StockLatestQuoteRequest(symbol_or_symbols=SYMBOLS, feed=DataFeed.IEX)
    quotes = client.get_stock_latest_quote(request)
    prices = {}
    for sym in SYMBOLS:
        if sym in quotes and quotes[sym].ask_price > 0:
            prices[sym] = quotes[sym].ask_price
        elif sym in quotes and quotes[sym].bid_price > 0:
            prices[sym] = quotes[sym].bid_price
        else:
            prices[sym] = 0.0
    return prices


def get_current_position(api) -> tuple:
    """Get current stock position."""
    try:
        positions = api.get_all_positions()
        for pos in positions:
            sym = pos.symbol
            if sym in SYMBOLS:
                qty = float(pos.qty)
                side = "long" if qty > 0 else "short"
                return sym, side, abs(qty)
    except Exception as e:
        logger.warning("Error getting positions: {}", e)
    return None, None, 0.0


def execute_signal(api, signal, prices: dict, allocation_usd: float = 1000.0):
    """Execute trading signal via Alpaca."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    cur_sym, cur_side, cur_qty = get_current_position(api)

    if signal.action == "flat":
        if cur_sym and cur_qty > 0:
            logger.info("Closing position: {} {} {}", cur_side, cur_qty, cur_sym)
            order = MarketOrderRequest(
                symbol=cur_sym,
                qty=int(cur_qty),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            api.submit_order(order)
        return

    if signal.confidence < 0.6:
        logger.info("Low confidence ({:.1%}), skipping", signal.confidence)
        return

    if cur_sym == signal.symbol and cur_side == signal.direction:
        logger.info("Already in {} {}", signal.direction, signal.symbol)
        return

    if cur_sym and cur_sym != signal.symbol and cur_qty > 0:
        logger.info("Closing {} {} before entering {}", cur_side, cur_sym, signal.symbol)
        order = MarketOrderRequest(
            symbol=cur_sym,
            qty=int(cur_qty),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        api.submit_order(order)
        time.sleep(1)

    price = prices.get(signal.symbol, 0)
    if price <= 0:
        logger.error("Invalid price for {}", signal.symbol)
        return

    # Apply allocation percentage from model
    alloc_pct = getattr(signal, 'allocation_pct', 1.0)
    effective_alloc = allocation_usd * alloc_pct
    qty = int(effective_alloc / price)
    if qty <= 0:
        logger.warning("Qty too small for {} @ ${:.2f} (alloc={:.0%})", signal.symbol, price, alloc_pct)
        return

    logger.info("Entering {} {}: {} shares @ ${:.2f} (alloc={:.0%})", signal.direction, signal.symbol, qty, price, alloc_pct)
    order = MarketOrderRequest(
        symbol=signal.symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    api.submit_order(order)


def is_market_open() -> bool:
    """Check if US stock market is open."""
    from zoneinfo import ZoneInfo
    from datetime import time as dt_time
    ny = datetime.now(ZoneInfo("America/New_York"))
    if ny.weekday() >= 5:
        return False
    t = ny.time()
    return dt_time(9, 30) <= t <= dt_time(16, 0)


def run_once(trader: StockPPOTrader, api, allocation_usd: float, ignore_market_hours: bool = False):
    """Run one trading cycle."""
    if not is_market_open() and not ignore_market_hours:
        logger.info("Market closed, skipping")
        return

    prices = get_current_prices()
    logger.info("Prices: {}", {k: f"${v:.2f}" for k, v in prices.items()})

    features = np.zeros((len(SYMBOLS), 16), dtype=np.float32)
    for i, sym in enumerate(SYMBOLS):
        try:
            df = fetch_hourly_data(sym, hours=200)
            features[i] = compute_hourly_features(df)
        except Exception as e:
            logger.error("Error fetching {}: {}", sym, e)
            features[i] = 0

    cur_sym, cur_side, cur_qty = get_current_position(api)
    if cur_sym:
        sym_idx = SYMBOLS.index(cur_sym) if cur_sym in SYMBOLS else 0
        trader.current_position = sym_idx
        trader.position_qty = cur_qty
    else:
        trader.current_position = None
        trader.position_qty = 0.0

    signal = trader.get_signal(features, prices)
    logger.info("Signal: {} (conf={:.1%}, val={:.2f})", signal.action, signal.confidence, signal.value_estimate)

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
    parser.add_argument("--ignore-market-hours", action="store_true", help="Trade outside market hours")
    args = parser.parse_args()

    if args.paper:
        os.environ["PAPER"] = "1"

    from alpaca.trading.client import TradingClient
    from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

    paper = os.environ.get("PAPER", "1") == "1" or args.paper
    key_id = ALP_KEY_ID if paper else ALP_KEY_ID_PROD
    secret = ALP_SECRET_KEY if paper else ALP_SECRET_KEY_PROD

    api = TradingClient(key_id, secret, paper=paper)
    logger.info("Connected to Alpaca ({})", "paper" if paper else "live")

    trader = StockPPOTrader(args.checkpoint, args.device, long_only=True)

    if args.once:
        run_once(trader, api, args.allocation_usd, args.ignore_market_hours)
        return

    logger.info("Starting hourly trading loop")
    while True:
        try:
            run_once(trader, api, args.allocation_usd, args.ignore_market_hours)
        except Exception as e:
            logger.error("Error in trading cycle: {}", e)

        wait = seconds_until_next_hour()
        logger.info("Sleeping {:.1f} min until next hour", wait / 60)
        time.sleep(wait)


if __name__ == "__main__":
    main()
