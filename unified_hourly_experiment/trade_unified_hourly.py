#!/usr/bin/env python3
"""Unified hourly trading bot for stocks (Alpaca) and crypto (Binance)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from env_real import ALP_KEY_ID, ALP_SECRET_KEY

import pandas as pd
import torch
from loguru import logger

from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer
from binanceneural.config import DatasetConfig
from binanceneural.model import build_policy, PolicyConfig
from binanceneural.inference import generate_latest_action
from src.torch_load_utils import torch_load_compat
from src.symbol_utils import is_crypto_symbol


def load_model(checkpoint_dir: Path):
    """Load model from checkpoint directory."""
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    best_ckpt = checkpoints[-1]
    logger.info("Loading checkpoint: {}", best_ckpt.name)

    ckpt = torch_load_compat(best_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    config_path = checkpoint_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    feature_columns = config.get("feature_columns", [])
    sequence_length = config.get("sequence_length", 32)

    input_dim = len(feature_columns)
    hidden_dim = config.get("transformer_dim", 128)
    num_heads = config.get("transformer_heads", 4)
    num_layers = config.get("transformer_layers", 3)
    policy_cfg = PolicyConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        model_arch="gemma",
        max_len=sequence_length,
    )
    model = build_policy(policy_cfg)

    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, feature_columns, sequence_length


def is_market_open_now() -> bool:
    """Check if US stock market is currently open."""
    from zoneinfo import ZoneInfo
    ny = datetime.now(ZoneInfo("America/New_York"))
    if ny.weekday() >= 5:
        return False
    t = ny.time()
    from datetime import time as dt_time
    return dt_time(9, 30) <= t <= dt_time(16, 0)


def get_alpaca_client(paper: bool = True):
    """Get Alpaca trading client."""
    from alpaca.trading.client import TradingClient
    return TradingClient(ALP_KEY_ID, ALP_SECRET_KEY, paper=paper)


def get_current_positions(api) -> Dict[str, float]:
    """Get current stock positions."""
    positions = {}
    try:
        for pos in api.get_all_positions():
            positions[pos.symbol] = float(pos.qty)
    except Exception as e:
        logger.error("Failed to get positions: {}", e)
    return positions


def get_buying_power(api) -> float:
    """Get available buying power."""
    try:
        account = api.get_account()
        return float(account.buying_power)
    except Exception as e:
        logger.error("Failed to get account: {}", e)
        return 0.0


def execute_stock_trades(api, signals: Dict, positions: Dict, allocation_per_symbol: float = 1000.0):
    """Execute stock trades via Alpaca."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    buying_power = get_buying_power(api)
    logger.info("Buying power: ${:.2f}", buying_power)

    for symbol, action in signals.items():
        try:
            buy_price = action.get("buy_price", 0)
            trade_amount = min(action.get("trade_amount", 0), 1.0)
            if buy_price <= 0 or trade_amount <= 0:
                continue

            target_value = allocation_per_symbol * trade_amount
            target_qty = int(target_value / buy_price)
            current_qty = positions.get(symbol, 0)

            if target_qty <= 0:
                continue

            if current_qty > 0:
                logger.info("{}: already holding {} shares", symbol, current_qty)
                continue

            if target_value > buying_power:
                logger.warning("{}: insufficient buying power (${:.0f} needed, ${:.0f} available)",
                              symbol, target_value, buying_power)
                continue

            logger.info("{}: buying {} shares @ ${:.2f}", symbol, target_qty, buy_price)
            order = MarketOrderRequest(
                symbol=symbol,
                qty=target_qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            api.submit_order(order)
            buying_power -= target_value

        except Exception as e:
            logger.error("{}: order failed - {}", symbol, e)


def execute_crypto_trades(api, signals: Dict, positions: Dict, allocation_per_symbol: float = 1000.0):
    """Execute crypto trades via Alpaca."""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    buying_power = get_buying_power(api)
    logger.info("Buying power: ${:.2f}", buying_power)

    for symbol, action in signals.items():
        try:
            buy_price = action.get("buy_price", 0)
            trade_amount = min(action.get("trade_amount", 0), 1.0)
            if buy_price <= 0 or trade_amount <= 0:
                continue

            target_value = allocation_per_symbol * trade_amount
            alpaca_sym = symbol[:3] + "/" + symbol[3:] if "/" not in symbol else symbol

            current_qty = positions.get(alpaca_sym, positions.get(symbol, 0))

            if current_qty > 0:
                logger.info("{}: already holding {:.4f}", symbol, current_qty)
                continue

            if target_value > buying_power:
                logger.warning("{}: insufficient buying power (${:.0f} needed, ${:.0f} available)",
                              symbol, target_value, buying_power)
                continue

            notional = min(target_value, buying_power)
            logger.info("{}: buying ${:.2f} notional @ ${:.2f}", symbol, notional, buy_price)
            order = MarketOrderRequest(
                symbol=alpaca_sym,
                notional=round(notional, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
            )
            api.submit_order(order)
            buying_power -= notional

        except Exception as e:
            logger.error("{}: order failed - {}", symbol, e)


def generate_signal_for_symbol(
    symbol: str,
    model: torch.nn.Module,
    feature_columns: list,
    sequence_length: int,
    data_root: Path,
    cache_root: Path,
    device: torch.device,
) -> Optional[Dict]:
    """Generate trading signal for a single symbol."""
    data_config = DatasetConfig(
        symbol=symbol,
        data_root=str(data_root),
        forecast_cache_root=str(cache_root),
        forecast_horizons=[1, 24],
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
        action = generate_latest_action(
            model=model,
            frame=frame,
            feature_columns=feature_columns,
            normalizer=data_module.normalizer,
            sequence_length=sequence_length,
            horizon=1,
            device=device,
        )
        return action
    except Exception as e:
        logger.error("Failed to generate action for {}: {}", symbol, e)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--stock-symbols", default="NVDA,MSFT,META,GOOG")
    parser.add_argument("--crypto-symbols", default="")
    parser.add_argument("--stock-data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--crypto-data-root", type=Path, default=Path("trainingdatahourly/crypto"))
    parser.add_argument("--stock-cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--crypto-cache-root", type=Path, default=Path("binanceneural/forecast_cache"))
    parser.add_argument("--dry-run", action="store_true", help="Print signals only, no trading")
    parser.add_argument("--paper", action="store_true", help="Use paper trading (default)")
    parser.add_argument("--live", action="store_true", help="Use live trading (DANGEROUS)")
    parser.add_argument("--min-edge", type=float, default=0.012, help="Min edge to execute trade")
    parser.add_argument("--fee-rate", type=float, default=0.001, help="Trading fee rate")
    parser.add_argument("--allocation-per-symbol", type=float, default=1000.0, help="USD allocation per symbol")
    parser.add_argument("--ignore-market-hours", action="store_true", help="Generate signals even outside market hours")
    parser.add_argument("--loop", action="store_true", help="Run in continuous loop")
    args = parser.parse_args()

    paper = not args.live

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feature_columns, sequence_length = load_model(args.checkpoint_dir)
    model = model.to(device)

    stocks = [s.strip().upper() for s in args.stock_symbols.split(",") if s.strip()]
    cryptos = [s.strip().upper() for s in args.crypto_symbols.split(",") if s.strip()]

    logger.info("=" * 60)
    logger.info("Unified Hourly Trading Bot")
    logger.info("=" * 60)
    logger.info("Stocks: {}", stocks)
    logger.info("Crypto: {}", cryptos)
    logger.info("Dry run: {}", args.dry_run)

    # Check market hours for stocks
    market_open = is_market_open_now()
    logger.info("Market open: {}", market_open)

    signals = {}

    # Generate stock signals
    for symbol in stocks:
        if not market_open and not args.ignore_market_hours:
            logger.info("{}: Market closed, skipping", symbol)
            continue

        action = generate_signal_for_symbol(
            symbol, model, feature_columns, sequence_length,
            args.stock_data_root, args.stock_cache_root, device
        )
        if action:
            pred_high = action.get("predicted_high", 0)
            buy_price = action.get("buy_price", 0)
            edge = (pred_high - buy_price) / buy_price - args.fee_rate if buy_price > 0 else 0
            if edge >= args.min_edge:
                signals[symbol] = action
                logger.info("{}: buy={:.2f} sell={:.2f} amount={:.4f} edge={:.4f}",
                           symbol, action["buy_price"], action["sell_price"], action["trade_amount"], edge)
            else:
                logger.info("{}: edge={:.4f} below threshold {:.4f}", symbol, edge, args.min_edge)

    # Generate crypto signals
    for symbol in cryptos:
        action = generate_signal_for_symbol(
            symbol, model, feature_columns, sequence_length,
            args.crypto_data_root, args.crypto_cache_root, device
        )
        if action:
            pred_high = action.get("predicted_high", 0)
            buy_price = action.get("buy_price", 0)
            edge = (pred_high - buy_price) / buy_price - args.fee_rate if buy_price > 0 else 0
            if edge >= args.min_edge:
                signals[symbol] = action
                logger.info("{}: buy={:.2f} sell={:.2f} amount={:.4f} edge={:.4f}",
                           symbol, action["buy_price"], action["sell_price"], action["trade_amount"], edge)
            else:
                logger.info("{}: edge={:.4f} below threshold {:.4f}", symbol, edge, args.min_edge)

    if args.dry_run:
        logger.info("Dry run - would execute signals: {}", list(signals.keys()))
        if not args.loop:
            return
    else:
        api = get_alpaca_client(paper=paper)
        positions = get_current_positions(api)
        logger.info("Current positions: {}", positions)

        stock_signals = {k: v for k, v in signals.items() if k in stocks}
        crypto_signals = {k: v for k, v in signals.items() if k in cryptos}

        if stock_signals:
            execute_stock_trades(api, stock_signals, positions, args.allocation_per_symbol)

        if crypto_signals:
            execute_crypto_trades(api, crypto_signals, positions, args.allocation_per_symbol)

        if not stock_signals and not crypto_signals:
            logger.info("No signals above threshold")

    if not args.loop:
        return

    while True:
        now = datetime.now(timezone.utc)
        next_hour = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
        wait_secs = (next_hour - now).total_seconds()
        logger.info("Sleeping {:.0f}s until next hour", wait_secs)
        time.sleep(wait_secs)

        signals = {}
        market_open = is_market_open_now()
        for symbol in stocks:
            if not market_open and not args.ignore_market_hours:
                continue
            action = generate_signal_for_symbol(
                symbol, model, feature_columns, sequence_length,
                args.stock_data_root, args.stock_cache_root, device
            )
            if action:
                pred_high = action.get("predicted_high", 0)
                buy_price = action.get("buy_price", 0)
                edge = (pred_high - buy_price) / buy_price - args.fee_rate if buy_price > 0 else 0
                if edge >= args.min_edge:
                    signals[symbol] = action
                    logger.info("{}: buy={:.2f} sell={:.2f} edge={:.4f}",
                               symbol, buy_price, action["sell_price"], edge)

        if not args.dry_run and signals:
            positions = get_current_positions(api)
            execute_stock_trades(api, signals, positions, args.allocation_per_symbol)


if __name__ == "__main__":
    main()
