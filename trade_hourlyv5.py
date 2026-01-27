#!/usr/bin/env python3
"""Trade script for Neural Hourly Trading V5.

Runs live/paper trading using the V5 model.
Use PAPER=1 environment variable for paper trading (default).

Example:
    PAPER=1 python trade_hourlyv5.py --symbol LINKUSD --daemon
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch

import alpaca_wrapper
from alpaca_data_wrapper import append_recent_crypto_data
from alpaca_wrapper import _get_min_order_notional
from src.process_utils import (
    spawn_close_position_at_maxdiff_takeprofit,
    spawn_open_position_at_maxdiff_takeprofit,
)
from src.price_guard import enforce_gap, record_buy, record_sell

from neuralhourlytradingv5.config import DefaultStrategyConfig, SimulationConfigV5
from neuralhourlytradingv5.data import FeatureNormalizer, HOURLY_FEATURES_V5
from neuralhourlytradingv5.model import HourlyCryptoPolicyV5

logger = logging.getLogger("hourlyv5")


@dataclass
class TradingPlan:
    """Trading plan output from model."""
    timestamp: pd.Timestamp
    buy_price: float
    sell_price: float
    position_length: int  # Hours to hold
    position_size: float  # 0-1 fraction of capital


def load_model(checkpoint_path: str, device: str = "cuda") -> Tuple[HourlyCryptoPolicyV5, FeatureNormalizer, list]:
    """Load V5 model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get config
    policy_config = checkpoint["config"]["policy"]

    # Create model
    model = HourlyCryptoPolicyV5(policy_config)

    # Handle torch.compile prefix in state dict keys
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Get normalizer
    normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
    feature_columns = checkpoint["feature_columns"]

    return model, normalizer, feature_columns


def load_and_prepare_data(symbol: str, feature_columns: list, data_root: str = "trainingdatahourly/crypto") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare data for inference."""
    data_path = Path(data_root) / f"{symbol}.csv"

    # Append recent data (function expects list of symbols)
    append_recent_crypto_data([symbol])

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Add missing features with defaults
    for feat in HOURLY_FEATURES_V5:
        if feat not in df.columns:
            df[feat] = 0.0

    return df, df[list(feature_columns)]


def get_trading_plan(
    model: HourlyCryptoPolicyV5,
    normalizer: FeatureNormalizer,
    features_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    sequence_length: int = 168,
    device: str = "cuda",
) -> Optional[TradingPlan]:
    """Get trading plan from model."""
    if len(features_df) < sequence_length:
        logger.warning(f"Not enough data: {len(features_df)} < {sequence_length}")
        return None

    # Get last sequence_length bars
    features = features_df.tail(sequence_length).values
    current_bar = bars_df.iloc[-1]
    current_close = float(current_bar["close"])
    current_ts = pd.to_datetime(current_bar["timestamp"])

    # Normalize features
    features_normalized = normalizer.transform(features)

    # Model inference
    feature_tensor = (
        torch.from_numpy(features_normalized)
        .unsqueeze(0)
        .float()
        .contiguous()
        .to(device)
    )
    ref_close_tensor = torch.tensor([current_close], device=device)

    with torch.no_grad():
        outputs = model(feature_tensor)
        actions = model.get_hard_actions(outputs, ref_close_tensor)

    position_length = int(actions["position_length"].item())
    position_size = float(actions["position_size"].item())
    buy_price = float(actions["buy_price"].item())
    sell_price = float(actions["sell_price"].item())

    logger.info(f"Model output: length={position_length}h, size={position_size:.2%}, "
                f"buy=${buy_price:.2f}, sell=${sell_price:.2f}")

    if position_length == 0:
        logger.info("Model suggests skipping this opportunity")
        return None

    return TradingPlan(
        timestamp=current_ts,
        buy_price=buy_price,
        sell_price=sell_price,
        position_length=position_length,
        position_size=position_size,
    )


def execute_trade(
    symbol: str,
    plan: TradingPlan,
    dry_run: bool = False,
) -> bool:
    """Execute trading plan via Alpaca."""
    try:
        # Get account info
        account = alpaca_wrapper.get_account()
        cash = float(account.cash)

        # Calculate position value and quantity
        position_value = cash * plan.position_size
        min_notional = _get_min_order_notional(symbol)

        if position_value < min_notional:
            logger.warning(f"Position value ${position_value:.2f} below minimum ${min_notional}")
            return False

        # CRITICAL SAFETY: Validate buy_price < sell_price before any trading
        # This is the last line of defense against catastrophic losses
        if plan.buy_price >= plan.sell_price:
            logger.error(
                f"CRITICAL: Inverted prices detected! buy=${plan.buy_price:.4f} >= sell=${plan.sell_price:.4f}. "
                f"Trade BLOCKED to prevent guaranteed loss."
            )
            return False

        # Enforce minimum spread (avoid trading inside fee zone)
        min_spread_pct = 0.0016  # 16 bps (2x maker fee of 8bps)
        actual_spread_pct = (plan.sell_price - plan.buy_price) / plan.buy_price
        if actual_spread_pct < min_spread_pct:
            logger.warning(
                f"Spread too small: {actual_spread_pct*100:.3f}% < {min_spread_pct*100:.3f}% minimum. "
                f"Trade blocked to avoid fee losses."
            )
            return False

        # Get adjusted prices from price guard (tracks recent trades to avoid whipsawing)
        adj_buy, adj_sell = enforce_gap(symbol, plan.buy_price, plan.sell_price)

        # Re-validate after adjustment
        if adj_buy >= adj_sell:
            logger.error(f"Price adjustment resulted in invalid spread: buy=${adj_buy:.4f} >= sell=${adj_sell:.4f}")
            return False

        # Calculate quantity to buy using adjusted buy price
        target_qty = position_value / adj_buy

        if dry_run:
            logger.info(f"[DRY RUN] Would open position: {symbol} @ ${adj_buy:.2f}, "
                       f"qty={target_qty:.4f}, TP @ ${adj_sell:.2f}, hold {plan.position_length}h")
            return True

        # Spawn order watcher with ADJUSTED prices
        logger.info(f"Opening position: {symbol} @ ${adj_buy:.2f}, "
                   f"qty={target_qty:.4f}, TP @ ${adj_sell:.2f}, hold {plan.position_length}h")

        # Use the existing spawn function with correct parameters
        # CRITICAL: Use adjusted prices, not original plan prices
        # CRITICAL: entry_strategy MUST be set to kill old watchers when price changes
        spawn_open_position_at_maxdiff_takeprofit(
            symbol=symbol,
            side="buy",
            limit_price=adj_buy,
            target_qty=target_qty,
            expiry_minutes=plan.position_length * 60,  # Convert hours to minutes
            entry_strategy="hourlyv5",  # Required for watcher cleanup!
        )

        # CRITICAL: Also spawn SELL exit watcher for take-profit!
        # This was missing - positions were opened but never closed automatically
        spawn_close_position_at_maxdiff_takeprofit(
            symbol=symbol,
            side="buy",  # Entry side was BUY, exit will be SELL
            takeprofit_price=adj_sell,
            target_qty=target_qty,
            expiry_minutes=plan.position_length * 60,
            entry_strategy="hourlyv5",
        )

        record_buy(symbol, adj_buy)
        return True

    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False


def execute_close_trade(
    symbol: str,
    plan: TradingPlan,
    dry_run: bool = False,
) -> bool:
    """
    Execute CLOSE-ONLY trade - use model's sell prediction to exit existing position.

    This is for closing out legacy positions at optimal times without buying more.
    Uses the model's sell_price as a take-profit target for existing positions.
    """
    try:
        # Check if we have an existing position to close
        positions = alpaca_wrapper.get_all_positions()
        position = None
        for p in positions:
            if p.symbol == symbol or p.symbol == symbol.replace('USD', '/USD'):
                position = p
                break

        if position is None:
            logger.info(f"No existing position for {symbol} to close")
            return False

        position_qty = float(position.qty)
        avg_entry = float(position.avg_entry_price)
        current_price = float(position.current_price)
        unrealized_pnl = float(position.unrealized_pl)

        # Use model's sell_price as take-profit target
        target_sell_price = plan.sell_price

        # Note: We exit at model's sell price even if below entry.
        # Backtests show this is more profitable - frees capital for new opportunities.
        if target_sell_price <= avg_entry:
            logger.info(
                f"Exiting at loss: sell ${target_sell_price:.4f} < entry ${avg_entry:.4f}. "
                f"Realized loss: ${unrealized_pnl:+.2f} (freeing capital for better trades)"
            )

        # Calculate expected profit
        expected_profit = position_qty * (target_sell_price - avg_entry)
        expected_profit_pct = ((target_sell_price / avg_entry) - 1) * 100

        logger.info(
            f"Close-only signal: {symbol} qty={position_qty:.6f} @ entry ${avg_entry:.4f}, "
            f"target ${target_sell_price:.4f} ({expected_profit_pct:+.2f}% = ${expected_profit:+.2f})"
        )

        if dry_run:
            logger.info(f"[DRY RUN] Would close {symbol} position at ${target_sell_price:.4f}")
            return True

        # Spawn sell watcher to close position at target price
        logger.info(f"Spawning sell watcher for {symbol} @ ${target_sell_price:.4f}")

        spawn_close_position_at_maxdiff_takeprofit(
            symbol=symbol,
            side="buy",  # Entry side was BUY (long position), exit side will be SELL
            takeprofit_price=target_sell_price,
            target_qty=position_qty,
            expiry_minutes=plan.position_length * 60,  # Use model's predicted hold time
            entry_strategy="hourlyv5_close",  # Required for watcher cleanup!
        )

        record_sell(symbol, target_sell_price)
        return True

    except Exception as e:
        logger.error(f"Error executing close trade: {e}")
        return False


def run_once(
    model: HourlyCryptoPolicyV5,
    normalizer: FeatureNormalizer,
    feature_columns: list,
    symbol: str,
    dry_run: bool = False,
    device: str = "cuda",
    close_only: bool = False,
) -> None:
    """Run single trading iteration.

    Args:
        close_only: If True, only close existing positions (don't open new ones).
                   Use this for exiting legacy positions at optimal times.
    """
    mode = "close-only" if close_only else "normal"
    logger.info(f"Running trading iteration for {symbol} (mode={mode})")

    # Load fresh data
    bars_df, features_df = load_and_prepare_data(symbol, feature_columns)

    # Get trading plan
    plan = get_trading_plan(
        model=model,
        normalizer=normalizer,
        features_df=features_df,
        bars_df=bars_df,
        device=device,
    )

    if plan is None:
        logger.info("No trade signal this hour")
        return

    # Execute based on mode
    if close_only:
        execute_close_trade(symbol, plan, dry_run=dry_run)
    else:
        execute_trade(symbol, plan, dry_run=dry_run)


def run_daemon(
    model: HourlyCryptoPolicyV5,
    normalizer: FeatureNormalizer,
    feature_columns: list,
    symbol: str,
    dry_run: bool = False,
    device: str = "cuda",
    close_only: bool = False,
) -> None:
    """Run continuous trading loop aligned to UTC hours.

    Args:
        close_only: If True, only close existing positions (don't open new ones).
    """
    mode = "close-only" if close_only else "normal"
    logger.info(f"Starting daemon mode for {symbol} (mode={mode})")

    while True:
        try:
            # Calculate time to next hour
            now = datetime.now(timezone.utc)
            next_hour = (now + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
            sleep_seconds = (next_hour - now).total_seconds()

            logger.info(f"Sleeping {sleep_seconds:.0f}s until {next_hour}")
            time.sleep(max(0, sleep_seconds))

            # Run trading iteration
            run_once(model, normalizer, feature_columns, symbol, dry_run, device, close_only)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in daemon loop: {e}")
            time.sleep(60)  # Wait before retry


def main():
    parser = argparse.ArgumentParser(description="Neural Hourly Trading V5")
    parser.add_argument(
        "--symbol",
        type=str,
        default="LINKUSD",
        help="Symbol to trade (default: LINKUSD - best alpha)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: use DefaultStrategyConfig)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuously, trading every hour",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't execute trades, just log what would happen",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--close-only",
        action="store_true",
        help="Close-only mode: only sell existing positions, don't buy more. "
             "Use this to exit legacy positions (e.g., ETHUSD, BTCUSD) at optimal times.",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Get checkpoint path (use symbol-specific if available)
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        default_config = DefaultStrategyConfig()
        checkpoint_path = default_config.get_checkpoint_for_symbol(args.symbol)

    logger.info(f"Loading model from: {checkpoint_path}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Paper trading: {os.environ.get('PAPER', '1')}")
    if args.close_only:
        logger.info(f"Mode: CLOSE-ONLY (will only sell existing positions)")

    # Load model
    model, normalizer, feature_columns = load_model(checkpoint_path, args.device)

    if args.daemon:
        run_daemon(model, normalizer, feature_columns, args.symbol, args.dry_run, args.device, args.close_only)
    else:
        run_once(model, normalizer, feature_columns, args.symbol, args.dry_run, args.device, args.close_only)


if __name__ == "__main__":
    main()
