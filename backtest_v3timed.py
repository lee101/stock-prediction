#!/usr/bin/env python3
"""Backtest neuraldailyv3timed model on historical data.

Runs the model on historical data and simulates 10-day trade episodes
to verify profitability before live deployment.

Usage:
    python backtest_v3timed.py --checkpoint neuraldailyv3timed/checkpoints/run_name/epoch_0100.pt
    python backtest_v3timed.py --checkpoint ... --days 90  # Backtest last 90 days
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from neuraldailyv3timed.config import DailyDatasetConfigV3, SimulationConfig
from neuraldailyv3timed.runtime import DailyTradingRuntimeV3, TradingPlan


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest V3 Timed model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--days", type=int, default=60, help="Number of days to backtest")
    parser.add_argument("--symbols", type=str, nargs="+", default=None, help="Symbols to test (default: all)")
    parser.add_argument("--verbose", action="store_true", help="Print each trade")
    parser.add_argument("--stop-loss", type=float, default=None, help="Stop loss percentage (e.g., 0.05 for 5%)")
    return parser.parse_args()


def simulate_trade(
    plan: TradingPlan,
    future_data: pd.DataFrame,
    slippage: float = 0.001,
    stop_loss_pct: Optional[float] = None,
) -> Dict:
    """
    Simulate a single trade episode using historical data.

    Args:
        plan: TradingPlan from the model
        future_data: DataFrame with 'high', 'low', 'close' for days after entry
        slippage: Slippage on forced exits
        stop_loss_pct: Stop loss percentage from entry (e.g., 0.05 for -5%)

    Returns:
        Dict with trade results
    """
    if len(future_data) < 1:
        return {"filled": False, "reason": "no_future_data"}

    # Day 0: Check entry fill
    day0 = future_data.iloc[0]
    entry_filled = day0["low"] <= plan.buy_price

    if not entry_filled:
        return {
            "filled": False,
            "reason": "entry_not_filled",
            "day0_low": day0["low"],
            "buy_price": plan.buy_price,
        }

    entry_price = plan.buy_price
    exit_days_int = int(round(plan.exit_days))
    exit_days_int = max(1, min(20, exit_days_int))  # Support up to 20 day holds

    # Calculate stop-loss price if specified
    stop_loss_price = entry_price * (1.0 - stop_loss_pct) if stop_loss_pct else None

    # Days 1 to exit_days: Check stop loss and take profit
    tp_hit = False
    sl_hit = False
    exit_price = 0.0
    actual_hold_days = exit_days_int

    for day in range(1, min(exit_days_int + 1, len(future_data))):
        day_data = future_data.iloc[day]

        # Check stop loss first (happens earlier in day)
        if stop_loss_price is not None and day_data["low"] <= stop_loss_price:
            sl_hit = True
            exit_price = stop_loss_price
            actual_hold_days = day
            break

        # Check take profit
        if day_data["high"] >= plan.sell_price:
            tp_hit = True
            exit_price = plan.sell_price
            actual_hold_days = day
            break

    # Forced exit at deadline (only if no TP or SL hit)
    if not tp_hit and not sl_hit:
        if exit_days_int < len(future_data):
            exit_price = future_data.iloc[exit_days_int]["close"] * (1.0 - slippage)
        else:
            # Not enough data, use last available close
            exit_price = future_data.iloc[-1]["close"] * (1.0 - slippage)
            actual_hold_days = len(future_data) - 1

    # Calculate P&L
    gross_pnl = (exit_price - entry_price) * plan.trade_amount
    fees = (entry_price + exit_price) * plan.trade_amount * 0.0008  # 8bps each way
    net_pnl = gross_pnl - fees
    return_pct = net_pnl / (entry_price * plan.trade_amount) if entry_price > 0 else 0

    return {
        "filled": True,
        "symbol": plan.symbol,
        "entry_date": plan.timestamp,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "target_exit_days": plan.exit_days,
        "actual_hold_days": actual_hold_days,
        "tp_hit": tp_hit,
        "sl_hit": sl_hit,
        "forced_exit": not tp_hit and not sl_hit,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "net_pnl": net_pnl,
        "return_pct": return_pct,
        "trade_amount": plan.trade_amount,
        "confidence": plan.confidence,
    }


def get_future_ohlc(symbol: str, start_date: pd.Timestamp, days: int = 21) -> Optional[pd.DataFrame]:
    """Get future OHLC data for simulation."""
    from neuraldailytraining.data import SymbolFrameBuilder
    from neuraldailytraining.config import DailyDatasetConfig

    config = DailyDatasetConfig(symbols=())
    builder = SymbolFrameBuilder(config, [])

    try:
        frame = builder.build(symbol)
        if frame.empty:
            return None

        # Find start index
        frame = frame.sort_values("date").reset_index(drop=True)
        mask = frame["date"] > start_date
        future = frame[mask].head(days)

        if len(future) < 1:
            return None

        return future[["date", "high", "low", "close"]].reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Failed to get future data for {symbol}: {e}")
        return None


def run_backtest(
    runtime: DailyTradingRuntimeV3,
    symbols: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    verbose: bool = False,
    stop_loss_pct: Optional[float] = None,
) -> List[Dict]:
    """Run backtest over date range."""
    all_trades = []

    # Generate dates to test (trading days approximation)
    current = start_date
    while current <= end_date:
        # Skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        # Get plans for this date
        try:
            plans = runtime.plan_batch(symbols, as_of=current)
        except Exception as e:
            logger.warning(f"Failed to get plans for {current.date()}: {e}")
            current += timedelta(days=1)
            continue

        for plan in plans:
            # Get future data for simulation
            future_data = get_future_ohlc(plan.symbol, current, days=22)
            if future_data is None:
                continue

            # Simulate the trade
            result = simulate_trade(plan, future_data, stop_loss_pct=stop_loss_pct)

            if result.get("filled"):
                all_trades.append(result)
                if verbose:
                    if result["tp_hit"]:
                        outcome = "TP"
                    elif result.get("sl_hit"):
                        outcome = "SL"
                    else:
                        outcome = "FORCED"
                    logger.info(
                        f"{plan.symbol} {current.date()}: "
                        f"Entry ${result['entry_price']:.2f} -> Exit ${result['exit_price']:.2f} "
                        f"({outcome}, {result['actual_hold_days']}d) "
                        f"Return: {result['return_pct']:.2%}"
                    )

        current += timedelta(days=1)

    return all_trades


def print_summary(trades: List[Dict]):
    """Print backtest summary statistics."""
    if not trades:
        logger.warning("No trades executed in backtest period")
        return

    df = pd.DataFrame(trades)

    logger.info("=" * 60)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 60)

    # Overall stats
    total_trades = len(df)
    tp_trades = df["tp_hit"].sum()
    sl_trades = df.get("sl_hit", pd.Series([False] * len(df))).sum()
    forced_trades = df["forced_exit"].sum()
    tp_rate = tp_trades / total_trades
    sl_rate = sl_trades / total_trades if sl_trades > 0 else 0

    logger.info(f"Total trades: {total_trades}")
    logger.info(f"Take profit hits: {tp_trades} ({tp_rate:.1%})")
    if sl_trades > 0:
        logger.info(f"Stop loss hits: {int(sl_trades)} ({sl_rate:.1%})")
    logger.info(f"Forced exits: {forced_trades} ({forced_trades/total_trades:.1%})")

    # P&L stats
    total_pnl = df["net_pnl"].sum()
    avg_pnl = df["net_pnl"].mean()
    win_rate = (df["net_pnl"] > 0).mean()

    logger.info(f"\nTotal Net P&L: ${total_pnl:,.2f}")
    logger.info(f"Average P&L per trade: ${avg_pnl:,.2f}")
    logger.info(f"Win rate: {win_rate:.1%}")

    # Return stats
    avg_return = df["return_pct"].mean()
    std_return = df["return_pct"].std()
    sharpe = avg_return / std_return if std_return > 0 else 0
    max_return = df["return_pct"].max()
    min_return = df["return_pct"].min()

    logger.info(f"\nAverage return: {avg_return:.2%}")
    logger.info(f"Std return: {std_return:.2%}")
    logger.info(f"Sharpe (per trade): {sharpe:.3f}")
    logger.info(f"Best trade: {max_return:.2%}")
    logger.info(f"Worst trade: {min_return:.2%}")

    # Hold time stats
    avg_hold = df["actual_hold_days"].mean()
    logger.info(f"\nAverage hold time: {avg_hold:.1f} days")

    # By symbol
    logger.info("\n" + "=" * 60)
    logger.info("BY SYMBOL")
    logger.info("=" * 60)

    by_symbol = df.groupby("symbol").agg({
        "net_pnl": ["count", "sum", "mean"],
        "return_pct": "mean",
        "tp_hit": "mean",
        "actual_hold_days": "mean",
    }).round(4)

    for symbol in by_symbol.index:
        stats = by_symbol.loc[symbol]
        count = int(stats[("net_pnl", "count")])
        total = stats[("net_pnl", "sum")]
        avg_ret = stats[("return_pct", "mean")]
        tp_pct = stats[("tp_hit", "mean")]
        hold = stats[("actual_hold_days", "mean")]
        logger.info(
            f"{symbol:8} | {count:3} trades | ${total:8.2f} total | "
            f"{avg_ret:6.2%} avg ret | {tp_pct:5.1%} TP | {hold:.1f}d hold"
        )

    # Profitability assessment
    logger.info("\n" + "=" * 60)
    logger.info("PROFITABILITY ASSESSMENT")
    logger.info("=" * 60)

    is_profitable = total_pnl > 0
    good_sharpe = sharpe > 0.5
    good_tp_rate = tp_rate > 0.3
    good_win_rate = win_rate > 0.45

    checks = [
        ("Total P&L > 0", is_profitable),
        ("Sharpe > 0.5", good_sharpe),
        ("TP Rate > 30%", good_tp_rate),
        ("Win Rate > 45%", good_win_rate),
    ]

    all_pass = True
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {check_name}")
        all_pass = all_pass and passed

    if all_pass:
        logger.info("\n✓ Model PASSES all profitability checks - ready for deployment")
    else:
        logger.warning("\n✗ Model FAILS some checks - review before deployment")


def main():
    args = parse_args()

    # Load runtime
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    runtime = DailyTradingRuntimeV3(Path(args.checkpoint))

    # Determine symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = list(runtime.dataset_config.symbols)

    logger.info(f"Testing on {len(symbols)} symbols: {symbols}")

    # Date range
    end_date = pd.Timestamp.now(tz="UTC")
    start_date = end_date - timedelta(days=args.days)

    logger.info(f"Backtest period: {start_date.date()} to {end_date.date()} ({args.days} days)")
    if args.stop_loss:
        logger.info(f"Stop loss: {args.stop_loss:.1%}")

    # Run backtest
    logger.info("Running backtest...")
    trades = run_backtest(runtime, symbols, start_date, end_date, verbose=args.verbose, stop_loss_pct=args.stop_loss)

    # Print summary
    print_summary(trades)


if __name__ == "__main__":
    main()
