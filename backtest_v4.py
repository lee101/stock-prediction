#!/usr/bin/env python3
"""Backtest neuraldailyv4 model on historical data.

Runs the Chronos-2 inspired V4 model on historical data and simulates trades
to verify profitability before live deployment.

Usage:
    python backtest_v4.py --checkpoint neuraldailyv4/checkpoints/v4_equity_quiet/epoch_0193.pt
    python backtest_v4.py --checkpoint ... --days 90  # Backtest last 90 days
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

sys.path.insert(0, str(Path(__file__).parent))

from neuraldailyv4.config import DailyDatasetConfigV4
from neuraldailyv4.runtime import DailyTradingRuntimeV4, TradingPlanV4
# Use stockagent constants which match actual Alpaca fees
from stockagent.constants import TRADING_FEE, CRYPTO_TRADING_FEE
# TRADING_FEE = 0.0005 (5 bps per leg = 10 bps round-trip for equities)
# CRYPTO_TRADING_FEE = 0.0008 (8 bps per leg = 16 bps round-trip for crypto)


def is_crypto_symbol(symbol: str) -> bool:
    """Check if symbol is a crypto asset (ends in USD and length > 4)."""
    s = symbol.upper()
    return s.endswith("USD") and len(s) > 4


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest V4 model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--days", type=int, default=60, help="Number of days to backtest")
    parser.add_argument("--symbols", type=str, nargs="+", default=None, help="Symbols to test")
    parser.add_argument("--verbose", action="store_true", help="Print each trade")
    parser.add_argument("--stop-loss", type=float, default=None, help="Stop loss percentage")
    parser.add_argument("--trade-crypto", action="store_true", help="Enable crypto trading (16 bps round-trip fees)")
    return parser.parse_args()


def simulate_trade(
    plan: TradingPlanV4,
    future_data: pd.DataFrame,
    slippage: float = 0.001,
    stop_loss_pct: Optional[float] = None,
) -> Dict:
    """Simulate a single trade episode using historical data."""
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
    exit_days_int = max(1, min(20, exit_days_int))

    stop_loss_price = entry_price * (1.0 - stop_loss_pct) if stop_loss_pct else None

    tp_hit = False
    sl_hit = False
    exit_price = 0.0
    actual_hold_days = exit_days_int

    for day in range(1, min(exit_days_int + 1, len(future_data))):
        day_data = future_data.iloc[day]

        if stop_loss_price is not None and day_data["low"] <= stop_loss_price:
            sl_hit = True
            exit_price = stop_loss_price
            actual_hold_days = day
            break

        if day_data["high"] >= plan.sell_price:
            tp_hit = True
            exit_price = plan.sell_price
            actual_hold_days = day
            break

    if not tp_hit and not sl_hit:
        if exit_days_int < len(future_data):
            exit_price = future_data.iloc[exit_days_int]["close"] * (1.0 - slippage)
        else:
            exit_price = future_data.iloc[-1]["close"] * (1.0 - slippage)
            actual_hold_days = len(future_data) - 1

    # Use correct fee rate based on asset class
    is_crypto = is_crypto_symbol(plan.symbol)
    fee_rate = CRYPTO_TRADING_FEE if is_crypto else TRADING_FEE

    gross_return_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
    # Fees on both entry and exit legs
    fees_pct = fee_rate * 2
    net_return_pct = gross_return_pct - fees_pct

    # Also calculate dollar amounts for reference
    notional = entry_price * plan.trade_amount
    gross_pnl = (exit_price - entry_price) * plan.trade_amount
    fees = notional * fees_pct
    net_pnl = gross_pnl - fees

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
        "return_pct": net_return_pct,
        "gross_return_pct": gross_return_pct,
        "fees_pct": fees_pct,
        "is_crypto": is_crypto,
        "trade_amount": plan.trade_amount,
        "confidence": plan.confidence,
    }


def get_future_ohlc(symbol: str, start_date: pd.Timestamp, days: int = 25) -> Optional[pd.DataFrame]:
    """Get future OHLC data for simulation."""
    from neuraldailytraining.data import SymbolFrameBuilder
    from neuraldailytraining.config import DailyDatasetConfig

    config = DailyDatasetConfig(symbols=())
    builder = SymbolFrameBuilder(config, [])

    try:
        frame = builder.build(symbol)
        if frame.empty:
            return None

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
    runtime: DailyTradingRuntimeV4,
    symbols: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    verbose: bool = False,
    stop_loss_pct: Optional[float] = None,
) -> List[Dict]:
    """Run backtest over date range."""
    all_trades = []

    # Separate equity and crypto symbols
    equity_symbols = [s for s in symbols if not is_crypto_symbol(s)]
    crypto_symbols = [s for s in symbols if is_crypto_symbol(s)]

    current = start_date
    while current <= end_date:
        is_weekend = current.weekday() >= 5

        # Equities only trade on weekdays
        if equity_symbols and not is_weekend:
            try:
                plans = runtime.plan_batch(equity_symbols, as_of=current)
                for plan in plans:
                    future_data = get_future_ohlc(plan.symbol, current, days=25)
                    if future_data is None:
                        continue

                    result = simulate_trade(plan, future_data, stop_loss_pct=stop_loss_pct)
                    if result.get("filled"):
                        all_trades.append(result)
                        if verbose:
                            outcome = "TP" if result["tp_hit"] else ("SL" if result.get("sl_hit") else "FORCED")
                            logger.info(
                                f"{plan.symbol} {current.date()}: "
                                f"Entry ${result['entry_price']:.2f} -> Exit ${result['exit_price']:.2f} "
                                f"({outcome}, {result['actual_hold_days']}d) "
                                f"Return: {result['return_pct']:.2%}"
                            )
            except Exception as e:
                logger.warning(f"Failed to get equity plans for {current.date()}: {e}")

        # Crypto trades 24/7 (every day including weekends)
        if crypto_symbols:
            try:
                plans = runtime.plan_batch(crypto_symbols, as_of=current)
                for plan in plans:
                    future_data = get_future_ohlc(plan.symbol, current, days=25)
                    if future_data is None:
                        continue

                    result = simulate_trade(plan, future_data, stop_loss_pct=stop_loss_pct)
                    if result.get("filled"):
                        all_trades.append(result)
                        if verbose:
                            outcome = "TP" if result["tp_hit"] else ("SL" if result.get("sl_hit") else "FORCED")
                            logger.info(
                                f"{plan.symbol} {current.date()}: "
                                f"Entry ${result['entry_price']:.2f} -> Exit ${result['exit_price']:.2f} "
                                f"({outcome}, {result['actual_hold_days']}d) "
                                f"Return: {result['return_pct']:.2%}"
                            )
            except Exception as e:
                logger.warning(f"Failed to get crypto plans for {current.date()}: {e}")

        current += timedelta(days=1)

    return all_trades


def print_summary(trades: List[Dict]):
    """Print backtest summary statistics."""
    if not trades:
        logger.warning("No trades executed in backtest period")
        return

    df = pd.DataFrame(trades)

    logger.info("=" * 60)
    logger.info("V4 BACKTEST SUMMARY")
    logger.info("=" * 60)

    total_trades = len(df)
    tp_trades = df["tp_hit"].sum()
    sl_trades = df.get("sl_hit", pd.Series([False] * len(df))).sum()
    forced_trades = df["forced_exit"].sum()
    tp_rate = tp_trades / total_trades

    # Count crypto vs equity trades
    crypto_trades = df["is_crypto"].sum() if "is_crypto" in df.columns else 0
    equity_trades = total_trades - crypto_trades

    logger.info(f"Total trades: {total_trades} ({equity_trades} equity, {crypto_trades} crypto)")
    logger.info(f"Take profit hits: {tp_trades} ({tp_rate:.1%})")
    if sl_trades > 0:
        logger.info(f"Stop loss hits: {int(sl_trades)} ({sl_trades/total_trades:.1%})")
    logger.info(f"Forced exits: {forced_trades} ({forced_trades/total_trades:.1%})")

    # Report returns as percentages (primary metric)
    avg_return = df["return_pct"].mean()
    total_return = df["return_pct"].sum()
    std_return = df["return_pct"].std()
    sharpe = avg_return / std_return if std_return > 0 else 0
    max_return = df["return_pct"].max()
    min_return = df["return_pct"].min()
    win_rate = (df["return_pct"] > 0).mean()

    logger.info(f"\nTotal return: {total_return:.2%} (sum of {total_trades} trades)")
    logger.info(f"Average return per trade: {avg_return:.2%}")
    logger.info(f"Std return: {std_return:.2%}")
    logger.info(f"Sharpe (per trade): {sharpe:.3f}")
    logger.info(f"Win rate: {win_rate:.1%}")
    logger.info(f"Best trade: {max_return:.2%}")
    logger.info(f"Worst trade: {min_return:.2%}")

    # Show fees impact
    if "fees_pct" in df.columns:
        avg_fees = df["fees_pct"].mean()
        logger.info(f"Average fees per trade: {avg_fees:.3%}")

    avg_hold = df["actual_hold_days"].mean()
    logger.info(f"Average hold time: {avg_hold:.1f} days")

    logger.info("\n" + "=" * 60)
    logger.info("BY SYMBOL")
    logger.info("=" * 60)

    by_symbol = df.groupby("symbol").agg({
        "return_pct": ["count", "sum", "mean"],
        "tp_hit": "mean",
        "actual_hold_days": "mean",
    }).round(4)

    for symbol in sorted(by_symbol.index):
        stats = by_symbol.loc[symbol]
        count = int(stats[("return_pct", "count")])
        total_ret = stats[("return_pct", "sum")]
        avg_ret = stats[("return_pct", "mean")]
        tp_pct = stats[("tp_hit", "mean")]
        hold = stats[("actual_hold_days", "mean")]
        is_crypto = is_crypto_symbol(symbol)
        asset_tag = "₿" if is_crypto else "S"
        logger.info(
            f"{asset_tag} {symbol:8} | {count:3} trades | {total_ret:7.2%} total | "
            f"{avg_ret:6.2%} avg | {tp_pct:5.1%} TP | {hold:.1f}d hold"
        )

    logger.info("\n" + "=" * 60)
    logger.info("PROFITABILITY ASSESSMENT")
    logger.info("=" * 60)

    is_profitable = avg_return > 0
    good_sharpe = sharpe > 0.5
    good_tp_rate = tp_rate > 0.3
    good_win_rate = win_rate > 0.45

    checks = [
        ("Average return > 0", is_profitable),
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
        logger.info("\nModel PASSES all profitability checks - ready for deployment")
    else:
        logger.warning("\nModel FAILS some checks - review before deployment")


def main():
    args = parse_args()

    logger.info(f"Loading V4 checkpoint: {args.checkpoint}")
    runtime = DailyTradingRuntimeV4(Path(args.checkpoint), trade_crypto=args.trade_crypto)

    if args.symbols:
        symbols = args.symbols
    else:
        # Default symbols: equities + crypto
        equity_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
        crypto_symbols = ["BTCUSD", "ETHUSD"]
        symbols = equity_symbols + crypto_symbols

    # Count asset classes
    equities = [s for s in symbols if not is_crypto_symbol(s)]
    cryptos = [s for s in symbols if is_crypto_symbol(s)]
    logger.info(f"Testing on {len(symbols)} symbols: {len(equities)} equities, {len(cryptos)} crypto")
    logger.info(f"Equities: {equities}")
    if cryptos:
        logger.info(f"Crypto: {cryptos} (trades 24/7, higher fees: {CRYPTO_TRADING_FEE:.2%})")

    end_date = pd.Timestamp.now(tz="UTC")
    start_date = end_date - timedelta(days=args.days)

    logger.info(f"Backtest period: {start_date.date()} to {end_date.date()} ({args.days} days)")
    if args.stop_loss:
        logger.info(f"Stop loss: {args.stop_loss:.1%}")

    logger.info("Running backtest...")
    trades = run_backtest(runtime, symbols, start_date, end_date, verbose=args.verbose, stop_loss_pct=args.stop_loss)

    print_summary(trades)


if __name__ == "__main__":
    main()
