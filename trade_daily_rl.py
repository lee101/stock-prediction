#!/usr/bin/env python3
"""
Daily RL trading bot for crypto.
Runs once per day, generates signals from trade_pen_05 model,
optionally filtered by Gemini LLM.

Usage:
  # One-shot signal generation
  python trade_daily_rl.py --once

  # Daemon mode (runs daily at midnight UTC)
  python trade_daily_rl.py --daemon

  # Backtest mode
  python trade_daily_rl.py --backtest --backtest-days 90

  # With Gemini LLM overlay
  python trade_daily_rl.py --once --use-llm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from pufferlib_market.inference_daily import DailyPPOTrader, compute_daily_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("daily_rl")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = "pufferlib_market/checkpoints/autoresearch_daily/trade_pen_05/best.pt"

# Ensemble checkpoints (top OOS performers from mass sweep + original autoresearch)
ENSEMBLE_CHECKPOINTS = [
    "pufferlib_market/checkpoints/autoresearch_daily/trade_pen_05/best.pt",  # +20.0%, original
    "pufferlib_market/checkpoints/mass_daily/tp0.15_s314/best.pt",           # +22.4%, best mass
    "pufferlib_market/checkpoints/mass_daily/tp0.05_s123/best.pt",           # +14.9%
    "pufferlib_market/checkpoints/mass_daily/tp0.10_s314/best.pt",           # +8.2%
    "pufferlib_market/checkpoints/mass_daily/tp0.20_s7/best.pt",             # +7.7%
]
SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"]
FEE_BPS = 10.0  # 10bps per side for crypto
INITIAL_CAPITAL = 10_000.0
DATA_DIR = "trainingdatahourly/crypto"  # hourly data, aggregated to daily


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_daily_data(symbol: str, data_dir: str = DATA_DIR, min_days: int = 90) -> pd.DataFrame:
    """Load and aggregate hourly OHLCV to daily bars."""
    path = REPO / data_dir / f"{symbol}.csv"
    if not path.exists():
        # Try trainingdata/train/ (already daily)
        path = REPO / "trainingdata" / "train" / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol}")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # If hourly data, aggregate to daily
    if len(df) > 500 and (df["timestamp"].diff().median() < pd.Timedelta(hours=2)):
        df["date"] = df["timestamp"].dt.floor("D")
        daily = df.groupby("date").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).reset_index()
        # Only keep complete days (at least 20 bars)
        bar_counts = df.groupby("date").size()
        complete = bar_counts[bar_counts >= 20].index
        daily = daily[daily["date"].isin(complete)].copy()
        daily = daily.rename(columns={"date": "timestamp"})
        df = daily
    else:
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < min_days:
        raise ValueError(f"{symbol}: only {len(df)} days, need {min_days}")

    return df


def get_latest_prices(dfs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Get latest close price for each symbol."""
    return {sym: float(df["close"].iloc[-1]) for sym, df in dfs.items()}


# ---------------------------------------------------------------------------
# LLM overlay
# ---------------------------------------------------------------------------

def get_llm_filter(
    signal_action: str,
    signal_symbol: Optional[str],
    signal_confidence: float,
    daily_dfs: Dict[str, pd.DataFrame],
    prices: Dict[str, float],
) -> tuple[bool, str]:
    """
    Use Gemini to filter/confirm the RL signal.

    Returns:
        (should_trade, reasoning)
    """
    try:
        from llm_hourly_trader.providers import call_llm
    except ImportError:
        logger.warning("LLM module not available, skipping filter")
        return True, "LLM unavailable"

    # Build context for LLM
    context_parts = []
    context_parts.append(f"RL Model Signal: {signal_action} (confidence: {signal_confidence:.1%})")
    context_parts.append("")

    for sym in SYMBOLS:
        if sym in daily_dfs:
            df = daily_dfs[sym]
            close = df["close"].iloc[-1]
            ret_1d = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100 if len(df) >= 2 else 0
            ret_7d = (df["close"].iloc[-1] / df["close"].iloc[-8] - 1) * 100 if len(df) >= 8 else 0
            ret_30d = (df["close"].iloc[-1] / df["close"].iloc[-31] - 1) * 100 if len(df) >= 31 else 0
            high_20d = df["close"].tail(20).max()
            dd = (close / high_20d - 1) * 100
            context_parts.append(
                f"{sym}: ${close:,.2f} | 1d: {ret_1d:+.1f}% | 7d: {ret_7d:+.1f}% | "
                f"30d: {ret_30d:+.1f}% | DD from 20d high: {dd:+.1f}%"
            )

    prompt = (
        "You are a crypto trading risk manager. An RL trading model has generated "
        "a daily signal. Your job is to CONFIRM or REJECT the signal based on the "
        "current market context.\n\n"
        f"{''.join(context_parts)}\n\n"
        "Rules:\n"
        "- CONFIRM unless there's a clear reason not to (the RL model is well-calibrated)\n"
        "- REJECT if: extreme drawdown (>20% from recent high), obvious macro risk, "
        "or signal contradicts strong trend\n"
        "- Be concise. One sentence reasoning.\n\n"
        "Respond with JSON: {\"action\": \"CONFIRM\" or \"REJECT\", \"reason\": \"...\"}"
    )

    try:
        response = call_llm(prompt, model="gemini-2.5-flash")
        if isinstance(response, str):
            import re
            match = re.search(r'\{[^}]+\}', response)
            if match:
                result = json.loads(match.group())
                return result.get("action", "CONFIRM") == "CONFIRM", result.get("reason", "")
        return True, "Could not parse LLM response"
    except Exception as e:
        logger.warning(f"LLM filter error: {e}")
        return True, f"Error: {e}"


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def run_backtest(
    checkpoint: str,
    days: int = 90,
    use_llm: bool = False,
) -> Dict:
    """Run backtest on historical daily data."""
    logger.info(f"Loading data for backtest ({days} days)...")
    daily_dfs = {}
    for sym in SYMBOLS:
        try:
            daily_dfs[sym] = load_daily_data(sym, min_days=days + 90)
        except Exception as e:
            logger.warning(f"Skipping {sym}: {e}")

    if not daily_dfs:
        raise RuntimeError("No data loaded")

    trader = DailyPPOTrader(checkpoint, device="cpu", symbols=list(daily_dfs.keys()))

    # Find common date range
    min_len = min(len(df) for df in daily_dfs.values())
    test_start = min_len - days

    cash = INITIAL_CAPITAL
    position = None  # (symbol, qty, entry_price, direction)
    equity_curve = []
    trades = []
    fee_rate = FEE_BPS / 10000.0

    for t in range(test_start, min_len):
        # Get features from history up to t
        features = np.zeros((len(daily_dfs), 16), dtype=np.float32)
        prices = {}
        for i, (sym, df) in enumerate(daily_dfs.items()):
            sub = df.iloc[:t + 1]
            features[i] = compute_daily_features(sub)
            prices[sym] = float(df["close"].iloc[t])

        signal = trader.get_signal(features, prices)

        # Calculate current equity
        pos_value = 0.0
        if position is not None:
            sym, qty, entry, direction = position
            cur_price = prices.get(sym, entry)
            if direction == "long":
                pos_value = qty * cur_price
            else:
                pos_value = qty * (2 * entry - cur_price)  # short PnL

        equity = cash + pos_value
        equity_curve.append(equity)

        # Execute signal
        if signal.action == "flat" and position is not None:
            # Close position
            sym, qty, entry, direction = position
            cur_price = prices[sym]
            if direction == "long":
                proceeds = qty * cur_price * (1 - fee_rate)
            else:
                proceeds = qty * (2 * entry - cur_price) * (1 - fee_rate)
            cash += proceeds
            trades.append({
                "day": t - test_start,
                "action": "close",
                "symbol": sym,
                "price": cur_price,
                "pnl": proceeds - qty * entry,
            })
            position = None
            trader.update_state(0, 0, "")

        elif signal.symbol and signal.direction and position is None:
            # Open new position
            sym = signal.symbol
            price = prices[sym]
            invest = cash * 0.95  # keep 5% buffer
            qty = invest / (price * (1 + fee_rate))
            cash -= qty * price * (1 + fee_rate)
            position = (sym, qty, price, signal.direction)
            trades.append({
                "day": t - test_start,
                "action": f"open_{signal.direction}",
                "symbol": sym,
                "price": price,
            })
            # Find action index for state update
            sym_idx = list(daily_dfs.keys()).index(sym)
            action_idx = sym_idx + 1 if signal.direction == "long" else sym_idx + 1 + len(daily_dfs)
            trader.update_state(action_idx, price, sym)

        elif signal.symbol and signal.direction and position is not None:
            # Already in a position — check if switching
            if signal.symbol != position[0] or signal.direction != position[3]:
                # Close old + open new
                sym, qty, entry, direction = position
                cur_price = prices[sym]
                if direction == "long":
                    proceeds = qty * cur_price * (1 - fee_rate)
                else:
                    proceeds = qty * (2 * entry - cur_price) * (1 - fee_rate)
                cash += proceeds
                trades.append({
                    "day": t - test_start,
                    "action": "close",
                    "symbol": sym,
                    "price": cur_price,
                    "pnl": proceeds - qty * entry,
                })

                new_sym = signal.symbol
                new_price = prices[new_sym]
                invest = cash * 0.95
                new_qty = invest / (new_price * (1 + fee_rate))
                cash -= new_qty * new_price * (1 + fee_rate)
                position = (new_sym, new_qty, new_price, signal.direction)
                trades.append({
                    "day": t - test_start,
                    "action": f"open_{signal.direction}",
                    "symbol": new_sym,
                    "price": new_price,
                })

        trader.step_day()

    # Final equity
    if position is not None:
        sym, qty, entry, direction = position
        cur_price = prices[sym]
        if direction == "long":
            pos_value = qty * cur_price
        else:
            pos_value = qty * (2 * entry - cur_price)
        equity = cash + pos_value
    else:
        equity = cash
    equity_curve.append(equity)

    # Compute metrics
    eq = np.array(equity_curve)
    total_return = (eq[-1] / eq[0] - 1)
    daily_returns = np.diff(eq) / eq[:-1]
    neg_returns = daily_returns[daily_returns < 0]
    downside_dev = np.sqrt(np.mean(neg_returns ** 2)) if len(neg_returns) > 0 else 1e-8
    sortino = (np.mean(daily_returns) / downside_dev) * np.sqrt(365)
    max_dd = np.min(eq / np.maximum.accumulate(eq) - 1)
    annualized = (1 + total_return) ** (365.0 / days) - 1

    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0) < 0]
    total_trades = len([t for t in trades if "pnl" in t])
    win_rate = len(winning_trades) / max(1, total_trades)

    results = {
        "total_return": total_return,
        "annualized_return": annualized,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "days": days,
        "final_equity": equity,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTEST RESULTS ({days} days)")
    logger.info(f"{'='*60}")
    logger.info(f"Total return:     {total_return:+.2%}")
    logger.info(f"Annualized:       {annualized:+.2%}")
    logger.info(f"Sortino:          {sortino:.2f}")
    logger.info(f"Max drawdown:     {max_dd:.2%}")
    logger.info(f"Trades:           {total_trades}")
    logger.info(f"Win rate:         {win_rate:.1%}")
    logger.info(f"Final equity:     ${equity:,.2f}")
    logger.info(f"{'='*60}")

    return results


# ---------------------------------------------------------------------------
# Live / Paper trading
# ---------------------------------------------------------------------------

def run_ensemble(
    daily_dfs: Dict[str, pd.DataFrame],
    prices: Dict[str, float],
    checkpoints: List[str] = None,
) -> tuple:
    """Run ensemble of models and return majority-vote signal."""
    if checkpoints is None:
        checkpoints = [str(REPO / c) for c in ENSEMBLE_CHECKPOINTS if (REPO / c).exists()]

    if not checkpoints:
        raise RuntimeError("No ensemble checkpoints found")

    signals = []
    symbols_list = list(daily_dfs.keys())

    for ckpt in checkpoints:
        try:
            trader = DailyPPOTrader(ckpt, device="cpu", symbols=symbols_list)
            signal = trader.get_daily_signal(daily_dfs, prices)
            signals.append(signal)
        except Exception as e:
            logger.warning(f"Ensemble member failed ({ckpt}): {e}")

    if not signals:
        raise RuntimeError("All ensemble members failed")

    # Majority vote on action
    from collections import Counter
    actions = [s.action for s in signals]
    vote = Counter(actions).most_common(1)[0]
    winning_action = vote[0]
    vote_count = vote[1]
    total = len(signals)

    # Find the signal matching the winning action (use highest confidence)
    matching = [s for s in signals if s.action == winning_action]
    best = max(matching, key=lambda s: s.confidence)

    logger.info(f"Ensemble vote: {winning_action} ({vote_count}/{total} models agree)")
    for i, s in enumerate(signals):
        marker = " <--" if s.action == winning_action else ""
        logger.info(f"  Model {i}: {s.action} (conf={s.confidence:.1%}){marker}")

    return best, vote_count, total


def run_once(
    checkpoint: str,
    use_llm: bool = False,
    paper: bool = True,
    ensemble: bool = False,
):
    """Generate daily trading signal and optionally execute."""
    logger.info("Loading daily data...")
    daily_dfs = {}
    for sym in SYMBOLS:
        try:
            daily_dfs[sym] = load_daily_data(sym, min_days=90)
        except Exception as e:
            logger.warning(f"Skipping {sym}: {e}")

    if not daily_dfs:
        raise RuntimeError("No data loaded")

    prices = get_latest_prices(daily_dfs)

    if ensemble:
        logger.info("Running ensemble inference...")
        signal, vote_count, total = run_ensemble(daily_dfs, prices)
        agreement = f" (ensemble: {vote_count}/{total} agree)"
    else:
        logger.info(f"Loading model: {checkpoint}")
        trader = DailyPPOTrader(checkpoint, device="cpu", symbols=list(daily_dfs.keys()))
        signal = trader.get_daily_signal(daily_dfs, prices)
        agreement = ""

    logger.info(f"\n{'='*60}")
    logger.info(f"DAILY RL SIGNAL ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}){agreement}")
    logger.info(f"{'='*60}")
    logger.info(f"Action:     {signal.action}")
    logger.info(f"Symbol:     {signal.symbol or 'N/A'}")
    logger.info(f"Direction:  {signal.direction or 'N/A'}")
    logger.info(f"Confidence: {signal.confidence:.1%}")
    logger.info(f"Value est:  {signal.value_estimate:.4f}")

    for sym, price in sorted(prices.items()):
        logger.info(f"  {sym}: ${price:,.2f}")

    # LLM filter
    if use_llm and signal.symbol:
        should_trade, reason = get_llm_filter(
            signal.action, signal.symbol, signal.confidence,
            daily_dfs, prices,
        )
        logger.info(f"\nLLM filter: {'CONFIRM' if should_trade else 'REJECT'} — {reason}")
        if not should_trade:
            logger.info("Signal REJECTED by LLM. No action taken.")
            return

    # Execute (if not paper mode)
    if not paper and signal.symbol:
        try:
            import alpaca_wrapper
            logger.info(f"\nExecuting: {signal.action} at market...")
            # TODO: implement actual Alpaca execution
            logger.info("Execution not yet implemented — use paper mode for now")
        except ImportError:
            logger.warning("alpaca_wrapper not available")
    else:
        if signal.symbol:
            logger.info(f"\n[PAPER] Would execute: {signal.action}")

    logger.info(f"{'='*60}")


def run_daemon(
    checkpoint: str,
    use_llm: bool = False,
    paper: bool = True,
):
    """Run daily trading loop."""
    logger.info("Starting daily RL trading daemon...")

    while True:
        try:
            run_once(checkpoint, use_llm=use_llm, paper=paper)
        except Exception as e:
            logger.error(f"Error in daily cycle: {e}")
            traceback.print_exc()

        # Sleep until next midnight UTC
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(
            hour=0, minute=5, second=0, microsecond=0,
        )
        sleep_seconds = (tomorrow - now).total_seconds()
        logger.info(f"Sleeping {sleep_seconds/3600:.1f}h until {tomorrow.isoformat()}")
        time.sleep(sleep_seconds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Daily RL crypto trading bot")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--daemon", action="store_true", help="Run as daily daemon")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--backtest-days", type=int, default=90)
    parser.add_argument("--use-llm", action="store_true", help="Use Gemini LLM as filter")
    parser.add_argument("--paper", action="store_true", default=True, help="Paper mode (no real trades)")
    parser.add_argument("--live", action="store_true", help="Live trading mode")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble of top models (majority vote)")
    args = parser.parse_args()

    if args.live:
        args.paper = False

    checkpoint = str(REPO / args.checkpoint)

    if args.backtest:
        run_backtest(checkpoint, days=args.backtest_days, use_llm=args.use_llm)
    elif args.daemon:
        run_daemon(checkpoint, use_llm=args.use_llm, paper=args.paper)
    else:
        run_once(checkpoint, use_llm=args.use_llm, paper=args.paper, ensemble=args.ensemble)


if __name__ == "__main__":
    main()
