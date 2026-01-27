"""Backtest Chronos2 direct forecaster with optimized thresholds."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .optimizer import (
    OptimizedParams,
    load_and_prepare_data,
    precompute_forecasts,
    simulate_strategy,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    equity_curve: np.ndarray,
    trades: list,
) -> dict:
    """Compute performance metrics."""
    if len(equity_curve) < 2:
        return {}

    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[np.isfinite(returns)]

    total_return = (equity_curve[-1] / equity_curve[0]) - 1

    # Sharpe (annualized, assuming 10-min bars)
    bars_per_year = 365.25 * 24 * 6
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = np.mean(returns) / np.std(downside) * np.sqrt(bars_per_year)
    else:
        sortino = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_dd = np.max(drawdown)

    # Win rate
    if trades:
        wins = sum(1 for t in trades if t.get("return", 0) > 0)
        win_rate = wins / len(trades)
    else:
        win_rate = 0.0

    return {
        "total_return_pct": total_return * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd * 100,
        "num_trades": len(trades),
        "win_rate_pct": win_rate * 100,
    }


def run_backtest(
    prices: np.ndarray,
    forecasts: list,
    params: OptimizedParams,
    cost_bps: float = 130.0,
) -> tuple[float, list, np.ndarray]:
    """Run backtest and return detailed results.

    Returns:
        (total_return, trades, equity_curve)
    """
    n = len(prices)
    cost_mult = 1 - (cost_bps / 10000)

    holding = False
    entry_price = 0.0
    entry_idx = 0
    equity = 1.0
    equity_curve = [1.0]
    trades = []

    for i in range(n):
        if forecasts[i] is None:
            equity_curve.append(equity)
            continue

        pred_return, upside_ratio = forecasts[i]
        current_price = prices[i]

        if not holding:
            # Check buy conditions
            if (pred_return >= params.buy_threshold and
                upside_ratio >= params.upside_ratio_min):
                holding = True
                entry_price = current_price
                entry_idx = i
                logger.debug(f"BUY @ {i}: price={current_price:.10f}, pred_ret={pred_return:.4f}")

        else:
            # Update equity based on position
            position_value = equity * (current_price / entry_price)

            # Check sell conditions
            bars_held = i - entry_idx
            current_return = (current_price - entry_price) / entry_price

            should_sell = False
            sell_reason = ""

            if pred_return <= params.sell_threshold:
                should_sell = True
                sell_reason = "forecast_negative"
            elif bars_held >= params.hold_bars and current_return > 0:
                should_sell = True
                sell_reason = "hold_complete_profit"

            if should_sell:
                trade_return = (current_price / entry_price) * cost_mult - 1
                equity *= (1 + trade_return)
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "return": trade_return,
                    "bars_held": bars_held,
                    "reason": sell_reason,
                })
                logger.debug(f"SELL @ {i}: price={current_price:.10f}, return={trade_return:.4f}")
                holding = False

        equity_curve.append(equity if not holding else equity * (current_price / entry_price))

    # Close any open position
    if holding:
        current_price = prices[-1]
        trade_return = (current_price / entry_price) * cost_mult - 1
        equity *= (1 + trade_return)
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": n - 1,
            "entry_price": entry_price,
            "exit_price": current_price,
            "return": trade_return,
            "bars_held": n - 1 - entry_idx,
            "reason": "end_of_data",
        })

    return equity - 1, trades, np.array(equity_curve)


def main():
    parser = argparse.ArgumentParser(description="Backtest Chronos2 direct forecaster")
    parser.add_argument("--ohlc", type=Path, default=Path("bagstraining/ohlc_data.csv"))
    parser.add_argument("--mint", type=str, required=True)
    parser.add_argument("--params", type=Path, default=Path("bagsdirect/optimized_params.json"))
    parser.add_argument("--test-split", type=float, default=0.3)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--full", action="store_true", help="Run on full data (not just test)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load params
    if args.params.exists():
        with open(args.params) as f:
            saved = json.load(f)
        params = OptimizedParams.from_dict(saved["params"])
        prediction_length = saved["config"]["prediction_length"]
        logger.info(f"Loaded params from {args.params}")
    else:
        # Default params if not optimized yet
        logger.warning("No optimized params found, using defaults")
        params = OptimizedParams(
            buy_threshold=0.02,
            sell_threshold=-0.02,
            upside_ratio_min=1.5,
            hold_bars=3,
            prediction_length=6,
        )
        prediction_length = 6

    print(f"\nUsing parameters:")
    print(f"  Buy threshold: {params.buy_threshold:.4f} ({params.buy_threshold*100:.2f}%)")
    print(f"  Sell threshold: {params.sell_threshold:.4f} ({params.sell_threshold*100:.2f}%)")
    print(f"  Upside ratio min: {params.upside_ratio_min:.2f}")
    print(f"  Hold bars: {params.hold_bars}")

    # Load data
    logger.info(f"Loading data from {args.ohlc}")
    df = load_and_prepare_data(args.ohlc, args.mint)
    logger.info(f"Loaded {len(df)} bars")

    # Split for test
    if args.full:
        test_df = df
        logger.info("Running on full data")
    else:
        train_size = int(len(df) * (1 - args.test_split))
        test_df = df.iloc[train_size:].reset_index(drop=True)
        logger.info(f"Testing on {len(test_df)} bars (last {args.test_split*100:.0f}%)")

    prices = test_df["close"].to_numpy(dtype=np.float64)

    # Compute buy & hold
    buy_hold_return = (prices[-1] / prices[0]) - 1
    print(f"\nBuy & Hold return: {buy_hold_return*100:.2f}%")

    # Precompute forecasts
    forecasts = precompute_forecasts(
        prices,
        context_length=args.context_length,
        prediction_length=prediction_length,
    )

    # Run backtest
    total_return, trades, equity_curve = run_backtest(prices, forecasts, params)
    metrics = compute_metrics(equity_curve, trades)

    print("\n=== Chronos2 Direct Backtest Results ===")
    print(f"Total return: {metrics['total_return_pct']:.2f}%")
    print(f"Sharpe: {metrics['sharpe']:.2f}")
    print(f"Sortino: {metrics['sortino']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Trades: {metrics['num_trades']}")
    print(f"Win rate: {metrics['win_rate_pct']:.1f}%")
    print(f"\nBuy & Hold: {buy_hold_return*100:.2f}%")
    print(f"Alpha: {(total_return - buy_hold_return)*100:.2f}%")

    # Print trade summary
    if trades:
        print(f"\nTrade Details:")
        for i, t in enumerate(trades[:10]):  # Show first 10
            print(f"  {i+1}. Entry:{t['entry_idx']} Exit:{t['exit_idx']} "
                  f"Return:{t['return']*100:+.2f}% Held:{t['bars_held']} bars "
                  f"({t['reason']})")
        if len(trades) > 10:
            print(f"  ... and {len(trades) - 10} more trades")


if __name__ == "__main__":
    main()
