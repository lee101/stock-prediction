#!/usr/bin/env python3
"""Run simulation with per-symbol tuned Chronos2 forecaster."""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from marketsimlong.config import DataConfigLong, ForecastConfigLong, SimulationConfigLong
from marketsimlong.data import DailyDataLoader, is_crypto_symbol
from marketsimlong.forecaster_tuned import TunedChronos2Forecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_tuned_simulation(
    top_n: int = 10,
    initial_capital: float = 100_000.0,
    start_date: date = date(2025, 1, 2),
    end_date: date = date(2025, 12, 22),
    output_dir: Path = Path("reports/marketsimlong_tuned"),
) -> Dict:
    """Run simulation with tuned per-symbol forecaster.

    Args:
        top_n: Number of top symbols to buy each day
        initial_capital: Starting capital
        start_date: Simulation start date
        end_date: Simulation end date
        output_dir: Output directory for reports

    Returns:
        Simulation results dictionary
    """
    # Setup
    data_config = DataConfigLong(
        data_root=Path("trainingdata/train"),
        start_date=date(2025, 1, 1),
        end_date=end_date,
    )

    forecast_config = ForecastConfigLong(
        model_id="amazon/chronos-2",
        context_length=512,  # Will be overridden per-symbol
        prediction_length=1,
        use_multivariate=True,  # Will be overridden per-symbol
        device_map="cuda",
        batch_size=64,
        quantile_levels=(0.1, 0.5, 0.9),
    )

    # Load data
    data_loader = DailyDataLoader(data_config)
    data_loader.load_all_symbols()
    logger.info("Loaded %d symbols", len(data_loader._data_cache))

    # Create tuned forecaster
    forecaster = TunedChronos2Forecaster(
        data_loader,
        forecast_config,
        symbol_configs_dir=Path("hyperparams/chronos2_long/daily"),
    )

    # Simulation state
    equity = initial_capital
    equity_curve = []
    positions = {}  # symbol -> (shares, entry_price)
    trades = []
    symbol_pnl = {s: 0.0 for s in data_loader._data_cache.keys()}

    # Generate trading dates
    current_date = start_date
    trading_days = 0

    try:
        while current_date <= end_date:
            # Get tradable symbols for this date
            tradable_symbols = data_loader.get_tradable_symbols_on_date(current_date)

            if not tradable_symbols:
                current_date += timedelta(days=1)
                continue

            trading_days += 1

            # Close any existing positions
            for symbol, (shares, entry_price) in list(positions.items()):
                actual = data_loader.get_price_on_date(symbol, current_date)
                if actual:
                    exit_price = actual["open"]  # Exit at open
                    pnl = shares * (exit_price - entry_price)
                    equity += pnl
                    symbol_pnl[symbol] += pnl

                    trades.append({
                        "date": str(current_date),
                        "symbol": symbol,
                        "action": "SELL",
                        "shares": shares,
                        "price": exit_price,
                        "pnl": pnl,
                    })

            positions = {}

            # Get forecasts for all tradable symbols
            forecasts = forecaster.forecast_all_symbols(current_date, tradable_symbols)

            if not forecasts.forecasts:
                equity_curve.append({
                    "date": str(current_date),
                    "equity": equity,
                    "positions": 0,
                })
                current_date += timedelta(days=1)
                continue

            # Get top N symbols by predicted return
            top_symbols = forecasts.get_top_n_symbols(
                n=top_n,
                metric="predicted_return",
                min_return=0.0,  # Only buy if predicting positive return
            )

            if not top_symbols:
                # No positive predictions, stay in cash
                equity_curve.append({
                    "date": str(current_date),
                    "equity": equity,
                    "positions": 0,
                })
                current_date += timedelta(days=1)
                continue

            # Allocate capital equally among top symbols
            allocation_per_symbol = equity / len(top_symbols)

            for symbol in top_symbols:
                actual = data_loader.get_price_on_date(symbol, current_date)
                if not actual:
                    continue

                entry_price = actual["open"]  # Buy at open
                shares = allocation_per_symbol / entry_price

                positions[symbol] = (shares, entry_price)

                trades.append({
                    "date": str(current_date),
                    "symbol": symbol,
                    "action": "BUY",
                    "shares": shares,
                    "price": entry_price,
                    "predicted_return": forecasts.forecasts[symbol].predicted_return,
                })

            # Mark to market at close
            mtm_equity = 0.0
            for symbol, (shares, entry_price) in positions.items():
                actual = data_loader.get_price_on_date(symbol, current_date)
                if actual:
                    mtm_equity += shares * actual["close"]
                else:
                    mtm_equity += shares * entry_price

            equity_curve.append({
                "date": str(current_date),
                "equity": mtm_equity,
                "positions": len(positions),
            })

            # Log progress
            if trading_days % 20 == 0:
                returns = (mtm_equity - initial_capital) / initial_capital
                logger.info(
                    "Day %d (%s): Equity=$%.2f, Return=%.2f%%, Positions=%d",
                    trading_days,
                    current_date,
                    mtm_equity,
                    returns * 100,
                    len(positions),
                )

            current_date += timedelta(days=1)

        # Final close
        final_equity = 0.0
        for symbol, (shares, entry_price) in positions.items():
            # Find last available price
            for days_back in range(7):
                check_date = end_date - timedelta(days=days_back)
                actual = data_loader.get_price_on_date(symbol, check_date)
                if actual:
                    final_equity += shares * actual["close"]
                    break
            else:
                final_equity += shares * entry_price

        if final_equity == 0.0:
            final_equity = equity

    finally:
        forecaster.unload()

    # Compute metrics
    equity_df = pd.DataFrame(equity_curve)
    equity_df["returns"] = equity_df["equity"].pct_change().fillna(0)

    total_return = (equity_df["equity"].iloc[-1] - initial_capital) / initial_capital
    annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0

    # Sharpe ratio (assuming risk-free rate = 0)
    mean_return = equity_df["returns"].mean()
    std_return = equity_df["returns"].std()
    sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0

    # Sortino ratio
    downside_returns = equity_df["returns"][equity_df["returns"] < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino_ratio = (mean_return * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Max drawdown
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = (equity_df["peak"] - equity_df["equity"]) / equity_df["peak"]
    max_drawdown = equity_df["drawdown"].max()

    # Win rate
    buy_trades = [t for t in trades if t["action"] == "BUY"]
    sell_trades = [t for t in trades if t["action"] == "SELL"]
    winning_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]
    win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0

    # Symbol returns as percentage of initial capital
    symbol_returns_pct = {s: pnl / initial_capital for s, pnl in symbol_pnl.items() if abs(pnl) > 0.01}

    results = {
        "top_n": top_n,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_trades": len(trades),
        "trading_days": trading_days,
        "final_value": equity_df["equity"].iloc[-1],
        "symbol_returns": symbol_returns_pct,
    }

    # Save results
    output_dir = output_dir / f"top_{top_n}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    equity_df.to_csv(output_dir / "equity_curve.csv", index=False)

    pd.DataFrame(trades).to_csv(output_dir / "trades.csv", index=False)

    logger.info("=" * 60)
    logger.info("SIMULATION RESULTS (top %d, tuned)", top_n)
    logger.info("=" * 60)
    logger.info("Total Return: %.2f%%", total_return * 100)
    logger.info("Annualized Return: %.2f%%", annualized_return * 100)
    logger.info("Sharpe Ratio: %.2f", sharpe_ratio)
    logger.info("Sortino Ratio: %.2f", sortino_ratio)
    logger.info("Max Drawdown: %.2f%%", max_drawdown * 100)
    logger.info("Win Rate: %.2f%%", win_rate * 100)
    logger.info("Total Trades: %d", len(trades))
    logger.info("Final Value: $%.2f", equity_df["equity"].iloc[-1])
    logger.info("=" * 60)

    return results


def run_comparison():
    """Run simulations for top 1, 2, 3, and 10."""
    all_results = {}

    for top_n in [1, 2, 3, 10]:
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING TUNED SIMULATION: TOP %d", top_n)
        logger.info("=" * 70 + "\n")

        results = run_tuned_simulation(top_n=top_n)
        all_results[top_n] = results

    # Print comparison
    print("\n" + "=" * 80)
    print("TUNED SIMULATION COMPARISON")
    print("=" * 80)
    print(f"{'TopN':<8} {'Return%':<12} {'Annual%':<12} {'Sharpe':<10} {'MaxDD%':<10} {'WinRate%':<10}")
    print("-" * 80)

    for top_n, r in sorted(all_results.items()):
        print(
            f"{top_n:<8} "
            f"{r['total_return']*100:<12.2f} "
            f"{r['annualized_return']*100:<12.2f} "
            f"{r['sharpe_ratio']:<10.2f} "
            f"{r['max_drawdown']*100:<10.2f} "
            f"{r['win_rate']*100:<10.2f}"
        )

    print("=" * 80)

    # Save comparison
    output_dir = Path("reports/marketsimlong_tuned")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=None, help="Run single top-N simulation")
    parser.add_argument("--compare", action="store_true", help="Run comparison of top 1,2,3,10")
    args = parser.parse_args()

    if args.compare or args.top_n is None:
        run_comparison()
    else:
        run_tuned_simulation(top_n=args.top_n)
