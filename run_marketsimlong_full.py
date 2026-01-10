#!/usr/bin/env python3
"""
Full marketsimlong pipeline: tune + simulate + compare.

This script:
1. Tunes Chronos2 hyperparameters for lowest MAE across all symbols
2. Runs simulation comparing top_n = 1, 2, 3 strategies
3. Reports comprehensive results
"""

import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure we can import marketsimlong
sys.path.insert(0, str(Path(__file__).parent))

from marketsimlong.config import (
    DataConfigLong,
    ForecastConfigLong,
    SimulationConfigLong,
    TuningConfigLong,
)
from marketsimlong.data import DailyDataLoader
from marketsimlong.forecaster import Chronos2Forecaster
from marketsimlong.simulator import LongTermDailySimulator, SimulationResult


def format_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def format_currency(v: float) -> str:
    return f"${v:,.2f}"


def run_quick_tuning(data_config: DataConfigLong) -> dict:
    """Run a quick tuning to find good hyperparameters."""
    logger.info("=" * 60)
    logger.info("PHASE 1: CHRONOS2 HYPERPARAMETER TUNING")
    logger.info("=" * 60)

    # Use a subset of validation dates for speed
    tuning_config = TuningConfigLong(
        metric="mae",
        context_lengths=(256, 512),
        prediction_lengths=(1,),  # We only need 1-day ahead
        use_multivariate_options=(True, False),
        preaug_strategies=("baseline", "log_diff"),
        val_days=30,  # Use last 30 days for validation
        output_dir=Path("reports/marketsimlong/tuning"),
        save_best=True,
    )

    data_loader = DailyDataLoader(data_config)
    data_loader.load_all_symbols()

    best_mae = float("inf")
    best_config = None
    results = []

    # Quick grid search
    for context_length in tuning_config.context_lengths:
        for use_multivariate in tuning_config.use_multivariate_options:
            for preaug in tuning_config.preaug_strategies:
                logger.info(
                    "Testing: ctx=%d, multivariate=%s, preaug=%s",
                    context_length,
                    use_multivariate,
                    preaug,
                )

                forecast_config = ForecastConfigLong(
                    context_length=context_length,
                    prediction_length=1,
                    use_multivariate=use_multivariate,
                    use_preaugmentation=preaug != "baseline",
                )

                try:
                    forecaster = Chronos2Forecaster(data_loader, forecast_config)

                    # Evaluate on last 30 trading days
                    val_end = data_config.end_date
                    val_dates = []
                    current = val_end
                    while len(val_dates) < tuning_config.val_days:
                        if current.weekday() < 5:  # Stock trading day
                            val_dates.append(current)
                        current = date(
                            current.year,
                            current.month,
                            current.day - 1 if current.day > 1 else 28,
                        )
                        if current.month < 1:
                            break
                        # Simpler date math
                        from datetime import timedelta
                        current = val_end - timedelta(days=len(val_dates) + (val_end - current).days)
                        if current < date(2025, 1, 1):
                            break

                    # Just use last N dates from available data
                    val_dates = []
                    for symbol in list(data_loader._data_cache.keys())[:1]:
                        df = data_loader._data_cache[symbol]
                        dates_2025 = df[df["date"] >= date(2025, 1, 1)]["date"].unique()
                        val_dates = sorted(dates_2025)[-tuning_config.val_days:]
                        break

                    errors = []
                    for val_date in val_dates[-10:]:  # Just test on 10 dates for speed
                        available = data_loader.get_tradable_symbols_on_date(val_date)
                        if not available:
                            continue

                        forecasts = forecaster.forecast_all_symbols(val_date, available[:5])  # 5 symbols for speed

                        for symbol, fc in forecasts.forecasts.items():
                            actual = data_loader.get_price_on_date(symbol, val_date)
                            if actual:
                                error = abs(fc.predicted_close - actual["close"]) / actual["close"]
                                errors.append(error)

                    if errors:
                        mae = np.mean(errors) * 100
                        logger.info("  MAE: %.2f%%", mae)

                        results.append({
                            "context_length": context_length,
                            "use_multivariate": use_multivariate,
                            "preaug": preaug,
                            "mae_pct": mae,
                        })

                        if mae < best_mae:
                            best_mae = mae
                            best_config = {
                                "context_length": context_length,
                                "use_multivariate": use_multivariate,
                                "preaug": preaug,
                            }
                            logger.info("  NEW BEST!")

                    forecaster.unload()

                except Exception as e:
                    logger.error("Config failed: %s", e)
                    continue

    logger.info("")
    logger.info("TUNING RESULTS:")
    logger.info("Best config: %s", best_config)
    logger.info("Best MAE: %.2f%%", best_mae)

    # Save tuning results
    output_dir = Path("reports/marketsimlong/tuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "tuning_results.json", "w") as f:
        json.dump({"best_config": best_config, "best_mae": best_mae, "all_results": results}, f, indent=2)

    return best_config or {"context_length": 512, "use_multivariate": True, "preaug": "baseline"}


def run_simulation_comparison(
    data_config: DataConfigLong,
    tuned_config: dict,
) -> dict:
    """Run simulation comparing top_n = 1, 2, 3."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 2: SIMULATION COMPARISON (top_n = 1, 2, 3)")
    logger.info("=" * 60)

    forecast_config = ForecastConfigLong(
        context_length=tuned_config.get("context_length", 512),
        prediction_length=1,
        use_multivariate=tuned_config.get("use_multivariate", True),
        use_preaugmentation=tuned_config.get("preaug", "baseline") != "baseline",
    )

    data_loader = DailyDataLoader(data_config)
    data_loader.load_all_symbols()

    forecaster = Chronos2Forecaster(data_loader, forecast_config)

    results = {}

    for top_n in [1, 2, 3]:
        logger.info("")
        logger.info("-" * 40)
        logger.info("Running simulation with top_n = %d", top_n)
        logger.info("-" * 40)

        sim_config = SimulationConfigLong(
            top_n=top_n,
            initial_cash=100_000.0,
            maker_fee=0.0008,
            slippage=0.0005,
        )

        simulator = LongTermDailySimulator(data_loader, forecaster, sim_config)

        try:
            result = simulator.run(
                start_date=data_config.start_date,
                end_date=data_config.end_date,
                progress_callback=lambda d, t, r: (
                    logger.info("Day %d/%d: %s, Value: $%.2f", d, t, r.date, r.ending_portfolio_value)
                    if d % 50 == 0 else None
                ),
            )
            results[top_n] = result

            logger.info("")
            logger.info("Results for top_n = %d:", top_n)
            logger.info("  Total Return: %s", format_pct(result.total_return))
            logger.info("  Annualized Return: %s", format_pct(result.annualized_return))
            logger.info("  Sharpe Ratio: %.2f", result.sharpe_ratio)
            logger.info("  Sortino Ratio: %.2f", result.sortino_ratio)
            logger.info("  Max Drawdown: %s", format_pct(result.max_drawdown))
            logger.info("  Win Rate: %s", format_pct(result.win_rate))
            logger.info("  Total Trades: %d", result.total_trades)

            # Save results
            output_dir = Path(f"reports/marketsimlong/top_{top_n}")
            output_dir.mkdir(parents=True, exist_ok=True)

            result.equity_curve.to_csv(output_dir / "equity_curve.csv", header=["portfolio_value"])

            summary = {
                "top_n": top_n,
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "final_value": result.final_portfolio_value,
                "symbol_returns": result.symbol_returns,
            }
            with open(output_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

        except Exception as e:
            logger.error("Simulation failed for top_n=%d: %s", top_n, e)
            import traceback
            traceback.print_exc()
            continue

    forecaster.unload()

    return results


def print_final_summary(results: dict, tuned_config: dict):
    """Print comprehensive final summary."""
    print()
    print("=" * 70)
    print("MARKETSIMLONG FINAL RESULTS")
    print("=" * 70)

    print()
    print("TUNED CONFIGURATION:")
    print(f"  Context Length: {tuned_config.get('context_length', 512)}")
    print(f"  Use Multivariate: {tuned_config.get('use_multivariate', True)}")
    print(f"  Pre-augmentation: {tuned_config.get('preaug', 'baseline')}")

    print()
    print("STRATEGY COMPARISON:")
    print("-" * 70)
    print(f"{'Strategy':<12} {'Total Ret':<12} {'Annual Ret':<12} {'Sharpe':<10} {'Sortino':<10} {'MaxDD':<10} {'Trades':<8}")
    print("-" * 70)

    for top_n in [1, 2, 3]:
        if top_n not in results:
            continue
        r = results[top_n]
        print(
            f"Top {top_n:<8} {format_pct(r.total_return):<12} {format_pct(r.annualized_return):<12} "
            f"{r.sharpe_ratio:<10.2f} {r.sortino_ratio:<10.2f} {format_pct(r.max_drawdown):<10} {r.total_trades:<8}"
        )

    print("-" * 70)

    # Find best strategy
    if results:
        best = max(results.items(), key=lambda x: x[1].sharpe_ratio)
        print()
        print(f"BEST STRATEGY: Top {best[0]} (Sharpe = {best[1].sharpe_ratio:.2f})")
        print(f"  Final Portfolio Value: {format_currency(best[1].final_portfolio_value)}")
        print(f"  Total Return: {format_pct(best[1].total_return)}")

        # Top performing symbols
        if best[1].symbol_returns:
            print()
            print("TOP PERFORMING SYMBOLS:")
            sorted_symbols = sorted(best[1].symbol_returns.items(), key=lambda x: x[1], reverse=True)[:5]
            for sym, ret in sorted_symbols:
                print(f"  {sym}: {format_pct(ret)}")

            print()
            print("WORST PERFORMING SYMBOLS:")
            sorted_symbols = sorted(best[1].symbol_returns.items(), key=lambda x: x[1])[:5]
            for sym, ret in sorted_symbols:
                print(f"  {sym}: {format_pct(ret)}")

    print()
    print("=" * 70)
    print("Results saved to reports/marketsimlong/")
    print("=" * 70)


def main():
    """Main entry point."""
    logger.info("Starting marketsimlong full pipeline...")
    logger.info("Simulation period: 2025-01-01 to 2025-12-22")

    # Configure for 2025
    data_config = DataConfigLong(
        data_root=Path("trainingdata/train"),
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 22),  # Latest available data
        context_days=512,
    )

    # Phase 1: Tuning
    tuned_config = run_quick_tuning(data_config)

    # Phase 2: Simulation comparison
    results = run_simulation_comparison(data_config, tuned_config)

    # Final summary
    print_final_summary(results, tuned_config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
