"""Tune max_hold_days parameter to find optimal hold duration.

Tests different max_hold_days values (per-symbol and global) to maximize returns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import DataConfigLong, ForecastConfigLong, SimulationConfigLong
from .data import DailyDataLoader
from .forecaster import Chronos2Forecaster
from .simulator import LongTermDailySimulator, SimulationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MaxHoldTuningResult:
    """Result of tuning max_hold_days."""

    max_hold_days: int
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    total_trades: int


def run_simulation_with_max_hold(
    data_loader: DailyDataLoader,
    forecaster: Chronos2Forecaster,
    sim_config: SimulationConfigLong,
    start_date: date,
    end_date: date,
    max_hold_days: int,
    max_hold_days_per_symbol: Optional[Dict[str, int]] = None,
) -> SimulationResult:
    """Run simulation with specific max_hold_days setting.

    Args:
        data_loader: Loaded data loader
        forecaster: Chronos2 forecaster
        sim_config: Base simulation config
        start_date: Start date
        end_date: End date
        max_hold_days: Global max hold days
        max_hold_days_per_symbol: Per-symbol overrides

    Returns:
        SimulationResult
    """
    # Create config with max_hold_days
    config = SimulationConfigLong(
        top_n=sim_config.top_n,
        initial_cash=sim_config.initial_cash,
        maker_fee=sim_config.maker_fee,
        taker_fee=sim_config.taker_fee,
        slippage=sim_config.slippage,
        leverage=sim_config.leverage,
        margin_rate_annual=sim_config.margin_rate_annual,
        leverage_stocks_only=sim_config.leverage_stocks_only,
        equal_weight=sim_config.equal_weight,
        max_position_size=sim_config.max_position_size,
        strategy=sim_config.strategy,
        min_predicted_return=sim_config.min_predicted_return,
        max_daily_loss=sim_config.max_daily_loss,
        max_hold_days=max_hold_days,
        max_hold_days_per_symbol=max_hold_days_per_symbol,
    )

    simulator = LongTermDailySimulator(data_loader, forecaster, config)
    return simulator.run(start_date, end_date)


def tune_global_max_hold(
    data_loader: DailyDataLoader,
    forecaster: Chronos2Forecaster,
    sim_config: SimulationConfigLong,
    start_date: date,
    end_date: date,
    max_hold_values: List[int] = None,
) -> Tuple[int, List[MaxHoldTuningResult]]:
    """Find optimal global max_hold_days.

    Args:
        data_loader: Loaded data loader
        forecaster: Chronos2 forecaster
        sim_config: Base simulation config
        start_date: Start date
        end_date: End date
        max_hold_values: Values to test (default: [1, 2, 3, 5, 7, 10, 14, 0])

    Returns:
        (best_max_hold_days, all_results)
    """
    if max_hold_values is None:
        max_hold_values = [1, 2, 3, 5, 7, 10, 14, 21, 0]  # 0 = disabled (hold forever)

    results = []

    for max_hold in max_hold_values:
        logger.info("Testing max_hold_days=%d...", max_hold)

        sim_result = run_simulation_with_max_hold(
            data_loader,
            forecaster,
            sim_config,
            start_date,
            end_date,
            max_hold_days=max_hold,
        )

        result = MaxHoldTuningResult(
            max_hold_days=max_hold,
            total_return=sim_result.total_return,
            annualized_return=sim_result.annualized_return,
            sharpe_ratio=sim_result.sharpe_ratio,
            sortino_ratio=sim_result.sortino_ratio,
            max_drawdown=sim_result.max_drawdown,
            total_trades=sim_result.total_trades,
        )
        results.append(result)

        logger.info(
            "  max_hold=%d: Return=%.2f%%, Sharpe=%.3f, MaxDD=%.2f%%, Trades=%d",
            max_hold,
            sim_result.total_return * 100,
            sim_result.sharpe_ratio,
            sim_result.max_drawdown * 100,
            sim_result.total_trades,
        )

    # Find best by Sharpe ratio (balance of return and risk)
    best_result = max(results, key=lambda r: r.sharpe_ratio)
    logger.info(
        "Best max_hold_days=%d (Sharpe=%.3f, Return=%.2f%%)",
        best_result.max_hold_days,
        best_result.sharpe_ratio,
        best_result.total_return * 100,
    )

    return best_result.max_hold_days, results


def tune_per_symbol_max_hold(
    data_loader: DailyDataLoader,
    forecaster: Chronos2Forecaster,
    sim_config: SimulationConfigLong,
    start_date: date,
    end_date: date,
    symbols: List[str],
    max_hold_values: List[int] = None,
    base_max_hold: int = 5,
) -> Dict[str, int]:
    """Find optimal max_hold_days per symbol.

    This is slower as it tests each symbol independently.

    Args:
        data_loader: Loaded data loader
        forecaster: Chronos2 forecaster
        sim_config: Base simulation config (with top_n=1 for single symbol testing)
        start_date: Start date
        end_date: End date
        symbols: Symbols to tune
        max_hold_values: Values to test
        base_max_hold: Default for symbols not being tested

    Returns:
        Dict of symbol -> optimal max_hold_days
    """
    if max_hold_values is None:
        max_hold_values = [1, 2, 3, 5, 7, 10, 14, 0]

    optimal_per_symbol = {}

    for symbol in symbols:
        logger.info("Tuning max_hold_days for %s...", symbol)

        best_sharpe = float('-inf')
        best_max_hold = base_max_hold

        for max_hold in max_hold_values:
            # Set per-symbol override for just this symbol
            per_symbol = {symbol: max_hold}

            sim_result = run_simulation_with_max_hold(
                data_loader,
                forecaster,
                sim_config,
                start_date,
                end_date,
                max_hold_days=base_max_hold,
                max_hold_days_per_symbol=per_symbol,
            )

            if sim_result.sharpe_ratio > best_sharpe:
                best_sharpe = sim_result.sharpe_ratio
                best_max_hold = max_hold

        optimal_per_symbol[symbol] = best_max_hold
        logger.info("  %s: optimal max_hold=%d (Sharpe=%.3f)", symbol, best_max_hold, best_sharpe)

    return optimal_per_symbol


def run_tuning(
    top_n: int = 4,
    output_dir: str = "reports/max_hold_tuning",
) -> None:
    """Run full max_hold_days tuning.

    Args:
        top_n: Number of top symbols to trade
        output_dir: Output directory for results
    """
    # Setup configs
    data_config = DataConfigLong(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
    )
    forecast_config = ForecastConfigLong()
    sim_config = SimulationConfigLong(
        top_n=top_n,
        initial_cash=100_000.0,
    )

    # Load data and forecaster
    logger.info("Loading data...")
    data_loader = DailyDataLoader(data_config)
    data_loader.load_all_symbols()

    logger.info("Loading Chronos2 forecaster...")
    forecaster = Chronos2Forecaster(data_loader, forecast_config)

    try:
        # Phase 1: Find optimal global max_hold_days
        logger.info("=" * 60)
        logger.info("Phase 1: Tuning global max_hold_days")
        logger.info("=" * 60)

        best_global, global_results = tune_global_max_hold(
            data_loader,
            forecaster,
            sim_config,
            data_config.start_date,
            data_config.end_date,
        )

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        global_results_dict = {
            "top_n": top_n,
            "best_global_max_hold_days": best_global,
            "results": [
                {
                    "max_hold_days": r.max_hold_days,
                    "total_return": r.total_return,
                    "annualized_return": r.annualized_return,
                    "sharpe_ratio": r.sharpe_ratio,
                    "sortino_ratio": r.sortino_ratio,
                    "max_drawdown": r.max_drawdown,
                    "total_trades": r.total_trades,
                }
                for r in global_results
            ],
        }

        with open(output_path / "global_tuning.json", "w") as f:
            json.dump(global_results_dict, f, indent=2)

        logger.info("Results saved to %s", output_path / "global_tuning.json")

        # Print summary table
        print("\n" + "=" * 80)
        print("MAX HOLD DAYS TUNING RESULTS")
        print("=" * 80)
        print(f"{'Max Hold':<10} {'Return':<12} {'Ann Return':<12} {'Sharpe':<10} {'Sortino':<10} {'MaxDD':<10} {'Trades':<8}")
        print("-" * 80)

        for r in sorted(global_results, key=lambda x: x.sharpe_ratio, reverse=True):
            max_hold_str = str(r.max_hold_days) if r.max_hold_days > 0 else "Inf"
            print(
                f"{max_hold_str:<10} {r.total_return*100:>10.2f}% {r.annualized_return*100:>10.2f}% "
                f"{r.sharpe_ratio:>8.3f} {r.sortino_ratio:>9.3f} {r.max_drawdown*100:>8.2f}% {r.total_trades:>7d}"
            )

        print("-" * 80)
        print(f"BEST: max_hold_days={best_global} (by Sharpe ratio)")
        print("=" * 80)

    finally:
        forecaster.unload()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune max_hold_days parameter")
    parser.add_argument("--top-n", type=int, default=4, help="Number of top symbols to trade")
    parser.add_argument("--output-dir", type=str, default="reports/max_hold_tuning", help="Output directory")

    args = parser.parse_args()

    run_tuning(top_n=args.top_n, output_dir=args.output_dir)
