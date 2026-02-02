#!/usr/bin/env python3
"""
Long-term daily market simulation runner.

This script runs the marketsimlong simulation which:
1. Optionally tunes Chronos2 hyperparameters for lowest MAE
2. Runs a daily trading simulation over the entire year
3. Each day, buys the top N symbols with highest predicted % growth
4. Reports performance metrics

Usage:
    # Run simulation with default settings
    python -m marketsimlong.run_simulation

    # Run with tuning first
    python -m marketsimlong.run_simulation --tune

    # Simulate top 3 symbols per day
    python -m marketsimlong.run_simulation --top-n 3

    # Specify date range
    python -m marketsimlong.run_simulation --start 2025-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List

import pandas as pd

from .config import DataConfigLong, ForecastConfigLong, SimulationConfigLong, TuningConfigLong
from .simulator import run_simulation, SimulationResult
from .tuner import tune_chronos2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> date:
    """Parse date string to date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _clean_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if not cleaned:
        return ""
    if cleaned in {"CORRELATION_MATRIX", "DATA_SUMMARY", "VOLATILITY_METRICS"}:
        return ""
    if not cleaned.replace("-", "").replace("_", "").isalnum():
        return ""
    return cleaned


def _load_symbols_from_dir(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbol directory not found: {path}")
    symbols = sorted({_clean_symbol(p.stem) for p in path.glob("*.csv")})
    return [s for s in symbols if s]


def _load_symbols_from_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbols file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            candidates = data.get("available_symbols") or data.get("symbols") or []
        else:
            candidates = data
        cleaned = [_clean_symbol(str(s)) for s in candidates if str(s).strip()]
        return [s for s in cleaned if s]

    symbols: List[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cleaned = _clean_symbol(stripped.strip("',\" "))
        if cleaned:
            symbols.append(cleaned)
    return symbols


def format_pct(value: float) -> str:
    """Format as percentage."""
    return f"{value * 100:.2f}%"


def format_currency(value: float) -> str:
    """Format as currency."""
    return f"${value:,.2f}"


def print_simulation_summary(result: SimulationResult) -> None:
    """Print simulation results summary."""
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    print(f"\nPeriod: {result.start_date} to {result.end_date}")
    print(f"Total Days: {result.total_days}")
    print(f"Total Trades: {result.total_trades}")

    print("\n--- Portfolio Performance ---")
    print(f"Initial Capital: {format_currency(result.initial_cash)}")
    print(f"Final Value: {format_currency(result.final_portfolio_value)}")
    print(f"Total Return: {format_pct(result.total_return)}")
    print(f"Annualized Return: {format_pct(result.annualized_return)}")

    print("\n--- Risk Metrics ---")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"Max Drawdown: {format_pct(result.max_drawdown)}")
    print(f"Win Rate: {format_pct(result.win_rate)}")

    print("\n--- Daily Statistics ---")
    print(f"Avg Daily Return: {format_pct(result.avg_daily_return)}")
    if result.total_margin_interest_paid or result.total_risk_penalty:
        print("\n--- Costs ---")
        print(f"Margin Interest Paid: {format_currency(result.total_margin_interest_paid)}")
        print(f"Risk Penalties: {format_currency(result.total_risk_penalty)}")

    if result.symbol_returns:
        print("\n--- Top Performing Symbols ---")
        sorted_symbols = sorted(
            result.symbol_returns.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        for symbol, ret in sorted_symbols:
            print(f"  {symbol}: {format_pct(ret)}")

    print("\n" + "=" * 60)


def save_results(
    result: SimulationResult,
    output_dir: Path,
    config_used: dict,
) -> None:
    """Save simulation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save equity curve
    equity_path = output_dir / "equity_curve.csv"
    result.equity_curve.to_csv(equity_path, header=["portfolio_value"])
    logger.info("Saved equity curve to %s", equity_path)

    # Save summary metrics
    summary = {
        "start_date": str(result.start_date),
        "end_date": str(result.end_date),
        "initial_cash": result.initial_cash,
        "final_portfolio_value": result.final_portfolio_value,
        "total_return": result.total_return,
        "annualized_return": result.annualized_return,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "total_days": result.total_days,
        "margin_interest_paid": result.total_margin_interest_paid,
        "risk_penalties": result.total_risk_penalty,
        "config": config_used,
        "symbol_returns": result.symbol_returns,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved summary to %s", summary_path)

    # Save trades
    if result.all_trades:
        trades_data = [
            {
                "date": str(t.timestamp),
                "symbol": t.symbol,
                "side": t.side,
                "quantity": t.quantity,
                "price": t.price,
                "notional": t.notional,
                "fee": t.fee,
            }
            for t in result.all_trades
        ]
        trades_df = pd.DataFrame(trades_data)
        trades_path = output_dir / "trades.csv"
        trades_df.to_csv(trades_path, index=False)
        logger.info("Saved %d trades to %s", len(trades_data), trades_path)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Long-term daily market simulation with Chronos2 forecasting",
    )

    # Simulation parameters
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Number of top symbols to buy each day (default: 1)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="Initial capital (default: $100,000)",
    )

    # Date range
    parser.add_argument(
        "--start",
        type=str,
        default="2025-01-01",
        help="Simulation start date (default: 2025-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-31",
        help="Simulation end date (default: 2025-12-31)",
    )
    parser.add_argument(
        "--symbols-dir",
        type=str,
        default="",
        help="Directory of CSVs to build symbol universe (overrides defaults)",
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default="",
        help="Optional symbols file (txt/json) to build symbol universe",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="Limit number of symbols (0 = no limit)",
    )

    # Tuning
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning before simulation",
    )
    parser.add_argument(
        "--tune-method",
        type=str,
        choices=["grid", "optuna"],
        default="grid",
        help="Tuning method (default: grid)",
    )

    # Forecasting controls
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device map for Chronos2 (default: cuda)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Chronos2 context length (default: 512)",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=1,
        help="Chronos2 prediction length (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Chronos2 batch size (default: 128)",
    )
    parser.add_argument(
        "--cross-learning",
        action="store_true",
        help="Enable Chronos2 cross-learning (predict_batches_jointly)",
    )
    parser.add_argument(
        "--cross-learning-min-batch",
        type=int,
        default=2,
        help="Minimum batch size to enable cross-learning",
    )
    parser.add_argument(
        "--cross-learning-chunk",
        type=int,
        default=0,
        help="Chunk size for cross-learning batches (0 = no chunking)",
    )
    parser.add_argument(
        "--cross-learning-no-group",
        action="store_true",
        help="Disable asset-type grouping for cross-learning",
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory (default: data)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/marketsimlong",
        help="Output directory for results (default: reports/marketsimlong)",
    )

    # Leverage
    parser.add_argument(
        "--leverage",
        type=float,
        default=1.0,
        help="Leverage multiplier (1.0=no leverage, 2.0=2x leverage). Stocks only. (default: 1.0)",
    )
    parser.add_argument(
        "--margin-rate",
        type=float,
        default=0.0625,
        help="Annual margin interest rate (default: 0.0625 = 6.25%%)",
    )
    parser.add_argument(
        "--leverage-soft-cap",
        type=float,
        default=0.0,
        help="Soft cap for leverage (0 disables penalties).",
    )
    parser.add_argument(
        "--leverage-penalty-rate",
        type=float,
        default=0.0,
        help="Daily penalty rate per unit leverage above soft cap.",
    )
    parser.add_argument(
        "--hold-penalty-start-days",
        type=int,
        default=0,
        help="Start applying hold penalty after this many days (0 disables).",
    )
    parser.add_argument(
        "--hold-penalty-rate",
        type=float,
        default=0.0,
        help="Daily penalty rate applied to notional for held positions beyond threshold.",
    )

    # Experiment variations
    parser.add_argument(
        "--compare-top-n",
        action="store_true",
        help="Run comparison across top_n=1,2,3",
    )

    args = parser.parse_args()

    # Parse dates
    start_date = parse_date(args.start)
    end_date = parse_date(args.end)

    symbols_override: List[str] = []
    if args.symbols_file:
        symbols_override = _load_symbols_from_file(Path(args.symbols_file))
    elif args.symbols_dir:
        symbols_override = _load_symbols_from_dir(Path(args.symbols_dir))

    if args.max_symbols and args.max_symbols > 0:
        symbols_override = symbols_override[: args.max_symbols]

    if symbols_override:
        stock_symbols = tuple(
            sym for sym in symbols_override if not sym.endswith(("USD", "USDT", "USDC", "BTC", "ETH"))
        )
        crypto_symbols = tuple(
            sym for sym in symbols_override if sym.endswith(("USD", "USDT", "USDC", "BTC", "ETH"))
        )
    else:
        stock_symbols = DataConfigLong().stock_symbols
        crypto_symbols = DataConfigLong().crypto_symbols

    # Build configs
    data_config = DataConfigLong(
        stock_symbols=stock_symbols,
        crypto_symbols=crypto_symbols,
        data_root=Path(args.data_dir),
        start_date=start_date,
        end_date=end_date,
    )

    forecast_config = ForecastConfigLong(
        device_map=args.device,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        use_multivariate=True,
        use_cross_learning=bool(args.cross_learning),
        cross_learning_min_batch=max(2, int(args.cross_learning_min_batch)),
        cross_learning_group_by_asset_type=not args.cross_learning_no_group,
        cross_learning_chunk_size=int(args.cross_learning_chunk) if args.cross_learning_chunk > 0 else None,
    )

    # Run tuning if requested
    if args.tune:
        logger.info("Running Chronos2 hyperparameter tuning...")
        tuning_config = TuningConfigLong()
        best_config, best_metrics = tune_chronos2(
            data_config,
            tuning_config,
            method=args.tune_method,
        )
        logger.info("Best config: %s", best_config)
        logger.info("Best metrics: %s", best_metrics)

        # Update forecast config with tuned parameters
        if best_config:
            forecast_config = ForecastConfigLong(
                device_map=args.device,
                context_length=best_config.get("context_length", args.context_length),
                prediction_length=best_config.get("prediction_length", args.prediction_length),
                use_multivariate=best_config.get("use_multivariate", True),
                use_preaugmentation=best_config.get("preaug_strategy", "baseline") != "baseline",
                batch_size=args.batch_size,
                use_cross_learning=bool(args.cross_learning),
                cross_learning_min_batch=max(2, int(args.cross_learning_min_batch)),
                cross_learning_group_by_asset_type=not args.cross_learning_no_group,
                cross_learning_chunk_size=int(args.cross_learning_chunk) if args.cross_learning_chunk > 0 else None,
            )

    # Compare top_n values if requested
    if args.compare_top_n:
        logger.info("Comparing top_n values: 1, 2, 3")
        results = {}

        for top_n in [1, 2, 3]:
            logger.info("\n=== Running simulation with top_n=%d ===", top_n)
            sim_config = SimulationConfigLong(
                top_n=top_n,
                initial_cash=args.initial_cash,
                leverage=args.leverage,
                margin_rate_annual=args.margin_rate,
                leverage_soft_cap=args.leverage_soft_cap,
                leverage_penalty_rate=args.leverage_penalty_rate,
                hold_penalty_start_days=args.hold_penalty_start_days,
                hold_penalty_rate=args.hold_penalty_rate,
            )

            result = run_simulation(data_config, forecast_config, sim_config)
            results[top_n] = result

            print(f"\n--- Results for top_n={top_n} ---")
            print(f"Total Return: {format_pct(result.total_return)}")
            print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {format_pct(result.max_drawdown)}")

            # Save individual results
            output_dir = Path(args.output_dir) / f"top_{top_n}"
            save_results(result, output_dir, {"top_n": top_n})

        # Print comparison summary
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Top N':<10} {'Total Return':<15} {'Sharpe':<10} {'Max DD':<10}")
        print("-" * 45)
        for top_n, result in results.items():
            print(
                f"{top_n:<10} {format_pct(result.total_return):<15} "
                f"{result.sharpe_ratio:<10.2f} {format_pct(result.max_drawdown):<10}"
            )

        return 0

    # Run single simulation
    sim_config = SimulationConfigLong(
        top_n=args.top_n,
        initial_cash=args.initial_cash,
        leverage=args.leverage,
        margin_rate_annual=args.margin_rate,
        leverage_soft_cap=args.leverage_soft_cap,
        leverage_penalty_rate=args.leverage_penalty_rate,
        hold_penalty_start_days=args.hold_penalty_start_days,
        hold_penalty_rate=args.hold_penalty_rate,
    )

    logger.info(
        "Running simulation: %s to %s, top_n=%d, initial=$%.0f, leverage=%.1fx",
        start_date,
        end_date,
        args.top_n,
        args.initial_cash,
        args.leverage,
    )

    def progress_callback(day_num: int, total_days: int, day_result):
        if day_num % 50 == 0:
            logger.info(
                "Progress: %d/%d days (%.1f%%) - Portfolio: $%.2f",
                day_num,
                total_days,
                day_num / total_days * 100,
                day_result.ending_portfolio_value,
            )

    result = run_simulation(
        data_config,
        forecast_config,
        sim_config,
        progress_callback=progress_callback,
    )

    # Print and save results
    print_simulation_summary(result)

    output_dir = Path(args.output_dir)
    config_used = {
        "top_n": args.top_n,
        "initial_cash": args.initial_cash,
        "leverage": args.leverage,
        "margin_rate": args.margin_rate,
        "leverage_soft_cap": args.leverage_soft_cap,
        "leverage_penalty_rate": args.leverage_penalty_rate,
        "hold_penalty_start_days": args.hold_penalty_start_days,
        "hold_penalty_rate": args.hold_penalty_rate,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "tuned": args.tune,
    }
    save_results(result, output_dir, config_used)

    return 0


if __name__ == "__main__":
    sys.exit(main())
