#!/usr/bin/env python3
"""Runner script for PnL Forecast meta-strategy backtest.

This script runs the full 365-day backtest of the PnL Forecast strategy:
1. Uses Chronos2 to forecast OHLC for each symbol
2. Simulates buy-at-low/sell-at-high strategies on historical data
3. Uses Chronos2 to forecast each strategy's future PnL
4. Selects the best strategy based on forecasted PnL
5. Tracks overall performance

Usage:
    python run_pnlforecast.py [--quick] [--symbols AAPL,MSFT] [--start 2025-01-01] [--end 2025-12-31]
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pnlforecast_backtest.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PnL Forecast meta-strategy backtest"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with limited symbols and date range",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols to trade",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2025-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="Initial cash amount",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pnlforecast_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--pnl-history-days",
        type=int,
        default=7,
        help="Days of PnL history for forecasting",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Quick mode overrides
    if args.quick:
        logger.info("Running in QUICK mode")
        symbols = symbols or ["AAPL", "MSFT", "BTCUSD"]
        # Use 30 days for quick test
        end_date = min(end_date, start_date + timedelta(days=30))

    logger.info("=" * 60)
    logger.info("PnL Forecast Meta-Strategy Backtest")
    logger.info("=" * 60)
    logger.info("Start Date: %s", start_date)
    logger.info("End Date: %s", end_date)
    logger.info("Symbols: %s", symbols or "All")
    logger.info("Initial Cash: $%.2f", args.initial_cash)
    logger.info("PnL History Days: %d", args.pnl_history_days)
    logger.info("=" * 60)

    # Import here to avoid slow startup if just getting help
    from pnlforecast.config import (
        DataConfigPnL,
        ForecastConfigPnL,
        StrategyConfigPnL,
        SimulationConfigPnL,
        FullConfigPnL,
    )
    from pnlforecast.backtester import (
        PnLForecastBacktester,
        save_results,
    )

    # Configure
    data_config = DataConfigPnL(
        start_date=start_date,
        end_date=end_date,
    )

    forecast_config = ForecastConfigPnL(
        pnl_context_days=args.pnl_history_days,
    )

    strategy_config = StrategyConfigPnL()

    sim_config = SimulationConfigPnL(
        initial_cash=args.initial_cash,
        min_pnl_history_days=args.pnl_history_days,
        output_dir=Path(args.output),
    )

    config = FullConfigPnL(
        data=data_config,
        forecast=forecast_config,
        strategy=strategy_config,
        simulation=sim_config,
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Progress callback
    start_time = datetime.now()
    last_log_time = start_time

    def progress_callback(day_num: int, total_days: int, trade_date: date):
        nonlocal last_log_time
        now = datetime.now()
        elapsed = (now - start_time).total_seconds()

        # Log every 60 seconds
        if (now - last_log_time).total_seconds() >= 60:
            days_per_sec = day_num / elapsed if elapsed > 0 else 0
            remaining_days = total_days - day_num
            eta_seconds = remaining_days / days_per_sec if days_per_sec > 0 else 0
            eta_hours = eta_seconds / 3600

            logger.info(
                "Progress: %d/%d days (%.1f%%) - Date: %s - ETA: %.1f hours",
                day_num,
                total_days,
                100 * day_num / total_days,
                trade_date,
                eta_hours,
            )
            last_log_time = now

    # Run backtest
    logger.info("Starting backtest...")
    backtester = PnLForecastBacktester(config)

    try:
        result = backtester.run(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            progress_callback=progress_callback,
        )

        # Print summary
        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 60)
        logger.info("Final Portfolio Value: $%.2f", result.final_portfolio_value)
        logger.info("Total Return: %.2f%%", result.total_return * 100)
        logger.info("Annualized Return: %.2f%%", result.annualized_return * 100)
        logger.info("Sharpe Ratio: %.2f", result.sharpe_ratio)
        logger.info("Max Drawdown: %.2f%%", result.max_drawdown * 100)
        logger.info("Win Rate: %.2f%%", result.win_rate * 100)
        logger.info("Total Trades: %d", result.total_trades)
        logger.info("Total Days: %d", result.total_days)
        logger.info("Prediction Accuracy: %.2f", result.prediction_accuracy)
        logger.info("=" * 60)

        # Per-symbol summary
        logger.info("Per-Symbol Performance:")
        for symbol, sr in sorted(
            result.symbol_results.items(),
            key=lambda x: x[1].total_pnl,
            reverse=True,
        ):
            logger.info(
                "  %s: PnL=%.2f%%, Trades=%d, WinRate=%.1f%%",
                symbol,
                sr.total_pnl * 100,
                sr.total_trades,
                sr.win_rate * 100,
            )

        # Strategy usage
        logger.info("Top Strategy Selections:")
        sorted_strategies = sorted(
            result.strategy_selection_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        for strat_id, count in sorted_strategies:
            logger.info("  %s: %d times", strat_id, count)

        # Save results
        result_path = output_dir / f"backtest_{start_date}_{end_date}.json"
        save_results(result, result_path)

        # Save equity curve
        equity_path = output_dir / f"equity_curve_{start_date}_{end_date}.csv"
        result.equity_curve.to_csv(equity_path, header=["portfolio_value"])
        logger.info("Saved equity curve to %s", equity_path)

        # Final PnL score
        pnl_score = result.total_return * 100
        logger.info("=" * 60)
        logger.info("FINAL PnL SCORE: %.2f%%", pnl_score)
        logger.info("=" * 60)

        return result

    finally:
        backtester.cleanup()


if __name__ == "__main__":
    main()
