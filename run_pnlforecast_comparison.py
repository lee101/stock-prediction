#!/usr/bin/env python3
"""Run comparison of PnL forecast strategies with realistic capital constraints.

Compares:
1. Buy-and-hold baseline (BTCUSD)
2. Buy-and-hold best performers (SOLUSD, UNIUSD)
3. Fixed threshold strategy on SOLUSD
4. Chronos adaptive thresholds (uses predicted high/low)
5. Chronos symbol selection (picks best symbol each day)
6. Full Chronos (adaptive thresholds + symbol selection)

Usage:
    python run_pnlforecast_comparison.py [--quick] [--start 2025-01-01] [--end 2025-12-31]
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
        logging.FileHandler("pnlforecast_comparison.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PnL forecast strategies")
    parser.add_argument("--quick", action="store_true", help="Quick test (30 days)")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Initial capital")
    parser.add_argument("--output", type=str, default="pnlforecast_results", help="Output dir")
    return parser.parse_args()


def main():
    args = parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    if args.quick:
        logger.info("Running in QUICK mode (30 days)")
        end_date = min(end_date, start_date + timedelta(days=30))

    # All symbols to compare
    symbols = [
        "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD",
        "NFLX", "AVGO", "ADBE", "CRM", "COST", "COIN", "SHOP",
        "BTCUSD", "ETHUSD", "SOLUSD", "UNIUSD",
    ]

    logger.info("=" * 80)
    logger.info("PnL Forecast Strategy Comparison")
    logger.info("=" * 80)
    logger.info("Start Date: %s", start_date)
    logger.info("End Date: %s", end_date)
    logger.info("Initial Capital: $%.2f", args.initial_cash)
    logger.info("Symbols: %d total", len(symbols))
    logger.info("=" * 80)

    from pnlforecast.comparison import (
        run_comparison,
        print_comparison,
        save_comparison,
    )

    # Run comparison
    results = run_comparison(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.initial_cash,
        baseline_symbol="BTCUSD",
    )

    # Print results
    print_comparison(results)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_comparison(results, output_dir / f"comparison_{start_date}_{end_date}.json")

    # Print best strategy
    best = max(results.items(), key=lambda x: x[1].total_return_pct)
    logger.info("")
    logger.info("BEST STRATEGY: %s", best[0])
    logger.info("  Total Return: %.2f%%", best[1].total_return_pct)
    logger.info("  Alpha vs Buy-Hold: %.2f%%", best[1].alpha_pct)
    logger.info("  Sharpe Ratio: %.2f", best[1].sharpe_ratio)
    logger.info("  Win Rate: %.1f%%", best[1].win_rate * 100)

    return results


if __name__ == "__main__":
    main()
