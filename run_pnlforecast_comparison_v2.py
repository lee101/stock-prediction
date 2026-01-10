#!/usr/bin/env python3
"""Run rigorous strategy comparison v2."""

import logging
import sys
from datetime import date
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Symbols to trade
STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "AVGO", "ADBE", "CRM", "COST", "COIN", "SHOP",
]
CRYPTO_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD", "UNIUSD"]
ALL_SYMBOLS = STOCK_SYMBOLS + CRYPTO_SYMBOLS


def main():
    from pnlforecast.comparison_v2 import (
        run_rigorous_comparison,
        print_rigorous_comparison,
        analyze_trades,
    )

    # Full year backtest
    start_date = date(2025, 1, 1)
    end_date = date(2025, 12, 31)
    initial_capital = 100_000.0

    logger.info("=" * 80)
    logger.info("RIGOROUS PnL Forecast Strategy Comparison V2")
    logger.info("=" * 80)
    logger.info("Start Date: %s", start_date)
    logger.info("End Date: %s", end_date)
    logger.info("Initial Capital: $%.2f", initial_capital)
    logger.info("Symbols: %d total", len(ALL_SYMBOLS))
    logger.info("=" * 80)

    results = run_rigorous_comparison(
        symbols=ALL_SYMBOLS,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )

    # Print comparison
    print_rigorous_comparison(results)

    # Analyze best strategy
    best_result = max(results.values(), key=lambda r: r.total_return_pct)
    analyze_trades(best_result)

    # Also analyze SAME_DAY_ONLY top1 (most realistic)
    realistic_key = "chronos_full_v2_same_day_only_top1"
    if realistic_key in results:
        print("\n" + "=" * 80)
        print("MOST REALISTIC STRATEGY (same_day_only, top1):")
        print("=" * 80)
        r = results[realistic_key]
        print(f"Return: {r.total_return_pct:.2f}%")
        print(f"Sharpe: {r.sharpe_ratio:.2f}")
        print(f"Max DD: {r.max_drawdown_pct:.2f}%")
        print(f"Win Rate: {r.win_rate*100:.1f}%")
        print(f"Total Trades: {r.total_trades}")
        print(f"  Round-trip same day: {r.round_trip_same_day}")
        print(f"  Forced exit at close: {r.forced_exits_at_close}")
        analyze_trades(r)


if __name__ == "__main__":
    main()
