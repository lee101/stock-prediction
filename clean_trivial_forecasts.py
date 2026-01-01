#!/usr/bin/env python3
"""Remove trivial forecasts from the cache.

A trivial forecast is one where:
- predicted_high ≈ predicted_low ≈ predicted_close ≈ context_close

This happens when Chronos fails and the cache uses a fallback persistence fill.
Removing these allows proper regeneration.

Usage:
    python clean_trivial_forecasts.py --dry-run  # Preview only
    python clean_trivial_forecasts.py            # Actually remove
    python clean_trivial_forecasts.py --symbol SPY  # Single symbol
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

CACHE_DIR = Path("strategytraining/forecast_cache")


def is_trivial_forecast(row: pd.Series, tolerance: float = 0.001) -> bool:
    """Check if a forecast row is trivial (zero spread - high == low == close).

    This happens when Chronos fails and falls back to persistence fill.
    Real forecasts should have predicted_high > predicted_low.
    """
    predicted_close = row.get("predicted_close", 0)
    predicted_high = row.get("predicted_high", 0)
    predicted_low = row.get("predicted_low", 0)

    if predicted_close <= 0:
        return False

    # A trivial forecast has high == low == close (zero spread)
    # Real Chronos forecasts will have some spread
    spread = abs(predicted_high - predicted_low)
    relative_spread = spread / predicted_close if predicted_close > 0 else 0

    # Also check if high == close and low == close
    high_eq_close = abs(predicted_high - predicted_close) / predicted_close < tolerance
    low_eq_close = abs(predicted_low - predicted_close) / predicted_close < tolerance

    # Trivial if spread is near zero AND all three are equal
    return relative_spread < tolerance and high_eq_close and low_eq_close


def clean_symbol_forecasts(
    symbol: str,
    dry_run: bool = True,
    start_date: str = None,
) -> tuple[int, int]:
    """Remove trivial forecasts from a symbol's cache.

    Returns: (total_rows, removed_rows)
    """
    path = CACHE_DIR / f"{symbol}.parquet"
    if not path.exists():
        logger.debug(f"No cache file for {symbol}")
        return 0, 0

    df = pd.read_parquet(path)
    if df.empty:
        return 0, 0

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Apply start_date filter if specified
    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        mask = df["timestamp"] >= start_ts
        check_df = df[mask]
    else:
        check_df = df

    # Find trivial rows
    trivial_mask = check_df.apply(is_trivial_forecast, axis=1)
    trivial_timestamps = set(check_df.loc[trivial_mask, "timestamp"])

    if not trivial_timestamps:
        return len(df), 0

    # Remove trivial rows from full dataframe
    keep_mask = ~df["timestamp"].isin(trivial_timestamps)
    cleaned_df = df[keep_mask].reset_index(drop=True)

    removed = len(trivial_timestamps)

    if not dry_run and removed > 0:
        cleaned_df.to_parquet(path, index=False)
        logger.info(f"{symbol}: Removed {removed} trivial forecasts (kept {len(cleaned_df)})")
    elif removed > 0:
        logger.info(f"{symbol}: Would remove {removed} trivial forecasts (keeping {len(cleaned_df)})")

    return len(df), removed


def main():
    parser = argparse.ArgumentParser(description="Clean trivial forecasts from cache")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't modify")
    parser.add_argument("--symbol", type=str, help="Clean only this symbol")
    parser.add_argument("--start-date", type=str, default="2024-11-01",
                        help="Only check forecasts from this date onwards (default: 2024-11-01)")
    parser.add_argument("--all-dates", action="store_true",
                        help="Check all dates, not just recent ones")
    args = parser.parse_args()

    if not CACHE_DIR.exists():
        logger.error(f"Cache directory not found: {CACHE_DIR}")
        return

    start_date = None if args.all_dates else args.start_date

    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = sorted([p.stem for p in CACHE_DIR.glob("*.parquet")])

    logger.info(f"Checking {len(symbols)} symbols for trivial forecasts")
    if start_date:
        logger.info(f"Only checking forecasts from {start_date} onwards")

    total_removed = 0
    total_rows = 0
    affected_symbols = []

    for symbol in symbols:
        rows, removed = clean_symbol_forecasts(
            symbol,
            dry_run=args.dry_run,
            start_date=start_date,
        )
        total_rows += rows
        total_removed += removed
        if removed > 0:
            affected_symbols.append((symbol, removed))

    # Summary
    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN - No changes made")
    print(f"Total rows checked: {total_rows}")
    print(f"Total trivial forecasts: {total_removed}")
    print(f"Affected symbols: {len(affected_symbols)}")

    if affected_symbols:
        print("\nSymbols with trivial forecasts:")
        for sym, count in sorted(affected_symbols, key=lambda x: -x[1])[:20]:
            print(f"  {sym}: {count}")
        if len(affected_symbols) > 20:
            print(f"  ... and {len(affected_symbols) - 20} more")

    if args.dry_run and total_removed > 0:
        print("\nRun without --dry-run to remove trivial forecasts")


if __name__ == "__main__":
    main()
