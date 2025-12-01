#!/usr/bin/env python3
"""Create basis points (bp) formatted training data from existing trainingdata/.

Transforms OHLC price data into daily percentage changes in basis points format,
suitable for training Chronos2 on bp-style time series.

Output format: daily bp changes (close_bps, high_bps, low_bps)
- close_bps: daily close change in basis points (e.g., +45 = +0.45%)
- high_bps: intraday high above close in bps
- low_bps: intraday low below close in bps

Usage:
    python create_trainingdatabp.py  # Process all files
    python create_trainingdatabp.py --symbols AAPL MSFT  # Specific symbols
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


INPUT_DIR = Path("trainingdata")
OUTPUT_DIR = Path("trainingdatabp")


def transform_to_bp(df: pd.DataFrame) -> pd.DataFrame:
    """Transform OHLC price data to basis points changes.

    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume

    Returns:
        DataFrame with columns: timestamp, close_bps, high_bps, low_bps
    """
    if len(df) < 2:
        return pd.DataFrame()

    df = df.copy()

    # Ensure proper column names
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "timestamp"})
    if "Timestamp" in df.columns:
        df = df.rename(columns={"Timestamp": "timestamp"})

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Required columns
    required = ["close", "high", "low"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Compute values
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    # close_bps: daily close change in basis points (1 bp = 0.01%)
    # pct_change * 10000 = basis points
    close_pct = np.diff(close) / close[:-1]
    close_bps = close_pct * 10000

    # high_bps: how much the high was above close that day (in bps)
    high_pct = (high[1:] - close[1:]) / close[1:]
    high_bps = high_pct * 10000

    # low_bps: how much the low was below close that day (in bps, typically negative)
    low_pct = (low[1:] - close[1:]) / close[1:]
    low_bps = low_pct * 10000

    # Build output DataFrame (skip first row since we need previous close)
    if "timestamp" in df.columns:
        timestamps = df["timestamp"].values[1:]
    else:
        timestamps = np.arange(len(close_bps))

    return pd.DataFrame({
        "timestamp": timestamps,
        "close_bps": np.round(close_bps, 2),
        "high_bps": np.round(high_bps, 2),
        "low_bps": np.round(low_bps, 2),
        # Keep original close for reference/reconstruction
        "close_price": close[1:],
    })


def process_symbol(symbol: str, input_dir: Path, output_dir: Path) -> bool:
    """Process a single symbol's data.

    Returns:
        True if successful, False otherwise
    """
    input_file = input_dir / f"{symbol}.csv"
    output_file = output_dir / f"{symbol}.csv"

    if not input_file.exists():
        logger.warning(f"Input file not found: {input_file}")
        return False

    try:
        df = pd.read_csv(input_file)
        bp_df = transform_to_bp(df)

        if bp_df.empty:
            logger.warning(f"Empty result for {symbol}")
            return False

        bp_df.to_csv(output_file, index=False)
        logger.info(f"Created {output_file} with {len(bp_df)} rows")
        return True

    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return False


def compute_stats(bp_df: pd.DataFrame) -> dict:
    """Compute summary statistics for bp data."""
    close_bps = bp_df["close_bps"].values

    return {
        "count": len(close_bps),
        "mean_bps": float(np.mean(close_bps)),
        "std_bps": float(np.std(close_bps)),
        "min_bps": float(np.min(close_bps)),
        "max_bps": float(np.max(close_bps)),
        "median_bps": float(np.median(close_bps)),
        "positive_pct": float(np.mean(close_bps > 0) * 100),
    }


def main():
    parser = argparse.ArgumentParser(description="Create bp-formatted training data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Specific symbols to process (default: all)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Input directory with OHLC CSVs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for bp-formatted CSVs"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    # Get symbols to process
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = [f.stem for f in args.input_dir.glob("*.csv")]

    logger.info(f"Processing {len(symbols)} symbols...")

    success_count = 0
    all_stats = []

    for symbol in sorted(symbols):
        if process_symbol(symbol, args.input_dir, args.output_dir):
            success_count += 1

            if args.stats:
                bp_df = pd.read_csv(args.output_dir / f"{symbol}.csv")
                stats = compute_stats(bp_df)
                stats["symbol"] = symbol
                all_stats.append(stats)

    logger.info(f"Successfully processed {success_count}/{len(symbols)} symbols")

    if args.stats and all_stats:
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS (in basis points)")
        print("=" * 70)

        # Aggregate stats
        all_means = [s["mean_bps"] for s in all_stats]
        all_stds = [s["std_bps"] for s in all_stats]

        print(f"Average daily return: {np.mean(all_means):.2f} bps")
        print(f"Average daily std: {np.mean(all_stds):.2f} bps")

        print("\nPer-symbol stats:")
        for stats in sorted(all_stats, key=lambda x: x["std_bps"], reverse=True)[:10]:
            print(f"  {stats['symbol']:10s}: mean={stats['mean_bps']:+6.1f} bps, "
                  f"std={stats['std_bps']:6.1f} bps, "
                  f"range=[{stats['min_bps']:+7.1f}, {stats['max_bps']:+7.1f}]")


if __name__ == "__main__":
    main()
