#!/usr/bin/env python3
"""Backfill hourly crypto bars from Alpaca for a specific date range."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alpaca_data_wrapper import backfill_crypto_range

logger = logging.getLogger("backfill_hourly_crypto")


def _parse_timestamp(value: str) -> datetime:
    ts = pd.to_datetime(value, utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value}")
    return ts.to_pydatetime()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill hourly crypto data from Alpaca.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to backfill (e.g., SOLUSD BTCUSD)")
    parser.add_argument("--start", required=True, help="Start timestamp (e.g., 2025-04-16T00:00:00Z)")
    parser.add_argument("--end", required=True, help="End timestamp (e.g., 2025-12-20T00:00:00Z)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("trainingdatahourly") / "crypto",
        help="Output directory for crypto CSVs.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between symbols (default: 0).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Log actions without writing files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    start = _parse_timestamp(args.start)
    end = _parse_timestamp(args.end)
    if end <= start:
        raise ValueError("--end must be after --start")

    results = backfill_crypto_range(
        args.symbols,
        start=start,
        end=end,
        output_dir=args.output_dir,
        sleep_seconds=args.sleep,
        dry_run=args.dry_run,
    )

    for result in results:
        logger.info(
            "%s | status=%s appended=%d total=%d start=%s end=%s",
            result.symbol,
            result.status,
            result.appended,
            result.total,
            result.start,
            result.end,
        )


if __name__ == "__main__":
    main()
