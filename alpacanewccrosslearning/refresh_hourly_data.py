from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from loguru import logger

from src.hourly_data_refresh import HourlyDataRefresher
from src.hourly_data_utils import HourlyDataValidator, summarize_statuses


def _parse_symbols(raw: str) -> List[str]:
    symbols = [token.strip().upper() for token in raw.split(",") if token.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh hourly CSVs for Alpaca cross-learning symbols.")
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--data-root", default="trainingdatahourly")
    parser.add_argument("--backfill-hours", type=int, default=48)
    parser.add_argument("--overlap-hours", type=int, default=2)
    parser.add_argument("--crypto-max-staleness-hours", type=float, default=1.5)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    data_root = Path(args.data_root)
    validator = HourlyDataValidator(data_root, max_staleness_hours=6)
    refresher = HourlyDataRefresher(
        data_root,
        validator,
        backfill_hours=args.backfill_hours,
        overlap_hours=args.overlap_hours,
        crypto_max_staleness_hours=args.crypto_max_staleness_hours,
        sleep_seconds=args.sleep_seconds,
    )
    statuses, issues = refresher.refresh(symbols)
    logger.info("Hourly refresh complete: {}", summarize_statuses(statuses))
    if issues:
        logger.warning("Issues: {}", ", ".join(sorted({issue.reason for issue in issues})))


if __name__ == "__main__":
    main()
