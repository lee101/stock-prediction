#!/usr/bin/env python3
"""
Single-command daily refresh that:
1. Downloads + appends the latest OHLC CSV rows into trainingdata/train.
2. Regenerates Chronos forecast caches for the refreshed symbols.

The script is idempotent (no duplicate rows or forecasts) and guarded by a
process-level lock so overlapping invocations safely serialize work.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import fcntl
from loguru import logger

from update_daily_data import (
    TRAINING_DIR,
    download_and_sync,
    list_symbols,
    _storage_symbol,
)
from update_key_forecasts import collect_forecasts

LOCK_FILE = Path(".locks/refresh_daily_inputs.lock")


class FileLock:
    """Simple advisory file lock for process coordination."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w")
        fcntl.flock(self._handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle:
            fcntl.flock(self._handle, fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def _resolve_symbols(explicit: Optional[Sequence[str]]) -> List[str]:
    if explicit:
        return [symbol.upper() for symbol in explicit]
    return list_symbols(TRAINING_DIR)


def refresh_daily_inputs(
    *,
    symbols: Optional[Sequence[str]] = None,
    skip_forecasts: bool = False,
    use_lock: bool = True,
) -> None:
    def _run() -> None:
        selected_symbols = _resolve_symbols(symbols)
        if not selected_symbols:
            raise RuntimeError("No symbols available to refresh.")

        logger.info("Refreshing %d symbols.", len(selected_symbols))
        appended_map = download_and_sync(selected_symbols)
        total_new_rows = sum(appended_map.values())
        logger.info("Data sync added %d new rows.", total_new_rows)

        if skip_forecasts:
            logger.info("Skipping forecast refresh per --skip-forecasts.")
            return

        forecast_symbols = [_storage_symbol(symbol) for symbol in selected_symbols]
        logger.info("Updating Chronos forecasts for %d symbols.", len(forecast_symbols))
        rc = collect_forecasts(forecast_symbols)
        if rc != 0:
            raise RuntimeError(f"Chronos forecast refresh exited with status {rc}.")

    if use_lock:
        with FileLock(LOCK_FILE):
            _run()
    else:
        _run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh training data and Chronos forecasts.")
    parser.add_argument(
        "--symbol",
        dest="symbols",
        action="append",
        help="Specific symbol to refresh (can pass multiple times). Defaults to all training symbols.",
    )
    parser.add_argument(
        "--skip-forecasts",
        action="store_true",
        help="Only refresh CSV data; leave Chronos caches untouched.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        refresh_daily_inputs(symbols=args.symbols, skip_forecasts=args.skip_forecasts, use_lock=True)
    except Exception as exc:
        logger.exception("Refresh failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
