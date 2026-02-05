from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from loguru import logger

from src.hourly_data_refresh import HourlyDataRefresher
from src.hourly_data_utils import HourlyDataValidator, summarize_statuses

from .symbols import (
    DEFAULT_LONG_CRYPTO,
    LONGABLE_STOCKS,
    SHORTABLE_STOCKS,
    normalize_symbols,
)


def _parse_symbols(raw: str | None) -> List[str]:
    if raw is None:
        return []
    symbols = normalize_symbols([token for token in raw.split(",") if token.strip()])
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def _merge_defaults(
    *,
    symbols: Sequence[str],
    long_stocks: Sequence[str],
    short_stocks: Sequence[str],
    crypto: Sequence[str],
) -> List[str]:
    if symbols:
        return normalize_symbols(symbols)
    return normalize_symbols(list(long_stocks) + list(short_stocks) + list(crypto))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh hourly CSVs for constrained Alpaca symbols (longable + shortable)."
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols to refresh.")
    parser.add_argument(
        "--long-stocks",
        default=",".join(LONGABLE_STOCKS),
        help="Comma-separated longable stock symbols.",
    )
    parser.add_argument(
        "--short-stocks",
        default=",".join(SHORTABLE_STOCKS),
        help="Comma-separated shortable stock symbols.",
    )
    parser.add_argument(
        "--crypto",
        default=",".join(DEFAULT_LONG_CRYPTO),
        help="Comma-separated crypto symbols to refresh.",
    )
    parser.add_argument("--data-root", default="trainingdatahourly")
    parser.add_argument("--backfill-hours", type=int, default=48)
    parser.add_argument("--overlap-hours", type=int, default=2)
    parser.add_argument("--crypto-max-staleness-hours", type=float, default=1.5)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    symbols = _merge_defaults(
        symbols=_parse_symbols(args.symbols),
        long_stocks=_parse_symbols(args.long_stocks),
        short_stocks=_parse_symbols(args.short_stocks),
        crypto=_parse_symbols(args.crypto),
    )

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
