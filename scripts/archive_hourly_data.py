#!/usr/bin/env python3
"""Archive large hourly CSV datasets while keeping a recent window.

Moves the full CSV into an archive directory and writes a trimmed file
containing only the most recent N rows.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple

from src.data_loading_utils import read_csv_tail

logger = logging.getLogger("archive_hourly_data")


def _archive_file(
    path: Path,
    archive_dir: Path,
    keep_rows: int,
    *,
    chunksize: int,
    min_bytes: int,
    dry_run: bool,
) -> Tuple[str, int]:
    if not path.exists():
        return "missing", 0

    size_bytes = path.stat().st_size
    if size_bytes < min_bytes:
        return "small", 0

    tail_df, total_rows = read_csv_tail(
        path,
        max_rows=keep_rows,
        chunksize=chunksize,
        low_memory=False,
        return_total=True,
    )

    if total_rows <= keep_rows:
        return "already_small", total_rows

    if dry_run:
        logger.info(
            "Would archive %s (%d rows, %.1f MB) keeping last %d rows",
            path,
            total_rows,
            size_bytes / (1024 * 1024),
            keep_rows,
        )
        return "dry_run", total_rows

    temp_path = path.with_suffix(path.suffix + ".tmp")
    tail_df.to_csv(temp_path, index=False)

    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    archive_path = archive_dir / f"{path.stem}_full_{stamp}.csv"

    path.replace(archive_path)
    temp_path.replace(path)

    logger.info(
        "Archived %s -> %s (rows=%d); wrote trimmed file with %d rows",
        path.name,
        archive_path,
        total_rows,
        len(tail_df),
    )
    return "archived", total_rows


def _iter_symbol_files(root: Path, asset_dir: str) -> Iterable[Path]:
    target = root / asset_dir
    if not target.exists():
        return []
    return sorted(target.glob("*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive large hourly CSV datasets.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("trainingdatahourly"),
        help="Root directory containing crypto/stocks folders.",
    )
    parser.add_argument(
        "--asset-class",
        choices=["crypto", "stocks", "both"],
        default="both",
        help="Which asset class to process.",
    )
    parser.add_argument(
        "--keep-hours",
        type=int,
        default=24 * 180,
        help="Number of most recent rows (hours) to keep in the active file.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("trainingdatahourly/archive"),
        help="Directory to store archived full datasets.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="CSV chunk size for streaming reads.",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=50_000_000,
        help="Minimum file size in bytes before archiving is considered.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without modifying files.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.keep_hours <= 0:
        raise ValueError("--keep-hours must be positive")

    assets = ["crypto", "stocks"] if args.asset_class == "both" else [args.asset_class]
    for asset in assets:
        files = list(_iter_symbol_files(args.data_root, asset))
        if not files:
            logger.info("No %s files found under %s", asset, args.data_root)
            continue
        archive_root = args.archive_dir / asset
        for path in files:
            status, total_rows = _archive_file(
                path,
                archive_root,
                args.keep_hours,
                chunksize=args.chunksize,
                min_bytes=args.min_bytes,
                dry_run=args.dry_run,
            )
            if status == "archived":
                logger.info("%s archived (%d rows)", path.name, total_rows)


if __name__ == "__main__":
    main()
