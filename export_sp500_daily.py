#!/usr/bin/env python3
"""Export S&P500 daily OHLCV CSVs to MKTD v2 binary files for RL training.

Usage:
  python scripts/export_sp500_daily.py --dry-run
  python scripts/export_sp500_daily.py --data-dir trainingdatadaily/stocks/
  python scripts/export_sp500_daily.py --batch-size 30 --val-split-date 2025-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Export logic lives in export_data_daily (root-level module mirrored to pufferlib_market)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from export_data_daily import export_binary  # noqa: E402
from scripts.download_sp500_data import load_symbols_from_file  # noqa: E402

DEFAULT_DATA_DIR = "trainingdatadaily/stocks"
DEFAULT_OUTPUT_DIR = "pufferlib_market/data"
DEFAULT_BATCH_SIZE = 50
DEFAULT_VAL_SPLIT_DATE = "2025-06-01"


def find_csv_symbols(data_dir: Path) -> list[str]:
    """Return sorted list of symbols from CSV files in data_dir."""
    csvs = sorted(data_dir.glob("*.csv"))
    symbols = []
    for csv_path in csvs:
        name = csv_path.stem.upper()
        # Skip the symbol list file itself
        if name.lower() == "sp500_symbols":
            continue
        symbols.append(name)
    return symbols


def group_into_batches(symbols: list[str], batch_size: int) -> list[list[str]]:
    """Split symbols into batches of at most batch_size."""
    batches = []
    for i in range(0, len(symbols), batch_size):
        batches.append(symbols[i : i + batch_size])
    return batches


def export_batch(
    batch_idx: int,
    total_batches: int,
    symbols: list[str],
    data_dir: Path,
    output_dir: Path,
    val_split_date: str,
    *,
    dry_run: bool = False,
) -> None:
    """Export one batch to train + val binary files."""
    train_path = output_dir / f"sp500_batch_{batch_idx}_daily_train.bin"
    val_path = output_dir / f"sp500_batch_{batch_idx}_daily_val.bin"

    if dry_run:
        print(
            f"  [dry-run] batch {batch_idx}/{total_batches}: {len(symbols)} symbols"
            f" → {train_path.name} / {val_path.name}"
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        export_binary(
            symbols=symbols,
            data_root=data_dir,
            output_path=train_path,
            end_date=val_split_date,
            min_days=60,
        )
        train_mb = train_path.stat().st_size / (1024 * 1024)
    except Exception as exc:
        print(f"  WARNING: train export failed for batch {batch_idx}: {exc}", file=sys.stderr)
        train_mb = 0.0

    try:
        export_binary(
            symbols=symbols,
            data_root=data_dir,
            output_path=val_path,
            start_date=val_split_date,
            min_days=30,
        )
        val_mb = val_path.stat().st_size / (1024 * 1024)
    except Exception as exc:
        print(f"  WARNING: val export failed for batch {batch_idx}: {exc}", file=sys.stderr)
        val_mb = 0.0

    print(
        f"  Exported batch {batch_idx}/{total_batches}: {len(symbols)} symbols"
        f" → {train_path.name} ({train_mb:.1f}MB)"
        f" / {val_path.name} ({val_mb:.1f}MB)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export S&P500 daily CSVs to MKTD v2 binary files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing per-symbol daily OHLCV CSV files",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write .bin output files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of symbols per .bin file (max 64 per MKTD v2 format)",
    )
    parser.add_argument(
        "--val-split-date",
        default=DEFAULT_VAL_SPLIT_DATE,
        help="ISO date to split train/val",
    )
    parser.add_argument(
        "--symbols-file",
        default=None,
        help="Optional: text file with one symbol per line (overrides CSV auto-discovery)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be exported without writing files",
    )
    args = parser.parse_args(argv)

    if args.batch_size > 64:
        print(
            f"WARNING: --batch-size {args.batch_size} exceeds MKTD v2 limit of 64; capping to 64.",
            file=sys.stderr,
        )
        args.batch_size = 64

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.symbols_file:
        symbols = load_symbols_from_file(args.symbols_file)
        print(f"Loaded {len(symbols)} symbols from {args.symbols_file}")
    else:
        if not data_dir.is_dir():
            print(f"ERROR: data directory does not exist: {data_dir}", file=sys.stderr)
            return 1
        symbols = find_csv_symbols(data_dir)
        print(f"Found {len(symbols)} CSV files in {data_dir}")

    if not symbols:
        print("ERROR: No symbols found.", file=sys.stderr)
        return 1

    batches = group_into_batches(symbols, args.batch_size)
    total_batches = len(batches)

    print(
        f"Exporting {len(symbols)} symbols → {total_batches} batches"
        f" (batch_size={args.batch_size}, val_split={args.val_split_date})"
    )

    for i, batch in enumerate(batches, start=1):
        export_batch(
            batch_idx=i,
            total_batches=total_batches,
            symbols=batch,
            data_dir=data_dir,
            output_dir=output_dir,
            val_split_date=args.val_split_date,
            dry_run=args.dry_run,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
