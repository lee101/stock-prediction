#!/usr/bin/env python3
"""Download S&P500 OHLCV data for RL training.

Usage:
  python scripts/download_sp500_data.py --dry-run --limit 5
  python scripts/download_sp500_data.py --output-dir trainingdatadaily/stocks/
  python scripts/download_sp500_data.py --symbols-file my_symbols.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd


SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_OUTPUT_DIR = "trainingdatadaily/stocks"
DEFAULT_START_DATE = "2020-01-01"
BATCH_SIZE = 50
BATCH_DELAY_SECONDS = 1.0


def fetch_sp500_symbols_from_wikipedia() -> list[str]:
    """Fetch current S&P 500 constituent symbols from Wikipedia."""
    tables = pd.read_html(SP500_WIKIPEDIA_URL)
    df = tables[0]
    col = next((c for c in df.columns if str(c).strip().lower() in ("symbol", "ticker")), df.columns[0])
    symbols = [str(s).strip().replace(".", "-") for s in df[col].tolist() if str(s).strip()]
    return symbols


def load_symbols_from_file(path: str) -> list[str]:
    """Load one symbol per line from a text file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip().upper() for line in lines if line.strip() and not line.startswith("#")]


def save_symbol_list(symbols: list[str], output_dir: Path) -> None:
    """Save the symbol list to {output_dir}/sp500_symbols.txt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "sp500_symbols.txt"
    dest.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    print(f"Saved symbol list ({len(symbols)} symbols) → {dest}")


def download_symbols(
    symbols: list[str],
    output_dir: Path,
    *,
    start_date: str,
    end_date: str,
    force: bool = False,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Download OHLCV data for each symbol via yfinance.

    Returns (downloaded_count, skipped_count).
    """
    import yfinance as yf  # noqa: PLC0415  (lazy import — optional dep)

    downloaded = 0
    skipped = 0
    total = len(symbols)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = symbols[batch_start : batch_start + BATCH_SIZE]

        for symbol in batch:
            dest = output_dir / f"{symbol}.csv"
            if dest.exists() and not force:
                skipped += 1
                continue

            if dry_run:
                print(f"  [dry-run] Would download {symbol} → {dest}")
                downloaded += 1
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                )
                if df.empty:
                    print(f"  WARNING: No data for {symbol}", file=sys.stderr)
                    skipped += 1
                    continue

                # Flatten multi-level columns (yfinance can produce them)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [str(col[0]).lower() for col in df.columns]
                else:
                    df.columns = [str(c).lower() for c in df.columns]

                df.index.name = "date"
                df = df.reset_index()
                df.to_csv(dest, index=False)
                downloaded += 1
            except Exception as exc:
                print(f"  ERROR downloading {symbol}: {exc}", file=sys.stderr)
                skipped += 1
                continue

        done = min(batch_start + BATCH_SIZE, total)
        print(f"Downloaded {done}/{total} symbols...")

        # Throttle between batches (not after the last one)
        if batch_start + BATCH_SIZE < total and not dry_run:
            time.sleep(BATCH_DELAY_SECONDS)

    return downloaded, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download S&P500 OHLCV data for RL training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save CSV files",
    )
    parser.add_argument(
        "--symbols-file",
        default=None,
        help="Path to a text file with one symbol per line (skips Wikipedia fetch)",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Start date for OHLCV download (ISO format)",
    )
    parser.add_argument(
        "--end-date",
        default=str(date.today()),
        help="End date for OHLCV download (ISO format)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N symbols (useful for testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the CSV already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)

    if args.symbols_file:
        print(f"Loading symbols from {args.symbols_file}...")
        symbols = load_symbols_from_file(args.symbols_file)
    else:
        print("Fetching S&P 500 constituent list from Wikipedia...")
        symbols = fetch_sp500_symbols_from_wikipedia()

    print(f"Found {len(symbols)} symbols total")

    if args.limit is not None:
        symbols = symbols[: args.limit]
        print(f"Limiting to first {len(symbols)} symbols (--limit {args.limit})")

    if not args.dry_run:
        save_symbol_list(symbols, output_dir)

    downloaded, skipped = download_symbols(
        symbols,
        output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        force=args.force,
        dry_run=args.dry_run,
    )

    print(f"Done: {downloaded} downloaded, {skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
