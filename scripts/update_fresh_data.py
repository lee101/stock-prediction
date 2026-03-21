#!/usr/bin/env python3
"""Update mixed23/mixed32 binary files with the freshest available data.

Checks what data CSVs are available in trainingdata/ and trainingdatahourly/,
then re-exports the binary files for mixed23 and mixed32 with the latest data
cutoff (today as val end date).

Usage:
  python scripts/update_fresh_data.py --dry-run
  python scripts/update_fresh_data.py --val-start 2025-06-01 --output-dir pufferlib_market/data/
  python scripts/update_fresh_data.py --universe mixed23 --val-start 2025-06-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Symbol universes
MIXED23_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "PLTR", "NET",
    "JPM", "V", "SPY", "QQQ", "NFLX", "AMD",
    "BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD", "DOGEUSD", "LINKUSD", "AAVEUSD",
]

MIXED32_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "PLTR", "NET",
    "JPM", "V", "SPY", "QQQ", "NFLX", "AMD",
    "AFRM", "PANW", "SNOW", "DDOG", "COIN", "HOOD", "UBER", "PYPL", "ROKU",
    "BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD", "DOGEUSD", "LINKUSD", "AAVEUSD",
]

UNIVERSES = {
    "mixed23": MIXED23_SYMBOLS,
    "mixed32": MIXED32_SYMBOLS,
}

# Data root directories to search for daily CSVs
DAILY_DATA_ROOTS = [
    "trainingdata",
    "trainingdatadaily",
    "trainingdatadaily/stocks",
]


def find_symbol_csv(symbol: str, data_roots: list[Path]) -> tuple[Path | None, str | None]:
    """Search data_roots for a symbol's CSV. Returns (path, source_label)."""
    sym_upper = symbol.upper()
    for root in data_roots:
        candidates = [
            root / f"{sym_upper}.csv",
            root / "crypto" / f"{sym_upper}.csv",
            root / "stocks" / f"{sym_upper}.csv",
        ]
        for p in candidates:
            if p.exists():
                return p, str(root)
    return None, None


def get_last_date(csv_path: Path) -> str | None:
    """Return the last date from a CSV file as ISO string."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, usecols=lambda c: c in ("timestamp", "date"))
        if df.empty:
            return None
        col = "timestamp" if "timestamp" in df.columns else "date"
        ts = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()
        if ts.empty:
            return None
        return str(ts.max().date())
    except Exception:
        return None


def check_availability(
    symbols: list[str],
    data_roots: list[Path],
) -> tuple[list[str], list[str], dict[str, str]]:
    """Check which symbols are available.

    Returns:
        available: list of symbols with data
        missing: list of symbols without data
        dates: dict mapping symbol -> last date string
    """
    available: list[str] = []
    missing: list[str] = []
    dates: dict[str, str] = {}

    for sym in symbols:
        path, _ = find_symbol_csv(sym, data_roots)
        if path is not None:
            available.append(sym)
            last = get_last_date(path)
            if last:
                dates[sym] = last
        else:
            missing.append(sym)

    return available, missing, dates


def export_binary_for_universe(
    universe_name: str,
    symbols: list[str],
    data_root: Path,
    output_dir: Path,
    val_start: str,
    timestamp: str,
    *,
    dry_run: bool = False,
) -> tuple[Path | None, Path | None]:
    """Export train+val binaries for a universe. Returns (train_path, val_path)."""
    from export_data_daily import export_binary

    train_name = f"{universe_name}_fresh_{timestamp}_train.bin"
    val_name = f"{universe_name}_fresh_{timestamp}_val.bin"
    train_path = output_dir / train_name
    val_path = output_dir / val_name

    end_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    if dry_run:
        print(f"  [DRY RUN] Would export train: {train_path}")
        print(f"            start=earliest available, end={val_start} (exclusive)")
        print(f"  [DRY RUN] Would export val:   {val_path}")
        print(f"            start={val_start}, end={end_date}")
        return train_path, val_path

    # Export train: from earliest to val_start
    print(f"  Exporting train binary: {train_path}")
    try:
        export_binary(
            symbols=symbols,
            data_root=data_root,
            output_path=train_path,
            end_date=val_start,
            min_days=60,
        )
    except Exception as e:
        print(f"  ERROR exporting train: {e}", file=sys.stderr)
        return None, None

    # Export val: from val_start to today
    print(f"  Exporting val binary: {val_path}")
    try:
        export_binary(
            symbols=symbols,
            data_root=data_root,
            output_path=val_path,
            start_date=val_start,
            end_date=end_date,
            min_days=30,
        )
    except Exception as e:
        print(f"  ERROR exporting val: {e}", file=sys.stderr)
        return train_path, None

    return train_path, val_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update mixed23/mixed32 binary files with freshest available data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--universe",
        choices=list(UNIVERSES.keys()) + ["all"],
        default="all",
        help="Which universe to export (default: all)",
    )
    parser.add_argument(
        "--val-start",
        default="2025-06-01",
        help="Train/val split date. Data before this date goes to train, after to val. (default: 2025-06-01)",
    )
    parser.add_argument(
        "--output-dir",
        default="pufferlib_market/data",
        help="Directory to write output .bin files (default: pufferlib_market/data)",
    )
    parser.add_argument(
        "--data-root",
        default="trainingdata",
        help="Primary daily data root directory (default: trainingdata)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be exported without actually writing files",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check symbol availability and dates, do not export",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = ROOT / data_root

    # Build list of data roots to search
    all_data_roots = [ROOT / r for r in DAILY_DATA_ROOTS]

    # Determine universes to process
    if args.universe == "all":
        to_process = list(UNIVERSES.items())
    else:
        to_process = [(args.universe, UNIVERSES[args.universe])]

    timestamp = datetime.now().strftime("%Y%m%d")
    exported_files: list[tuple[str, Path | None, Path | None]] = []

    for universe_name, symbols in to_process:
        print(f"\n{'='*60}")
        print(f"Universe: {universe_name} ({len(symbols)} symbols)")
        print(f"{'='*60}")

        available, missing, dates = check_availability(symbols, all_data_roots)

        if missing:
            print(f"  Missing symbols ({len(missing)}): {', '.join(missing)}")
        print(f"  Available symbols ({len(available)}): {', '.join(available)}")

        if dates:
            min_date = min(dates.values())
            max_date = max(dates.values())
            print(f"  Data dates: min_last={min_date}, max_last={max_date}")
            stale = [s for s, d in dates.items() if d < "2025-06-01"]
            if stale:
                print(f"  WARNING: stale symbols (last date < 2025-06-01): {', '.join(stale)}")

        if args.check_only:
            continue

        if not available:
            print(f"  SKIP: no symbols available for {universe_name}")
            continue

        if missing:
            print(f"  NOTE: {len(missing)} symbols missing, exporting with {len(available)} available.")
            use_symbols = available
        else:
            use_symbols = symbols

        train_path, val_path = export_binary_for_universe(
            universe_name=universe_name,
            symbols=use_symbols,
            data_root=data_root,
            output_dir=output_dir,
            val_start=args.val_start,
            timestamp=timestamp,
            dry_run=args.dry_run,
        )
        exported_files.append((universe_name, train_path, val_path))

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    if args.check_only:
        print("Check-only mode: no files exported.")
        return
    if args.dry_run:
        print("Dry-run mode: no files actually written.")
    for name, train_p, val_p in exported_files:
        if train_p and (args.dry_run or train_p.exists()):
            print(f"  {name} train: {train_p}")
        if val_p and (args.dry_run or val_p.exists()):
            print(f"  {name} val:   {val_p}")
        if not train_p and not val_p:
            print(f"  {name}: FAILED")

    if not args.dry_run and exported_files:
        print("\nTo evaluate with the new data:")
        for name, _, val_p in exported_files:
            if val_p and val_p.exists():
                print(f"  python scripts/eval_all_checkpoints.py --data-path {val_p} --output leaderboard_{name}_{timestamp}.csv")


if __name__ == "__main__":
    main()
