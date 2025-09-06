#!/usr/bin/env python3
"""
Collect and consolidate local stock CSVs into hftraining/trainingdata.

Sources:
  - ../trainingdata (and subfolders train/test)
  - ../data (timestamped folders and flat files like AAPL-YYYY-MM-DD.csv)

Output:
  - hftraining/trainingdata/{SYMBOL}.csv (deduplicated, sorted by date)
  - hftraining/trainingdata/summary.csv

Usage:
  python -m hftraining.scripts.collect_training_data \
    --sources ../trainingdata ../data \
    --output ./hftraining/trainingdata \
    --min-rows 200 \
    --since 2015-01-01 \
    --symbols-file ./symbolsofinterest.txt
"""

import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional

import pandas as pd


def infer_symbol_from_name(path: Path) -> Optional[str]:
    name = path.stem  # e.g., AAPL-2024-06-22 or AAPL
    # Prefer part before first dash/underscore
    base = re.split(r"[-_]", name)[0].strip()
    # Basic sanity: uppercase letters, numbers, dots (for e.g., BRK.B) or dashes
    if base:
        return base.upper()
    return None


def load_and_standardize(csv_path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        return None
    # Lowercase columns
    df.columns = df.columns.str.lower()
    # Check required columns
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        return None
    # Parse and sort by date if present
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
        except Exception:
            pass
    return df


def collect_sources(sources: List[Path]) -> List[Path]:
    files: List[Path] = []
    for src in sources:
        if not src.exists():
            continue
        # Recursively gather CSVs
        files.extend(src.rglob("*.csv"))
    # Deduplicate by path
    return sorted(set(files))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect local stock CSVs for hftraining")
    parser.add_argument("--sources", nargs="*", default=["../trainingdata", "../data"], help="Source directories to scan recursively")
    parser.add_argument("--output", default="./hftraining/trainingdata", help="Output directory for consolidated CSVs")
    parser.add_argument("--since", default=None, help="ISO date (YYYY-MM-DD) to filter rows on/after this date")
    parser.add_argument("--min-rows", type=int, default=200, help="Minimum rows required per symbol to keep")
    parser.add_argument("--symbols-file", default=None, help="Optional file listing symbols to include (one per line)")
    args = parser.parse_args(argv)

    sources = [Path(s).resolve() for s in args.sources]
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    allowlist: Optional[set] = None
    if args.symbols_file and Path(args.symbols_file).exists():
        allowlist = {line.strip().upper() for line in Path(args.symbols_file).read_text().splitlines() if line.strip()}

    print(f"Scanning sources: {', '.join(str(s) for s in sources)}")
    files = collect_sources(sources)
    print(f"Found {len(files)} CSV files to consider")

    by_symbol: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        sym = infer_symbol_from_name(f)
        if not sym:
            continue
        if allowlist and sym not in allowlist:
            continue
        by_symbol[sym].append(f)

    print(f"Identified {len(by_symbol)} symbols from file names")

    summary_rows = []
    kept = 0
    for sym, paths in sorted(by_symbol.items()):
        dfs = []
        for p in paths:
            df = load_and_standardize(p)
            if df is None:
                continue
            dfs.append(df)
        if not dfs:
            continue
        df_all = pd.concat(dfs, ignore_index=True)
        # Drop duplicates if date exists
        if "date" in df_all.columns:
            df_all = df_all.drop_duplicates(subset=["date"]).sort_values("date")
            if args.since:
                try:
                    cutoff = pd.to_datetime(args.since)
                    df_all = df_all[df_all["date"] >= cutoff]
                except Exception:
                    pass
        # Enforce minimum rows
        if len(df_all) < args.min_rows:
            continue
        # Save
        out_path = out_dir / f"{sym}.csv"
        df_all.to_csv(out_path, index=False)
        kept += 1
        summary_rows.append({
            "symbol": sym,
            "rows": len(df_all),
            "files": len(paths),
            "output": str(out_path)
        })
        print(f"Saved {sym}: {len(df_all)} rows from {len(paths)} files -> {out_path}")

    if summary_rows:
        pd.DataFrame(summary_rows).sort_values("symbol").to_csv(out_dir / "summary.csv", index=False)
        print(f"Summary written to {out_dir / 'summary.csv'}")
    print(f"Done. Consolidated {kept} symbols into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

