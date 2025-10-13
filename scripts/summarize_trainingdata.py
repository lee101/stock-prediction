#!/usr/bin/env python3
"""
Summarize available CSV data across one or more directories.

Default directories checked:
  - trainingdata/
  - hftraining/trainingdata/
  - externaldata/yahoo/

Outputs per-file rows and date ranges, plus a compact per-symbol summary.
"""

import argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict


def summarize_dirs(dirs: list[str]) -> None:
    entries = []
    for d in dirs:
        base = Path(d)
        if not base.exists():
            continue
        for p in base.rglob('*.csv'):
            try:
                df = pd.read_csv(p, nrows=5)
                cols = [c.lower() for c in df.columns]
                # Try to find a date column by common names
                date_col = None
                for cand in ['date', 'datetime', 'timestamp']:
                    if cand in cols:
                        date_col = cand
                        break
                # Re-read only necessary columns to avoid huge memory when summarizing
                if date_col:
                    df2 = pd.read_csv(p, usecols=[date_col])
                    df2[date_col] = pd.to_datetime(df2[date_col], errors='coerce')
                    n = len(df2)
                    dt_min = df2[date_col].min()
                    dt_max = df2[date_col].max()
                else:
                    df2 = pd.read_csv(p)
                    n = len(df2)
                    dt_min = None
                    dt_max = None
                entries.append((p, n, dt_min, dt_max))
            except Exception:
                continue

    # Print per-file summary
    print('Files:')
    for p, n, dt_min, dt_max in sorted(entries, key=lambda x: str(x[0])):
        if dt_min is not None:
            print(f"- {p}  rows={n}  range=[{dt_min.date()}..{dt_max.date()}]")
        else:
            print(f"- {p}  rows={n}")

    # Per-symbol summary (based on filename stem)
    by_symbol = defaultdict(list)
    for p, n, dt_min, dt_max in entries:
        sym = p.stem.upper()
        by_symbol[sym].append((n, dt_min, dt_max, p))

    print('\nPer-symbol summary:')
    for sym in sorted(by_symbol.keys()):
        items = by_symbol[sym]
        total_rows = sum(x[0] for x in items)
        all_min = min((x[1] for x in items if x[1] is not None), default=None)
        all_max = max((x[2] for x in items if x[2] is not None), default=None)
        span = f"[{all_min.date()}..{all_max.date()}]" if (all_min and all_max) else "[no-dates]"
        print(f"- {sym}: total_rows={total_rows}  span={span}  files={len(items)}")


def main():
    ap = argparse.ArgumentParser(description='Summarize CSV data directories')
    ap.add_argument('--dirs', nargs='*', default=['trainingdata', 'hftraining/trainingdata', 'externaldata/yahoo'],
                    help='Directories to scan (recursive)')
    args = ap.parse_args()
    summarize_dirs(args.dirs)


if __name__ == '__main__':
    main()

