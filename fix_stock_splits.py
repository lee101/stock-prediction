#!/usr/bin/env python3
"""Fix unadjusted stock splits in training/val CSV data.

Splits to fix (divide pre-split prices, multiply pre-split volume):
  NVDA: 10:1 forward split on 2024-06-10
  GOOG: 20:1 forward split on 2022-07-18
  TSLA:  3:1 forward split on 2022-08-25
  AMZN: 20:1 forward split on 2022-06-06
  AVGO: 10:1 forward split on 2024-07-15

NFLX: already fixed (10:1, 2025-11-15).
"""

import shutil
import pandas as pd
from pathlib import Path

# (symbol, split_date_exclusive, split_factor)
SPLITS = [
    ("NVDA", "2024-06-10", 10.0),
    ("GOOG", "2022-07-18", 20.0),
    ("TSLA", "2022-08-25",  3.0),
    ("AMZN", "2022-06-06", 20.0),
    ("AVGO", "2024-07-15", 10.0),
]

PRICE_COLS = ["open", "high", "low", "close"]
VOL_COL    = "volume"
DATE_COL   = "timestamp"


def adjust_csv(csv_path: Path, split_date: str, factor: float, dry_run: bool = False):
    df = pd.read_csv(csv_path)
    # Parse timestamp column (may have timezone info)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True).dt.normalize()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    split_ts = pd.Timestamp(split_date, tz="UTC")
    mask = df[DATE_COL] < split_ts
    n = mask.sum()
    if n == 0:
        print(f"  [{csv_path.name}] No rows before {split_date} — skipping.")
        return

    # Verify split ratio looks real
    idx_after = df[df[DATE_COL] >= split_ts].index
    if len(idx_after) > 0 and idx_after[0] > 0:
        first_after = idx_after[0]
        pre_close  = df.loc[first_after - 1, "close"]
        post_close = df.loc[first_after,     "close"]
        ratio = post_close / pre_close if pre_close > 0 else 999
        expected = 1.0 / factor
        if abs(ratio - expected) > 0.15:
            print(f"  [{csv_path.name}] WARNING: ratio {ratio:.3f} vs expected 1/{factor:.0f}={expected:.3f} "
                  f"— verify manually!")
        else:
            print(f"  [{csv_path.name}] Split ratio {ratio:.4f} ≈ 1/{factor:.0f} ✓  "
                  f"({n} pre-split rows to adjust)")
    else:
        print(f"  [{csv_path.name}] {n} pre-split rows to adjust (no post-split row to verify)")

    if dry_run:
        print(f"  [dry-run] would adjust {n} rows")
        return

    # Backup original (read raw bytes to preserve format)
    backup = csv_path.with_suffix(".csv.pre_split_backup")
    if not backup.exists():
        shutil.copy2(csv_path, backup)
        print(f"  Backed up to {backup.name}")
    else:
        print(f"  Backup already exists: {backup.name}")

    # Re-read raw (don't parse dates) so we preserve original timestamp strings
    df_raw = pd.read_csv(csv_path)
    df_raw_sorted = df_raw.copy()
    ts_parsed = pd.to_datetime(df_raw_sorted[DATE_COL], utc=True).dt.normalize()
    raw_mask = ts_parsed < split_ts

    for col in PRICE_COLS:
        if col in df_raw_sorted.columns:
            df_raw_sorted.loc[raw_mask, col] = (df_raw_sorted.loc[raw_mask, col] / factor)
    if VOL_COL in df_raw_sorted.columns:
        df_raw_sorted.loc[raw_mask, VOL_COL] = (df_raw_sorted.loc[raw_mask, VOL_COL] * factor)

    df_raw_sorted.to_csv(csv_path, index=False)
    print(f"  Wrote {len(df_raw_sorted)} rows to {csv_path}")


def verify_no_big_drop(csv_path: Path, threshold: float = 0.30):
    """Print any single-day drop exceeding threshold."""
    try:
        df = pd.read_csv(csv_path)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True).dt.normalize()
        df = df.sort_values(DATE_COL).reset_index(drop=True)
        closes = df["close"].values
        for i in range(1, len(closes)):
            pct = (closes[i] - closes[i - 1]) / closes[i - 1]
            if pct < -threshold:
                print(f"  [{csv_path.stem}] BIG DROP row {i}: "
                      f"{df.loc[i-1, DATE_COL].date()} {closes[i-1]:.2f} → "
                      f"{df.loc[i, DATE_COL].date()} {closes[i]:.2f}  ({pct*100:.1f}%)")
    except Exception as e:
        print(f"  [{csv_path.name}] error: {e}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    base_daily  = Path("trainingdata/train")
    base_hourly = Path("trainingdatahourly/stocks")

    all_syms = [
        "AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA", "AMZN", "AMD",
        "JPM", "SPY", "QQQ", "PLTR", "NET", "NFLX", "ADBE", "CRM",
        "AVGO", "V", "COST", "ADSK",
    ]

    if args.verify_only:
        print("=== Checking for remaining big drops (>30%) in daily CSVs ===")
        for sym in all_syms:
            p = base_daily / f"{sym}.csv"
            if p.exists():
                verify_no_big_drop(p)
        print("Done.")
    else:
        for sym, split_date, factor in SPLITS:
            print(f"\n=== {sym} ({factor:.0f}:1 on {split_date}) ===")
            daily = base_daily / f"{sym}.csv"
            if daily.exists():
                adjust_csv(daily, split_date, factor, dry_run=args.dry_run)
            else:
                print(f"  Not found: {daily}")

            hourly = base_hourly / f"{sym}.csv"
            if hourly.exists():
                adjust_csv(hourly, split_date, factor, dry_run=args.dry_run)
            else:
                print(f"  Not found (hourly): {hourly}")

        if not args.dry_run:
            print("\n=== Post-fix verification (>30% drops) ===")
            for sym in all_syms:
                p = base_daily / f"{sym}.csv"
                if p.exists():
                    verify_no_big_drop(p)
        print("\nDone.")
