#!/usr/bin/env python3
"""
Extend stocks12 training data back to PLTR IPO (2020-09-30) using yfinance.
Also creates stocks11 (without PLTR) going back to 2019.
Downloads split-adjusted OHLCV from yfinance, formats to match existing CSV format,
then re-exports MKTD binaries.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
import subprocess
import sys

STOCKS12 = ['AAPL','MSFT','NVDA','GOOG','META','TSLA','SPY','QQQ','PLTR','JPM','V','AMZN']
STOCKS11 = ['AAPL','MSFT','NVDA','GOOG','META','TSLA','SPY','QQQ','JPM','V','AMZN']

DATA_DIR = Path('trainingdata/train')
BACKUP_DIR = Path('trainingdata/train_backup_pre_extend')

# Date ranges
PLTR_START = '2020-09-30'       # PLTR IPO date
STOCKS11_START = '2019-01-02'   # Pre-PLTR extended
TRAIN_END = '2025-08-31'
VAL_START = '2025-09-01'
VAL_END = '2026-02-28'
DOWNLOAD_START = '2019-01-01'
DOWNLOAD_END = '2026-03-22'


def download_yfinance(sym: str) -> pd.DataFrame:
    """Download split-adjusted OHLCV from yfinance."""
    df = yf.download(sym, start=DOWNLOAD_START, end=DOWNLOAD_END,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {sym}")
    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index.name = 'timestamp'
    df = df.reset_index()
    # Format timestamp to match existing format
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['symbol'] = sym
    # Select needed columns, add zeros for missing ones
    if 'adj close' in df.columns:
        df['close'] = df['adj close']
    needed = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    for col in needed:
        if col not in df.columns:
            df[col] = 0.0
    df = df[needed + ['symbol']]
    df['trade_count'] = 0
    df['vwap'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d 05:00:00+00:00')
    return df[['timestamp','open','high','low','close','volume','trade_count','vwap','symbol']]


def verify_no_big_drop(df: pd.DataFrame, sym: str, threshold=0.35):
    """Warn about any single-day drops > threshold after split adjustment."""
    closes = df.sort_values('timestamp')['close'].values
    issues = []
    for i in range(1, len(closes)):
        if closes[i-1] > 0:
            pct = (closes[i] - closes[i-1]) / closes[i-1]
            if pct < -threshold:
                dt = df.sort_values('timestamp')['timestamp'].iloc[i]
                issues.append(f"  {sym}: {dt[:10]} drop {pct*100:.1f}%")
    if issues:
        print(f"  [WARN] {sym} has {len(issues)} big drop(s) >35%:")
        for s in issues[:5]:
            print(s)
    else:
        print(f"  {sym}: clean (no drops >35%)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-export', action='store_true', help='Skip binary export')
    args = ap.parse_args()

    BACKUP_DIR.mkdir(exist_ok=True)

    print("=== Downloading extended stocks12 data from yfinance ===")
    frames = {}
    for sym in STOCKS12:
        print(f"  Downloading {sym}...")
        try:
            df = download_yfinance(sym)
            frames[sym] = df
            print(f"  {sym}: {len(df)} rows ({df.timestamp.iloc[0][:10]} to {df.timestamp.iloc[-1][:10]})")
        except Exception as e:
            print(f"  {sym}: ERROR - {e}")
            sys.exit(1)

    print("\n=== Verifying split adjustment ===")
    for sym, df in frames.items():
        verify_no_big_drop(df, sym)

    if args.dry_run:
        print("\n[dry-run] would write CSVs and export binaries")
        return

    print("\n=== Saving extended CSVs ===")
    for sym, df in frames.items():
        existing = DATA_DIR / f'{sym}.csv'
        backup = BACKUP_DIR / f'{sym}.csv'
        if existing.exists() and not backup.exists():
            import shutil
            shutil.copy2(existing, backup)
            print(f"  Backed up {sym}.csv")
        df.to_csv(existing, index=False)
        print(f"  Wrote {len(df)} rows to {existing}")

    if args.no_export:
        print("Skipping binary export.")
        return

    print("\n=== Exporting MKTD binaries ===")
    venv = Path('.venv313/bin/python')
    syms12 = ','.join(STOCKS12)
    syms11 = ','.join(STOCKS11)

    exports = [
        # stocks12 extended train (PLTR limits to 2020-09-30)
        (syms12, PLTR_START, TRAIN_END,
         'pufferlib_market/data/stocks12_extended_train.bin',
         'stocks12_extended_daily_train'),
        # stocks12 extended val (same val period)
        (syms12, VAL_START, VAL_END,
         'pufferlib_market/data/stocks12_extended_val.bin',
         'stocks12_extended_daily_val'),
        # stocks11 extended train (goes back to 2019)
        (syms11, STOCKS11_START, TRAIN_END,
         'pufferlib_market/data/stocks11_train.bin',
         'stocks11_daily_train'),
        # stocks11 extended val
        (syms11, VAL_START, VAL_END,
         'pufferlib_market/data/stocks11_val.bin',
         'stocks11_daily_val'),
    ]

    for syms, start, end, out, label in exports:
        cmd = [
            str(venv), '-m', 'pufferlib_market.export_data_daily',
            '--symbols', syms,
            '--data-root', 'trainingdata/train',
            '--output', out,
            '--start-date', start,
            '--end-date', end,
            '--min-days', '100',
        ]
        print(f"\n  Exporting {label}...")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"  ERROR: export failed for {label}")
            sys.exit(1)

    print("\nDone. New binaries:")
    for f in ['stocks12_extended_train.bin', 'stocks12_extended_val.bin',
              'stocks11_train.bin', 'stocks11_val.bin']:
        p = Path(f'pufferlib_market/data/{f}')
        if p.exists():
            print(f"  {p}: {p.stat().st_size // 1024 // 1024} MB")


if __name__ == '__main__':
    main()
