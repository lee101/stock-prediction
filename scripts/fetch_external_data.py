#!/usr/bin/env python3
"""
Fetch supplemental market data into a separate directory.

This script intentionally keeps external data separate from the main
`trainingdata/` folder. By default, it writes to `externaldata/yahoo/`.

Example:
  python scripts/fetch_external_data.py --symbols AAPL MSFT --start 2015-01-01 --end 2024-12-31
  python scripts/fetch_external_data.py --symbols-file symbols.txt --out externaldata/yahoo
"""

import argparse
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd

try:
    import yfinance as yf  # Optional; may not be installed/available
except Exception:
    yf = None


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Lowercase columns and standardize names
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    # Align to common schema where possible
    rename = {
        'adj_close': 'adj_close',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
    }
    for k, v in list(rename.items()):
        if k in df.columns:
            df[v] = df[k]
    # Reset index if it's a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'date', 'datetime': 'date'})
    # Ensure date column exists and is sorted
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        except Exception:
            pass
    return df


def fetch_yahoo(symbols, start: str = '2015-01-01', end: str | None = None, out_dir: Path = Path('externaldata/yahoo')) -> int:
    if yf is None:
        print('yfinance not available (or blocked). Skipping download.')
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    end = end or datetime.now().strftime('%Y-%m-%d')
    success = 0
    for sym in symbols:
        sym = sym.strip()
        if not sym:
            continue
        print(f"Downloading {sym} [{start} -> {end}] ...")
        try:
            df = yf.download(sym, start=start, end=end, progress=False)
            if df is None or len(df) == 0:
                print(f"  No data for {sym}")
                continue
            df = _standardize_df(df)
            out_file = out_dir / f"{sym.upper()}.csv"
            df.to_csv(out_file, index=False)
            print(f"  Saved {len(df)} rows -> {out_file}")
            success += 1
        except Exception as e:
            print(f"  Error fetching {sym}: {e}")
    return 0 if success > 0 else 2


def main():
    ap = argparse.ArgumentParser(description='Fetch supplemental market data into externaldata/')
    ap.add_argument('--symbols', nargs='*', default=None, help='Symbols to download')
    ap.add_argument('--symbols-file', type=str, default=None, help='Path to a file with one symbol per line')
    ap.add_argument('--start', type=str, default='2015-01-01', help='Start date (YYYY-MM-DD)')
    ap.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    ap.add_argument('--out', type=str, default='externaldata/yahoo', help='Output directory')
    args = ap.parse_args()

    symbols = args.symbols or []
    if args.symbols_file:
        p = Path(args.symbols_file)
        if p.exists():
            symbols += [line.strip() for line in p.read_text().splitlines() if line.strip()]
    if not symbols:
        print('No symbols provided. Use --symbols or --symbols-file.')
        sys.exit(2)

    rc = fetch_yahoo(symbols, start=args.start, end=args.end, out_dir=Path(args.out))
    sys.exit(rc)


if __name__ == '__main__':
    main()

