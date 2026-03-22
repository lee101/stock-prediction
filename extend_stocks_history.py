#!/usr/bin/env python3
"""Download 2015-2018 historical data for stocks11 (no PLTR) and create extended training bins.

Adds ~4 years of pre-existing-CSV history including:
- 2015: China crash, oil crash, corrections
- 2016: Brexit, Trump election rally
- 2017-2018: Bull run + Q4 2018 correction (-20%)
- 2019 data is already in trainingdata/train/

This extends stocks12 training from 2020-09-30 (1797 days) to 2015-01-02 (~2600 days).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

STOCKS11 = ["AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA", "SPY", "QQQ", "JPM", "V", "AMZN"]
STOCKS12 = STOCKS11 + ["PLTR"]

EXTENDED_DATA_DIR = REPO / "trainingdata" / "train_extended"
EXISTING_DATA_DIR = REPO / "trainingdata" / "train"

# yfinance starts at 2012-05-18 for META (the limiting factor), 2015 is safe for all
EXTENSION_START = "2015-01-01"
EXTENSION_END = "2019-01-02"  # exclusive; existing CSVs start at 2019-01-02


def download_and_extend(symbol: str, extension_start: str, extension_end: str) -> pd.DataFrame:
    """Download older yfinance data and combine with existing CSV."""
    existing_path = EXISTING_DATA_DIR / f"{symbol}.csv"
    
    # Download pre-2019 data from yfinance
    print(f"  Downloading {symbol} {extension_start} -> {extension_end}...")
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=extension_start, end=extension_end, auto_adjust=True)
    if len(hist) == 0:
        print(f"  WARNING: No yfinance data for {symbol} in {extension_start}:{extension_end}")
        return None

    # Convert to our CSV format
    hist.index = hist.index.tz_convert("UTC") if hist.index.tz else hist.index.tz_localize("UTC")
    hist.index.name = "timestamp"
    yf_df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    yf_df.columns = ["open", "high", "low", "close", "volume"]
    yf_df["trade_count"] = 0
    yf_df["vwap"] = (yf_df["high"] + yf_df["low"] + yf_df["close"]) / 3.0
    yf_df["symbol"] = symbol
    yf_df = yf_df.reset_index()
    
    # Scale yfinance prices to match Alpaca at junction (2019-01-02)
    # This ensures price continuity at the seam
    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)
        existing_df.columns = [c.lower() for c in existing_df.columns]
        existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], utc=True)
        # Use first available existing date as anchor
        first_existing = existing_df.sort_values("timestamp").iloc[0]
        # Find same date in yfinance data
        first_date = first_existing["timestamp"].date()
        yf_mask = pd.to_datetime(yf_df["timestamp"]).dt.date == first_date
        if yf_mask.sum() > 0:
            yf_close_at_junction = float(yf_df.loc[yf_mask, "close"].iloc[0])
            alpaca_close_at_junction = float(first_existing["close"])
            if yf_close_at_junction > 0 and abs(yf_close_at_junction - alpaca_close_at_junction) / alpaca_close_at_junction > 0.001:
                scale = alpaca_close_at_junction / yf_close_at_junction
                print(f"  Scaling {symbol} yfinance by {scale:.6f} (yf={yf_close_at_junction:.2f} alpaca={alpaca_close_at_junction:.2f})")
                for col in ["open", "high", "low", "close", "vwap"]:
                    yf_df[col] *= scale
        
        # Remove any overlap with existing data (yfinance might have data on 2019-01-02)
        yf_df["_date"] = pd.to_datetime(yf_df["timestamp"]).dt.date
        existing_df["_date"] = pd.to_datetime(existing_df["timestamp"]).dt.date
        existing_dates = set(existing_df["_date"])
        yf_df = yf_df[~yf_df["_date"].isin(existing_dates)].drop("_date", axis=1)
        existing_df = existing_df.drop("_date", axis=1)
        
        # Combine: yfinance 2015-2018 + existing 2019+
        combined = pd.concat([yf_df, existing_df], ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    else:
        combined = yf_df
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    
    print(f"  {symbol}: {len(combined)} rows total ({combined['timestamp'].min().date()} to {combined['timestamp'].max().date()})")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend stocks training data with pre-2019 history")
    parser.add_argument("--symbols", default=",".join(STOCKS11))
    parser.add_argument("--start", default=EXTENSION_START)
    parser.add_argument("--end", default=EXTENSION_END)
    parser.add_argument("--out-dir", default=str(EXTENDED_DATA_DIR))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--export-bin", action="store_true", help="Also export MKTD bins")
    parser.add_argument("--train-end", default="2025-08-31", help="End of training period")
    parser.add_argument("--val-start", default="2025-09-01")
    parser.add_argument("--val-end", default="2026-03-20")
    parser.add_argument("--bin-suffix", default="2015", help="Suffix for output bin names (e.g. '2012' → stocks11_daily_train_2012.bin)")
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_download:
        print(f"Downloading {len(symbols)} symbols from {args.start} to {args.end}...")
        for sym in symbols:
            df = download_and_extend(sym, args.start, args.end)
            if df is not None:
                out_path = out_dir / f"{sym}.csv"
                df.to_csv(out_path, index=False)
                print(f"  -> {out_path}")
    
    if args.export_bin:
        sym_str = ",".join(symbols)
        # Export training bin
        train_bin = f"pufferlib_market/data/stocks{len(symbols)}_daily_train_{args.bin_suffix}.bin"
        print(f"\nExporting training bin: {train_bin}")
        cmd = [
            "python3", "-m", "pufferlib_market.export_data_daily",
            "--symbols", sym_str,
            "--data-root", str(out_dir),
            "--output", train_bin,
            "--end-date", args.train_end,
        ]
        subprocess.run(cmd, check=True, cwd=str(REPO))
        
        # Export val bin
        val_bin = f"pufferlib_market/data/stocks{len(symbols)}_daily_val_{args.bin_suffix}.bin"
        print(f"Exporting val bin: {val_bin}")
        cmd = [
            "python3", "-m", "pufferlib_market.export_data_daily",
            "--symbols", sym_str,
            "--data-root", str(out_dir),
            "--output", val_bin,
            "--start-date", args.val_start,
            "--end-date", args.val_end,
        ]
        subprocess.run(cmd, check=True, cwd=str(REPO))
        
        # Check sizes
        import struct
        for binpath in [train_bin, val_bin]:
            with open(REPO / binpath, 'rb') as f:
                f.read(4)  # magic
                f.read(4)  # version
                n_sym = struct.unpack('<I', f.read(4))[0]
                n_steps = struct.unpack('<I', f.read(4))[0]
            print(f"  {binpath}: {n_sym} symbols × {n_steps} days = {n_sym*n_steps} samples")


if __name__ == "__main__":
    main()
