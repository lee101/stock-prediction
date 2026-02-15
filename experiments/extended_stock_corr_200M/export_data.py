#!/usr/bin/env python3
"""Export training data for extended stock universe (30+ symbols).

Usage:
    python -m experiments.extended_stock_corr_200M.export_data \
        --output pufferlib_market/data/extended_stocks.bin \
        --forecast-cache alpacanewccrosslearning/forecast_cache/mega24_novol_baseline_20260206_0038_lb2400
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.extended_stock_corr_200M.config import ALL_SYMBOLS, LONG_ONLY_STOCKS, SHORT_ONLY_STOCKS
from pufferlib_market.export_data import (
    load_price_data,
    load_forecast,
    compute_features,
    FEATURES_PER_SYM,
    PRICE_FEATURES,
    MAGIC,
    VERSION,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="trainingdata")
    parser.add_argument("--forecast-cache", type=str, required=True)
    parser.add_argument("--lookback-hours", type=int, default=4000)
    parser.add_argument("--min-hours", type=int, default=500)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    cache_root = Path(args.forecast_cache)
    output_path = Path(args.output)

    print(f"Exporting {len(ALL_SYMBOLS)} symbols to {output_path}")
    print(f"  Long-only: {len(LONG_ONLY_STOCKS)}")
    print(f"  Short-only: {len(SHORT_ONLY_STOCKS)}")
    print(f"  Crypto: {len(ALL_SYMBOLS) - len(LONG_ONLY_STOCKS) - len(SHORT_ONLY_STOCKS)}")

    # Load all data
    all_features = []
    all_prices = []
    valid_symbols = []
    common_index = None

    for sym in ALL_SYMBOLS:
        try:
            price_df = load_price_data(sym, data_root)
            fc_h1 = load_forecast(sym, cache_root, 1)
            fc_h24 = load_forecast(sym, cache_root, 24)

            # Compute features
            feat_df = compute_features(price_df, fc_h1, fc_h24)

            # Align to hourly grid
            feat_df = feat_df.resample("1h").last().dropna()

            if len(feat_df) < args.min_hours:
                print(f"  {sym}: skipped (only {len(feat_df)} hours)")
                continue

            # Trim to lookback
            if len(feat_df) > args.lookback_hours:
                feat_df = feat_df.iloc[-args.lookback_hours:]

            all_features.append((sym, feat_df))

            # Price data
            price_cols = ["open", "high", "low", "close", "volume"]
            price_df = price_df[price_cols].reindex(feat_df.index).ffill()
            all_prices.append((sym, price_df))

            valid_symbols.append(sym)

            # Track common index
            if common_index is None:
                common_index = feat_df.index
            else:
                common_index = common_index.intersection(feat_df.index)

            print(f"  {sym}: {len(feat_df)} hours")

        except Exception as e:
            print(f"  {sym}: ERROR - {e}")
            continue

    if not valid_symbols:
        print("No valid symbols found!")
        return

    print(f"\nCommon timestamps: {len(common_index)}")
    print(f"Valid symbols: {len(valid_symbols)}")

    # Align all to common index
    T = len(common_index)
    S = len(valid_symbols)

    features = np.zeros((T, S, FEATURES_PER_SYM), dtype=np.float32)
    prices = np.zeros((T, S, PRICE_FEATURES), dtype=np.float32)

    for i, (sym, feat_df) in enumerate(all_features):
        aligned = feat_df.reindex(common_index).ffill().bfill()
        features[:, i, :] = aligned.values[:, :FEATURES_PER_SYM].astype(np.float32)

    for i, (sym, price_df) in enumerate(all_prices):
        aligned = price_df.reindex(common_index).ffill().bfill()
        prices[:, i, :] = aligned.values.astype(np.float32)

    # Replace NaN/inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)

    # Write binary
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        # Header (64 bytes)
        header = struct.pack(
            "4sIIIII40s",
            MAGIC,
            VERSION,
            S,
            T,
            FEATURES_PER_SYM,
            PRICE_FEATURES,
            b"\x00" * 40,
        )
        f.write(header)

        # Symbol names (16 bytes each)
        for sym in valid_symbols:
            name = sym.encode("ascii")[:15].ljust(16, b"\x00")
            f.write(name)

        # Feature data
        f.write(features.tobytes())

        # Price data
        f.write(prices.tobytes())

    file_size = output_path.stat().st_size
    print(f"\nWrote {output_path} ({file_size / 1e6:.1f} MB)")
    print(f"  {S} symbols x {T} timesteps")
    print(f"  Features: {FEATURES_PER_SYM} per symbol")


if __name__ == "__main__":
    main()
