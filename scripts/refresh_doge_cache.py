#!/usr/bin/env python3
"""Refresh price CSV + Chronos2 forecast cache for DOGE trading bot."""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "trainingdatahourlybinance"
CACHE_ROOT = PROJECT_ROOT / "binanceneural" / "forecast_cache"
BINANCE_SYMBOL = "DOGEUSDT"
DATA_SYMBOL = "DOGEUSD"
LOOKBACK_HOURS = 48
HORIZONS = (1,)
CONTEXT_HOURS = 512
QUANTILES = (0.1, 0.5, 0.9)
BATCH_SIZE = 32
SLEEP_SECONDS = 300


def refresh_price_csv() -> None:
    from binance_data_wrapper import fetch_binance_hourly_bars
    csv_path = DATA_ROOT / f"{DATA_SYMBOL}.csv"
    if not csv_path.exists():
        print(f"[price] {DATA_SYMBOL}: CSV not found at {csv_path}")
        return
    existing = pd.read_csv(csv_path)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    last_ts = existing["timestamp"].max()
    end_ts = datetime.now(timezone.utc)
    try:
        new = fetch_binance_hourly_bars(BINANCE_SYMBOL, start=last_ts, end=end_ts)
    except Exception as e:
        print(f"[price] {DATA_SYMBOL}: download failed: {e}")
        return
    if new is None or len(new) == 0:
        return
    new = new.reset_index()
    new["symbol"] = DATA_SYMBOL
    new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True)
    new = new[new["timestamp"] > last_ts]
    if len(new) == 0:
        return
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.to_csv(csv_path, index=False)
    print(f"[price] {DATA_SYMBOL}: +{len(new)} rows, last={combined['timestamp'].max()}")


def refresh_forecast_cache() -> None:
    from binanceneural.forecasts import build_forecast_bundle
    end_ts = pd.Timestamp(datetime.now(timezone.utc))
    start_ts = end_ts - pd.Timedelta(hours=LOOKBACK_HOURS)
    result = build_forecast_bundle(
        symbol=DATA_SYMBOL,
        data_root=DATA_ROOT,
        cache_root=CACHE_ROOT,
        horizons=HORIZONS,
        context_hours=CONTEXT_HOURS,
        quantile_levels=QUANTILES,
        batch_size=BATCH_SIZE,
        cache_only=False,
        start=start_ts,
        end=end_ts,
    )
    print(f"[forecast] {DATA_SYMBOL}: {len(result)} rows, last={result['timestamp'].max()}")


def run_once() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n=== DOGE cache refresh @ {ts} ===")
    try:
        refresh_price_csv()
    except Exception as e:
        print(f"[price] ERROR: {e}")
    try:
        refresh_forecast_cache()
    except Exception as e:
        print(f"[forecast] ERROR: {e}")


def main() -> int:
    print(f"Starting DOGE cache refresh loop (sleep={SLEEP_SECONDS}s)")
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[cycle error] {e}")
        print(f"Sleeping {SLEEP_SECONDS}s...")
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
