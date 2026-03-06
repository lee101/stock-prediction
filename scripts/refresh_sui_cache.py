#!/usr/bin/env python3
"""Refresh price CSV + Chronos2 forecast cache for SUI trading bot."""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.binance_hourly_csv_utils import append_hourly_binance_bars

DATA_ROOT = PROJECT_ROOT / "binance_spot_hourly"
CACHE_ROOT = PROJECT_ROOT / "binancechronossolexperiment" / "forecast_cache_sui_10bp"
SYMBOL = "SUIUSDT"
MODEL_ID = "chronos2_finetuned/binance_lora_20260208_newpairs_SUIUSDT/finetuned-ckpt"
LOOKBACK_HOURS = 48
HORIZONS = (1, 4, 24)
CONTEXT_HOURS = 512
QUANTILES = (0.1, 0.5, 0.9)
BATCH_SIZE = 32
SLEEP_SECONDS = 300


def refresh_price_csv() -> None:
    from binance_data_wrapper import fetch_binance_hourly_bars
    csv_path = DATA_ROOT / f"{SYMBOL}.csv"
    try:
        result = append_hourly_binance_bars(
            csv_path,
            fetch_symbol=SYMBOL,
            csv_symbol=SYMBOL,
            fetcher=fetch_binance_hourly_bars,
        )
    except Exception as e:
        print(f"[price] {SYMBOL}: download failed: {e}")
        return
    if result.get("status") in {"updated", "synced"}:
        print(
            f"[price] {SYMBOL}: +{result.get('rows_added', 0)} rows, "
            f"last={result.get('end')}"
        )


def refresh_forecast_cache() -> None:
    from binancechronossolexperiment.forecasts import build_forecast_bundle
    end_ts = pd.Timestamp(datetime.now(timezone.utc))
    start_ts = end_ts - pd.Timedelta(hours=LOOKBACK_HOURS)
    result = build_forecast_bundle(
        symbol=SYMBOL,
        data_root=DATA_ROOT,
        cache_root=CACHE_ROOT,
        horizons=HORIZONS,
        context_hours=CONTEXT_HOURS,
        quantile_levels=QUANTILES,
        batch_size=BATCH_SIZE,
        model_id=MODEL_ID,
        cache_only=False,
        start=start_ts,
        end=end_ts,
    )
    print(f"[forecast] {SYMBOL}: {len(result)} rows, last={result['timestamp'].max()}")


def run_once() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n=== SUI cache refresh @ {ts} ===")
    try:
        refresh_price_csv()
    except Exception as e:
        print(f"[price] ERROR: {e}")
    try:
        refresh_forecast_cache()
    except Exception as e:
        print(f"[forecast] ERROR: {e}")


def main() -> int:
    print(f"Starting SUI cache refresh loop (sleep={SLEEP_SECONDS}s)")
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[cycle error] {e}")
        print(f"Sleeping {SLEEP_SECONDS}s...")
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
