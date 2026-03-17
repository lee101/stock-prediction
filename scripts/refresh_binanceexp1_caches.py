#!/usr/bin/env python3
"""Refresh price CSVs + Chronos2 forecast caches for binanceexp1 trading bots."""
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

DATA_ROOT = PROJECT_ROOT / "trainingdatahourly" / "crypto"
CACHE_ROOT = PROJECT_ROOT / "binanceneural" / "forecast_cache"
SYMBOLS = ["SOLUSD", "LINKUSD", "UNIUSD", "BTCUSD", "ETHUSD"]
USDT_FALLBACK = {
    "SOLUSD": "SOLUSDT",
    "LINKUSD": "LINKUSDT",
    "UNIUSD": "UNIUSDT",
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
}
LOOKBACK_HOURS = 48
HORIZONS = (1, 24)
CONTEXT_HOURS = 24 * 14
QUANTILES = (0.1, 0.5, 0.9)
BATCH_SIZE = 128
SLEEP_SECONDS = 300  # 5 min between cycles


def refresh_price_csv(symbol: str) -> None:
    from binance_data_wrapper import fetch_binance_hourly_bars
    csv_path = DATA_ROOT / f"{symbol.upper()}.csv"
    binance_pair = USDT_FALLBACK.get(symbol.upper(), symbol.upper())
    try:
        result = append_hourly_binance_bars(
            csv_path,
            fetch_symbol=binance_pair,
            csv_symbol=symbol,
            fetcher=fetch_binance_hourly_bars,
        )
    except Exception as e:
        print(f"[price] {symbol}: download failed: {e}")
        return
    if result.get("status") in {"updated", "synced"}:
        print(
            f"[price] {symbol}: +{result.get('rows_added', 0)} rows, "
            f"last={result.get('end')}"
        )


def refresh_forecast_cache(symbol: str) -> None:
    from binanceneural.forecasts import ChronosForecastManager, ForecastConfig

    end_ts = pd.Timestamp(datetime.now(timezone.utc))
    start_ts = end_ts - pd.Timedelta(hours=LOOKBACK_HOURS)

    for horizon in HORIZONS:
        cfg = ForecastConfig(
            symbol=symbol,
            data_root=DATA_ROOT,
            context_hours=CONTEXT_HOURS,
            prediction_horizon_hours=int(horizon),
            quantile_levels=QUANTILES,
            batch_size=BATCH_SIZE,
            cache_dir=CACHE_ROOT / f"h{int(horizon)}",
        )
        mgr = ChronosForecastManager(cfg)
        result = mgr.ensure_latest(start=start_ts, end=end_ts, cache_only=False)
        print(f"[forecast] {symbol} h{horizon}: {len(result)} rows, last={result['timestamp'].max()}")


def run_once() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n=== Cache refresh cycle @ {ts} ===")
    for sym in SYMBOLS:
        try:
            refresh_price_csv(sym)
        except Exception as e:
            print(f"[price] {sym}: ERROR {e}")
    for sym in SYMBOLS:
        try:
            refresh_forecast_cache(sym)
        except Exception as e:
            print(f"[forecast] {sym}: ERROR {e}")


def main() -> int:
    print(f"Starting cache refresh loop (sleep={SLEEP_SECONDS}s)")
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[cycle error] {e}")
        print(f"Sleeping {SLEEP_SECONDS}s...")
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
