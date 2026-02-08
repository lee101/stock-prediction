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

DATA_ROOT = PROJECT_ROOT / "trainingdatahourly" / "crypto"
CACHE_ROOT = PROJECT_ROOT / "binanceneural" / "forecast_cache"
SYMBOLS = ["SOLUSD", "BTCUSD", "ETHUSD"]
USDT_FALLBACK = {"SOLUSD": "SOLUSDT", "BTCUSD": "BTCUSDT", "ETHUSD": "ETHUSDT"}
LOOKBACK_HOURS = 48
HORIZONS = (1, 24)
CONTEXT_HOURS = 24 * 14
QUANTILES = (0.1, 0.5, 0.9)
BATCH_SIZE = 128
SLEEP_SECONDS = 300  # 5 min between cycles


def refresh_price_csv(symbol: str) -> None:
    from binance_data_wrapper import fetch_binance_hourly_bars
    csv_path = DATA_ROOT / f"{symbol.upper()}.csv"
    if not csv_path.exists():
        print(f"[price] {symbol}: CSV not found at {csv_path}")
        return
    existing = pd.read_csv(csv_path)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    last_ts = existing["timestamp"].max()
    binance_pair = USDT_FALLBACK.get(symbol.upper(), symbol.upper())
    end_ts = datetime.now(timezone.utc)
    try:
        new = fetch_binance_hourly_bars(binance_pair, start=last_ts, end=end_ts)
    except Exception as e:
        print(f"[price] {symbol}: download failed: {e}")
        return
    if new is None or len(new) == 0:
        return
    new = new.reset_index()
    new["symbol"] = symbol.upper()
    new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True)
    new = new[new["timestamp"] > last_ts]
    if len(new) == 0:
        return
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.to_csv(csv_path, index=False)
    print(f"[price] {symbol}: +{len(new)} rows, last={combined['timestamp'].max()}")


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
