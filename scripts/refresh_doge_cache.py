#!/usr/bin/env python3
"""Refresh Binance hourly data + Chronos2 forecast caches for margin meta symbols.

Default pair set covers the active DOGE+AAVE meta bot:
- DOGEUSDT -> DOGEUSD
- AAVEUSDT -> AAVEUSD

The loop runs every few minutes, but forecast rebuilds happen only when a new
closed hourly bar appears for a symbol.
"""
from __future__ import annotations

import argparse
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
DEFAULT_PAIRS = ("DOGEUSDT:DOGEUSD", "AAVEUSDT:AAVEUSD")
LOOKBACK_HOURS = 96
HORIZONS = (1,)
CONTEXT_HOURS = 512
QUANTILES = (0.1, 0.5, 0.9)
BATCH_SIZE = 32
SLEEP_SECONDS = 300
_LAST_FORECAST_TARGET_TS: dict[str, pd.Timestamp] = {}


def _parse_pairs(pair_args: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for raw in pair_args:
        if ":" not in raw:
            raise ValueError(f"Invalid pair '{raw}'. Expected BINANCE:DATA (e.g. DOGEUSDT:DOGEUSD).")
        b_sym, d_sym = raw.split(":", 1)
        b_sym = b_sym.strip().upper()
        d_sym = d_sym.strip().upper()
        if not b_sym or not d_sym:
            raise ValueError(f"Invalid pair '{raw}'.")
        parsed.append((b_sym, d_sym))
    if not parsed:
        raise ValueError("At least one symbol pair is required.")
    return parsed


def _latest_price_timestamp(data_symbol: str) -> pd.Timestamp | None:
    csv_path = DATA_ROOT / f"{data_symbol.upper()}.csv"
    if not csv_path.exists():
        return None
    try:
        existing = pd.read_csv(csv_path, usecols=["timestamp"])
    except Exception:
        return None
    if existing.empty:
        return None
    ts = pd.to_datetime(existing["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        return None
    return ts.max()


def refresh_price_csv(binance_symbol: str, data_symbol: str) -> None:
    from binance_data_wrapper import fetch_binance_hourly_bars
    csv_path = DATA_ROOT / f"{data_symbol.upper()}.csv"
    if not csv_path.exists():
        print(f"[price] {data_symbol}: CSV not found at {csv_path}")
        return
    existing = pd.read_csv(csv_path)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
    last_ts = existing["timestamp"].max()
    end_ts = datetime.now(timezone.utc)
    try:
        new = fetch_binance_hourly_bars(binance_symbol.upper(), start=last_ts, end=end_ts)
    except Exception as e:
        print(f"[price] {data_symbol}: download failed: {e}")
        return
    if new is None or len(new) == 0:
        return
    new = new.reset_index()
    new["symbol"] = data_symbol.upper()
    new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True)
    new = new[new["timestamp"] > last_ts]
    if len(new) == 0:
        return
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.to_csv(csv_path, index=False)
    print(f"[price] {data_symbol}: +{len(new)} rows, last={combined['timestamp'].max()}")


def refresh_forecast_cache(data_symbol: str, *, lookback_hours: int) -> None:
    from binanceneural.forecasts import build_forecast_bundle

    latest_price_ts = _latest_price_timestamp(data_symbol)
    if latest_price_ts is None:
        print(f"[forecast] {data_symbol}: no price timestamp available")
        return
    target_end = latest_price_ts.floor("h")
    prev_target = _LAST_FORECAST_TARGET_TS.get(data_symbol.upper())
    if prev_target is not None and target_end <= prev_target:
        print(f"[forecast] {data_symbol}: up-to-date at {target_end}, skipping")
        return

    start_ts = target_end - pd.Timedelta(hours=float(max(1, int(lookback_hours))))
    result = build_forecast_bundle(
        symbol=data_symbol.upper(),
        data_root=DATA_ROOT,
        cache_root=CACHE_ROOT,
        horizons=HORIZONS,
        context_hours=CONTEXT_HOURS,
        quantile_levels=QUANTILES,
        batch_size=BATCH_SIZE,
        cache_only=False,
        start=start_ts,
        end=target_end,
    )
    _LAST_FORECAST_TARGET_TS[data_symbol.upper()] = target_end
    print(
        f"[forecast] {data_symbol}: {len(result)} rows, "
        f"last={result['timestamp'].max()} target_end={target_end}"
    )


def run_once(pairs: list[tuple[str, str]], *, lookback_hours: int) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n=== Margin cache refresh @ {ts} ===")
    for binance_symbol, data_symbol in pairs:
        try:
            refresh_price_csv(binance_symbol, data_symbol)
        except Exception as e:
            print(f"[price] {data_symbol}: ERROR {e}")
    for _, data_symbol in pairs:
        try:
            refresh_forecast_cache(data_symbol, lookback_hours=lookback_hours)
        except Exception as e:
            print(f"[forecast] {data_symbol}: ERROR {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh Binance meta trading forecast caches.")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=list(DEFAULT_PAIRS),
        help="Pairs in BINANCE:DATA format, e.g. DOGEUSDT:DOGEUSD AAVEUSDT:AAVEUSD",
    )
    parser.add_argument("--sleep-seconds", type=int, default=SLEEP_SECONDS)
    parser.add_argument("--lookback-hours", type=int, default=LOOKBACK_HOURS)
    args = parser.parse_args()

    pairs = _parse_pairs(args.pairs)
    print(
        "Starting margin cache refresh loop "
        f"(sleep={args.sleep_seconds}s, lookback={args.lookback_hours}h, pairs={pairs})"
    )
    while True:
        try:
            run_once(pairs, lookback_hours=args.lookback_hours)
        except Exception as e:
            print(f"[cycle error] {e}")
        print(f"Sleeping {args.sleep_seconds}s...")
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
