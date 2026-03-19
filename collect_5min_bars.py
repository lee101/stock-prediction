#!/usr/bin/env python3
"""Continuous 5-minute bar collector for Binance margin-traded symbols.

Runs as a supervisor daemon. Every 5 minutes, fetches the latest 5m kline
from Binance REST API and appends to trainingdata5min/{SYMBOL}.csv.

Also does a daily Vision backfill to catch any gaps.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
from datetime import datetime, timezone, timedelta
import pandas as pd
from loguru import logger

from src.binan.binance_wrapper import _resolve_client

REPO = Path(__file__).resolve().parents[1]
COLS = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap", "symbol"]


def fetch_recent_5m(symbol: str, limit: int = 3) -> pd.DataFrame:
    client = _resolve_client()
    raw = client.get_klines(symbol=symbol, interval="5m", limit=limit)
    rows = []
    for k in raw:
        o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
        vol = float(k[5])
        qvol = float(k[7])
        tc = int(k[8])
        vwap = qvol / vol if vol > 0 else c
        ts = pd.Timestamp(int(k[0]), unit="ms", tz="UTC")
        # skip current incomplete candle
        close_time_ms = int(k[6])
        if close_time_ms > int(datetime.now(timezone.utc).timestamp() * 1000):
            continue
        rows.append({
            "timestamp": ts, "open": o, "high": h, "low": l, "close": c,
            "volume": vol, "trade_count": tc, "vwap": vwap, "symbol": symbol,
        })
    return pd.DataFrame(rows, columns=COLS)


def append_bars(path: Path, new: pd.DataFrame):
    if new.empty:
        return 0
    if path.exists():
        existing = pd.read_csv(path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    else:
        combined = new
    combined.to_csv(path, index=False)
    return len(new)


def daily_backfill(symbols: list, out_root: Path):
    """Once per day, run Vision backfill for past 3 days to catch gaps."""
    try:
        from src.binan import binance_vision
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=3)
        for sym in symbols:
            frame = binance_vision.fetch_vision_hourly_klines(
                symbol=sym, start=start, end=end, interval="5m", daily_lookback_days=3)
            if frame is not None and not frame.empty:
                df = frame.reset_index()
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                path = out_root / f"{sym}.csv"
                n = append_bars(path, df)
                if n > 0:
                    logger.info(f"backfill {sym}: {n} bars")
    except Exception as e:
        logger.warning(f"backfill failed: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["DOGEUSDT"])
    p.add_argument("--out-root", type=Path, default=REPO / "trainingdata5min")
    p.add_argument("--interval-seconds", type=int, default=300)
    args = p.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"5min collector: symbols={args.symbols} out={args.out_root}")

    last_backfill = None

    while True:
        try:
            for sym in args.symbols:
                bars = fetch_recent_5m(sym, limit=3)
                path = args.out_root / f"{sym}.csv"
                n = append_bars(path, bars)
                if n > 0:
                    logger.info(f"{sym}: +{n} bars")

            now = datetime.now(timezone.utc)
            if last_backfill is None or (now - last_backfill).total_seconds() > 86400:
                daily_backfill(args.symbols, args.out_root)
                last_backfill = now

        except Exception as e:
            logger.error(f"cycle error: {e}")

        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
