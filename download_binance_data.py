#!/usr/bin/env python3
"""Download daily + hourly kline data from Binance REST API for all 30 crypto symbols."""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_30 = [
    "BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "AAVE", "LTC", "XRP", "DOT",
    "UNI", "NEAR", "APT", "ICP", "SHIB", "ADA", "FIL", "ARB", "OP", "INJ",
    "SUI", "TIA", "SEI", "ATOM", "ALGO", "BCH", "BNB", "TRX", "PEPE", "MATIC",
]

EXPANDED_40 = [
    "HBAR", "VET", "RENDER", "FET", "GRT",
    "SAND", "MANA", "AXS", "CRV", "COMP",
    "MKR", "SNX", "ENJ", "1INCH", "SUSHI",
    "YFI", "BAT", "ZRX", "THETA", "FTM",
    "RUNE", "KAVA", "EGLD", "CHZ", "GALA",
    "APE", "LDO", "GMX", "PENDLE", "WLD",
    "JUP", "W", "ENA", "STX", "FLOKI",
    "TON", "KAS", "ONDO", "JASMY", "CFX",
]

SYMBOLS = ORIGINAL_30 + EXPANDED_40

DAILY_DIR = _REPO_ROOT / "trainingdata" / "train"
HOURLY_DIR = _REPO_ROOT / "trainingdatahourlybinance"

KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000


def _ts_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    all_klines = []
    current_start = start_ms
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }
        resp = None
        for attempt in range(5):
            try:
                resp = requests.get(KLINES_URL, params=params, timeout=30)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 10))
                    logger.warning("{} rate limited, sleeping {}s", symbol, retry_after)
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise
        if resp is None or resp.status_code != 200:
            break
        data = resp.json()
        if not data:
            break
        all_klines.extend(data)
        last_open_time = data[-1][0]
        if interval == "1d":
            current_start = last_open_time + 86400000
        else:
            current_start = last_open_time + 3600000
        if len(data) < MAX_LIMIT:
            break
        time.sleep(0.1)
    return all_klines


def _klines_to_df(klines: list[list], symbol: str) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame()
    rows = []
    for k in klines:
        open_time_ms = k[0]
        o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
        vol = float(k[5])
        trade_count = int(k[8])
        quote_vol = float(k[7])
        vwap = quote_vol / vol if vol > 0 else c
        ts = pd.Timestamp(open_time_ms, unit="ms", tz="UTC")
        rows.append({
            "timestamp": ts,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": vol,
            "trade_count": trade_count,
            "vwap": vwap,
            "symbol": symbol,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return df


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new
    if new.empty:
        return existing
    combined = pd.concat([existing, new], axis=0, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return combined


def _verify_gaps(df: pd.DataFrame, interval: str, symbol: str) -> list[str]:
    if len(df) < 2:
        return []
    ts = pd.to_datetime(df["timestamp"], utc=True).sort_values().reset_index(drop=True)
    diffs = ts.diff().dropna()
    if interval == "1d":
        max_gap = pd.Timedelta(days=2)
        label = "daily"
    else:
        max_gap = pd.Timedelta(hours=2)
        label = "hourly"
    issues = []
    gaps = diffs[diffs > max_gap]
    for idx in gaps.index:
        issues.append(f"{symbol} {label}: gap of {diffs[idx]} at {ts[idx-1]} -> {ts[idx]}")
    return issues


def download_symbol(sym: str, interval: str, out_dir: Path, start: datetime, end: datetime) -> tuple[int, list[str]]:
    pair = f"{sym}USDT"
    out_path = out_dir / f"{pair}.csv"
    existing = _load_existing(out_path)

    fetch_start = start
    if not existing.empty:
        last_ts = existing["timestamp"].max()
        if interval == "1d":
            fetch_start_ts = last_ts - pd.Timedelta(days=2)
        else:
            fetch_start_ts = last_ts - pd.Timedelta(hours=2)
        candidate = fetch_start_ts.to_pydatetime()
        if candidate.tzinfo is None:
            candidate = candidate.replace(tzinfo=timezone.utc)
        if candidate > start:
            fetch_start = candidate
        logger.info("{} {} existing={} rows, last={}, fetching from {}", pair, interval, len(existing), last_ts, fetch_start)
    else:
        logger.info("{} {} no existing data, fetching from {}", pair, interval, start)

    start_ms = _ts_ms(fetch_start)
    end_ms = _ts_ms(end)
    klines = _fetch_klines(pair, interval, start_ms, end_ms)
    new_df = _klines_to_df(klines, pair)

    merged = _merge(existing, new_df)
    if merged.empty:
        logger.warning("{} {} no data", pair, interval)
        return 0, [f"{pair} {interval}: no data"]

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    issues = _verify_gaps(merged, interval, pair)
    if issues:
        for iss in issues:
            logger.warning("GAP: {}", iss)
    logger.info("{} {} wrote {} rows [{} -> {}]", pair, interval, len(merged),
                merged["timestamp"].iloc[0], merged["timestamp"].iloc[-1])
    return len(merged), issues


def main():
    parser = argparse.ArgumentParser(description="Download Binance klines for crypto symbols (70 default).")
    parser.add_argument("--symbols", nargs="+", default=None, help="Override symbol list")
    parser.add_argument("--start", default="2022-01-01", help="Start date (default: 2022-01-01)")
    parser.add_argument("--end", default=None, help="End date (default: now)")
    parser.add_argument("--daily-only", action="store_true")
    parser.add_argument("--hourly-only", action="store_true")
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--hourly-dir", type=Path, default=HOURLY_DIR)
    args = parser.parse_args()

    symbols = args.symbols or SYMBOLS
    start = pd.to_datetime(args.start, utc=True).to_pydatetime()
    end = pd.to_datetime(args.end, utc=True).to_pydatetime() if args.end else datetime.now(timezone.utc)

    do_daily = not args.hourly_only
    do_hourly = not args.daily_only

    all_issues = []
    total_daily = 0
    total_hourly = 0

    for sym in symbols:
        if do_daily:
            n, iss = download_symbol(sym, "1d", args.daily_dir, start, end)
            total_daily += n
            all_issues.extend(iss)

        if do_hourly:
            n, iss = download_symbol(sym, "1h", args.hourly_dir, start, end)
            total_hourly += n
            all_issues.extend(iss)

    logger.info("=== DONE === daily_rows={} hourly_rows={} issues={}", total_daily, total_hourly, len(all_issues))
    if all_issues:
        logger.warning("Issues found:")
        for iss in all_issues:
            logger.warning("  {}", iss)

    return 0 if not all_issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
