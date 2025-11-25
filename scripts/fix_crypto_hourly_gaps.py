from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env_real import BINANCE_API_KEY, BINANCE_SECRET
from binance import Client
import requests


def _to_utc(ts: datetime | pd.Timestamp) -> datetime:
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.to_pydatetime()


def _detect_gaps(frame: pd.DataFrame, max_gap_hours: float = 1.5) -> List[Tuple[datetime, datetime]]:
    ts = pd.to_datetime(frame["timestamp"], utc=True).sort_values()
    deltas = ts.diff().dt.total_seconds().div(3600)
    gaps = []
    for prev, curr, delta in zip(ts[:-1], ts[1:], deltas[1:]):
        if delta > max_gap_hours:
            gaps.append((_to_utc(prev), _to_utc(curr)))
    return gaps


def _binance_symbol(alpaca_symbol: str) -> str:
    sym = alpaca_symbol.upper()
    if sym.endswith("USD"):
        return sym.replace("USD", "USDT")
    return sym


def _fetch_binance_klines(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    client = Client(BINANCE_API_KEY, BINANCE_SECRET)
    binance_symbol = _binance_symbol(symbol)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    rows = []
    while True:
        klines = client.get_klines(
            symbol=binance_symbol,
            interval=Client.KLINE_INTERVAL_1HOUR,
            startTime=start_ms,
            endTime=end_ms,
            limit=1000,
        )
        if not klines:
            break
        rows.extend(klines)
        last_end = klines[-1][0]
        # Move window forward by one hour to avoid duplicates
        start_ms = last_end + 60 * 60 * 1000
        if start_ms >= end_ms:
            break
        # Safety sleep to respect binance limits
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    keep = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    keep["symbol"] = symbol.upper()
    return keep


def _fetch_coinbase_candles(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    # Coinbase product uses dash: e.g., UNI-USD
    product = symbol.upper().replace("USD", "-USD")
    granularity = 3600
    rows = []
    # Coinbase limits window; step by ~200 candles (200h)
    step = timedelta(hours=180)
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + step, end)
        params = {
            "start": cursor.isoformat(),
            "end": chunk_end.isoformat(),
            "granularity": granularity,
        }
        url = f"https://api.exchange.coinbase.com/products/{product}/candles"
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            cursor = chunk_end
            continue
        data = resp.json()
        for ohlc in data:
            ts, low, high, open_, close, volume = ohlc
            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "symbol": symbol.upper(),
                }
            )
        cursor = chunk_end
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp")
    return df


def _fill_gaps(frame: pd.DataFrame, gaps: List[Tuple[datetime, datetime]], symbol: str) -> pd.DataFrame:
    fills: List[pd.DataFrame] = []
    for start, end in gaps:
        # Extend window slightly to cover boundary bars
        pad_start = start - timedelta(hours=2)
        pad_end = end + timedelta(hours=2)
        try:
            fetched = _fetch_binance_klines(symbol, pad_start, pad_end)
        except Exception:
            fetched = _fetch_coinbase_candles(symbol, pad_start, pad_end)
        if fetched.empty:
            continue
        fills.append(fetched)
    if not fills:
        return frame
    combined_fill = pd.concat(fills, ignore_index=True)
    merged = (
        pd.concat([frame, combined_fill], ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return merged


def fix_gaps(symbol: str, data_path: Path, max_gap_hours: float = 1.5) -> Tuple[int, List[Tuple[datetime, datetime]]]:
    symbol = symbol.upper()
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file {data_path}")
    frame = pd.read_csv(data_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    gaps = _detect_gaps(frame, max_gap_hours=max_gap_hours)
    if not gaps:
        print("No gaps detected.")
        return 0, []
    print(f"Detected {len(gaps)} gaps; fetching from Binance…")
    merged = _fill_gaps(frame, gaps, symbol)
    merged.to_csv(data_path, index=False)
    remaining = _detect_gaps(merged, max_gap_hours=max_gap_hours)
    print(f"Saved merged data ({len(merged)} rows). Remaining gaps: {len(remaining)}")
    return len(gaps), remaining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill hourly crypto data gaps using Binance klines.")
    parser.add_argument("--symbol", required=True, help="Crypto symbol, e.g., UNIUSD")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("trainingdatahourly/crypto"),
        help="Directory containing SYMBOL.csv",
    )
    parser.add_argument("--max-gap-hours", type=float, default=1.5, help="Threshold to consider as a gap.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = args.data_root / f"{args.symbol.upper()}.csv"
    fix_gaps(args.symbol, path, max_gap_hours=args.max_gap_hours)
