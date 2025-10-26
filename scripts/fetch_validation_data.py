#!/usr/bin/env python3
"""
Fetch the most recent OHLCV data for validation purposes without depending on yfinance.

The script queries Yahoo Finance's public chart API directly and writes CSV files
with the same schema used in `trainingdata/`. It is intended for refreshing the
`trainingdata2/` directory with the latest hourly bars so validation always runs
against unseen data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import csv
import time
from pathlib import Path
from typing import Iterable, List, Optional

import requests

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


@dataclass(frozen=True)
class FetchResult:
    symbol: str
    rows_written: int
    output_path: Path
    last_timestamp: Optional[datetime]


def _read_last_timestamp(csv_path: Path) -> Optional[datetime]:
    if not csv_path.exists():
        return None
    try:
        with csv_path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                return None
            try:
                ts_idx = header.index("timestamp")
            except ValueError:
                return None
            last_value: Optional[str] = None
            for row in reader:
                if len(row) <= ts_idx:
                    continue
                candidate = row[ts_idx].strip()
                if candidate:
                    last_value = candidate
            if not last_value:
                return None
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(last_value, fmt)
                except ValueError:
                    continue
    except FileNotFoundError:
        return None
    return None


def load_latest_timestamp(symbol: str, base_dir: Path) -> Optional[datetime]:
    """
    Inspect an existing dataset directory and return the most recent timestamp
    available for the symbol. Handles either flat CSVs or nested `train/test` folders.
    """
    candidates: List[Optional[datetime]] = []
    if base_dir.is_file():
        timestamp = _read_last_timestamp(base_dir)
        if timestamp:
            candidates.append(timestamp)
    elif base_dir.is_dir():
        for sub in ("", "train", "test"):
            path = base_dir / sub / f"{symbol}.csv" if sub else base_dir / f"{symbol}.csv"
            timestamp = _read_last_timestamp(path)
            if timestamp:
                candidates.append(timestamp)
    return max(candidates) if candidates else None


def _format_timestamp(epoch_seconds: int) -> datetime:
    """
    Align epoch timestamps to the repository's canonical formatting where each bar
    is recorded at HH:07:57.601944 to match existing CSV cadence.
    """
    base = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    floored = base.replace(minute=0, second=0, microsecond=0)
    return (floored + timedelta(minutes=7, seconds=57, microseconds=601_944)).replace(tzinfo=None)


def fetch_chart(symbol: str, interval: str = "1h", range_: str = "1mo", max_attempts: int = 5) -> List[dict]:
    """
    Fetch hourly bars from Yahoo's public chart API and return a DataFrame matching
    the schema used in training CSVs.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "range": range_,
        "interval": interval,
        "includePrePost": "false",
        "region": "US",
        "lang": "en-US",
        "corsDomain": "finance.yahoo.com",
    }
    last_error: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=30)
            if response.status_code == 429:
                sleep_for = min(2 ** attempt, 30)
                time.sleep(sleep_for)
                continue
            response.raise_for_status()
            payload = response.json()
            break
        except Exception as exc:
            last_error = exc
            sleep_for = min(2 ** attempt, 30)
            time.sleep(sleep_for)
    else:
        raise RuntimeError(f"Failed to fetch data for {symbol} after {max_attempts} attempts") from last_error

    result = payload.get("chart", {}).get("result")
    if not result:
        error = payload.get("chart", {}).get("error")
        raise RuntimeError(f"No chart data for {symbol}: {error!r}")
    info = result[0]
    timestamps = info.get("timestamp", [])
    if not timestamps:
        return []

    quote = info.get("indicators", {}).get("quote", [{}])[0]
    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])

    entries: List[dict] = []
    for ts, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, volumes):
        if None in (o, h, l, c, v):
            continue
        volume = int(v)
        if volume <= 0:
            continue
        entries.append(
            {
                "timestamp": _format_timestamp(int(ts)),
                "Open": float(o),
                "High": float(h),
                "Low": float(l),
                "Close": float(c),
                "Volume": volume,
            }
        )
    entries.sort(key=lambda row: row["timestamp"])
    return entries


def write_validation_csv(rows: List[dict], symbol: str, out_dir: Path, start_after: Optional[datetime]) -> FetchResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{symbol}.csv"

    filtered: List[dict] = []
    for row in rows:
        if start_after is not None and row["timestamp"] <= start_after:
            continue
        filtered.append(row)

    if not filtered:
        return FetchResult(symbol=symbol, rows_written=0, output_path=csv_path, last_timestamp=start_after)

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        writer.writeheader()
        for row in filtered:
            serialized = row.copy()
            serialized["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S.%f")
            writer.writerow(serialized)

    last_ts = filtered[-1]["timestamp"]
    return FetchResult(symbol=symbol, rows_written=len(filtered), output_path=csv_path, last_timestamp=last_ts)


def parse_symbols(arg_symbols: Optional[Iterable[str]], default_symbols: Iterable[str]) -> List[str]:
    if arg_symbols:
        return [sym.strip().upper() for sym in arg_symbols if sym.strip()]
    return [sym.strip().upper() for sym in default_symbols if sym.strip()]


def read_default_symbols(config_path: Path) -> List[str]:
    try:
        data = json.loads(config_path.read_text())
        symbols = data.get("data", {}).get("symbols")
        if isinstance(symbols, list) and symbols:
            return [str(sym).upper() for sym in symbols]
    except Exception:
        pass
    return ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NFLX", "NVDA", "TSLA"]


def discover_symbols_in_dir(base_dir: Path) -> List[str]:
    if not base_dir.exists():
        return []
    candidates: List[str] = []
    search_dirs = [base_dir]
    if base_dir.is_dir():
        for sub in ("train", "test", "validation"):
            subdir = base_dir / sub
            if subdir.is_dir():
                search_dirs.append(subdir)

    for directory in search_dirs:
        if directory.is_dir():
            candidates.extend(p.stem.upper() for p in directory.glob("*.csv"))
        elif directory.is_file() and directory.suffix == ".csv":
            candidates.append(directory.stem.upper())
    return sorted({sym for sym in candidates if sym})


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch latest hourly OHLCV data for validation.")
    parser.add_argument("--symbols", nargs="*", help="Symbols to download (defaults to hftraining config symbols).")
    parser.add_argument("--base-dir", type=Path, default=Path("trainingdata"), help="Existing dataset directory for deduping.")
    parser.add_argument("--out", type=Path, default=Path("trainingdata2/validation"), help="Output directory for validation CSVs.")
    parser.add_argument("--range", dest="range_", default="1mo", help="Yahoo Finance range parameter (default: 1mo).")
    parser.add_argument("--interval", default="1h", help="Yahoo Finance interval (default: 1h).")
    parser.add_argument("--config", type=Path, default=Path("hftraining/cli_quick_config.json"), help="Config file for default symbols.")
    parser.add_argument("--force", action="store_true", help="Write all fetched rows even if they overlap with existing data.")
    args = parser.parse_args()

    default_symbols = read_default_symbols(args.config)
    discovered = discover_symbols_in_dir(args.base_dir)
    merged_defaults = default_symbols or discovered or ["AAPL"]
    if discovered and set(default_symbols) != set(discovered):
        merged_defaults = sorted({*default_symbols, *discovered})

    symbols = parse_symbols(args.symbols, merged_defaults)

    results: List[FetchResult] = []
    for symbol in symbols:
        latest = load_latest_timestamp(symbol, args.base_dir)
        df = fetch_chart(symbol, interval=args.interval, range_=args.range_)
        start_after = None if args.force else latest
        result = write_validation_csv(df, symbol, args.out, start_after=start_after)
        results.append(result)
        time.sleep(1.0)

    summary = {
        "downloaded": [
            {
                "symbol": res.symbol,
                "rows_written": res.rows_written,
                "output_path": str(res.output_path),
                "last_timestamp": res.last_timestamp.isoformat() if res.last_timestamp else None,
            }
            for res in results
        ]
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
