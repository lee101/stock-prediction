"""Fetch daily klines from Binance public API for arbitrary USDT pairs."""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

BASE_URL = "https://api.binance.com/api/v3"
DATA_DIR = Path(__file__).resolve().parent.parent / "trainingdata" / "train"
CSV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap", "symbol"]
MIN_DAYS = 30
MAX_KLINES_PER_REQUEST = 1000


def fetch_klines(symbol: str, interval: str = "1d", start_ms: Optional[int] = None,
                 limit: int = MAX_KLINES_PER_REQUEST) -> list:
    """Fetch klines from Binance. Returns list of kline arrays."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    resp = requests.get(f"{BASE_URL}/klines", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_all_klines(symbol: str, interval: str = "1d", start_ms: Optional[int] = None) -> list:
    """Fetch all available klines, paginating if needed."""
    all_klines = []
    current_start = start_ms
    while True:
        batch = fetch_klines(symbol, interval, start_ms=current_start)
        if not batch:
            break
        all_klines.extend(batch)
        if len(batch) < MAX_KLINES_PER_REQUEST:
            break
        current_start = batch[-1][0] + 1
        time.sleep(0.1)
    return all_klines


def kline_to_row(kline: list, symbol: str) -> dict:
    """Convert Binance kline array to CSV row dict.
    Binance kline: [open_time, open, high, low, close, volume, close_time,
                    quote_volume, trade_count, taker_buy_base, taker_buy_quote, ignore]
    """
    open_time_ms = kline[0]
    ts = datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc)
    o, h, l, c = float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4])
    vol = float(kline[5])
    quote_vol = float(kline[7])
    trades = int(kline[8])
    vwap = quote_vol / vol if vol > 0 else (o + h + l + c) / 4
    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": vol,
        "trade_count": trades,
        "vwap": vwap,
        "symbol": symbol,
    }


def read_existing_csv(path: Path) -> tuple[list[dict], Optional[datetime]]:
    """Read existing CSV, return rows and last timestamp."""
    if not path.exists():
        return [], None
    rows = []
    last_ts = None
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            try:
                ts_str = row["timestamp"]
                last_ts = datetime.fromisoformat(ts_str)
            except (ValueError, KeyError):
                pass
    return rows, last_ts


def write_csv(path: Path, rows: list[dict]):
    """Write rows to CSV with standard columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def fetch_and_save(symbol: str, data_dir: Path = DATA_DIR, days: Optional[int] = None) -> Path:
    """Fetch klines for symbol and save/update CSV. Returns path to CSV."""
    sym_upper = symbol.upper()
    if not sym_upper.endswith("USDT"):
        sym_upper = sym_upper + "USDT"
    csv_path = data_dir / f"{sym_upper}.csv"

    existing_rows, last_ts = read_existing_csv(csv_path)
    start_ms = None
    if last_ts is not None:
        start_ms = int(last_ts.timestamp() * 1000) + 1
        print(f"{sym_upper}: incremental from {last_ts.date()}, {len(existing_rows)} existing rows")
    else:
        if days:
            start_ms = int((datetime.now(timezone.utc).timestamp() - days * 86400) * 1000)
            print(f"{sym_upper}: fetching last {days} days")
        else:
            print(f"{sym_upper}: fetching all available data")

    klines = fetch_all_klines(sym_upper, start_ms=start_ms)

    if not klines:
        if existing_rows:
            print(f"{sym_upper}: no new data")
            return csv_path
        print(f"WARNING: {sym_upper}: no data returned from API")
        return csv_path

    new_rows = [kline_to_row(k, sym_upper) for k in klines]

    if existing_rows:
        existing_dates = {r["timestamp"][:10] for r in existing_rows}
        new_rows = [r for r in new_rows if r["timestamp"][:10] not in existing_dates]
        all_rows = existing_rows + new_rows
    else:
        all_rows = new_rows

    total_days = len(all_rows)
    if total_days < MIN_DAYS:
        print(f"WARNING: {sym_upper}: only {total_days} days of data (minimum {MIN_DAYS})")

    write_csv(csv_path, all_rows)
    print(f"{sym_upper}: saved {len(all_rows)} rows ({len(new_rows)} new) -> {csv_path}")
    return csv_path


def get_top_symbols(n: int = 50, exclude_stablecoins: bool = True) -> list[str]:
    """Get top N USDT pairs by 24h quote volume."""
    resp = requests.get(f"{BASE_URL}/ticker/24hr", timeout=30)
    resp.raise_for_status()
    tickers = resp.json()

    stablecoins = {"USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP", "USDD", "PYUSD"}
    usdt_pairs = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]
        if exclude_stablecoins and base in stablecoins:
            continue
        usdt_pairs.append((sym, float(t["quoteVolume"])))

    usdt_pairs.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in usdt_pairs[:n]]


def validate_symbol(symbol: str) -> bool:
    """Check if symbol exists on Binance."""
    try:
        resp = requests.get(f"{BASE_URL}/exchangeInfo", params={"symbol": symbol}, timeout=10)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def main():
    parser = argparse.ArgumentParser(description="Fetch Binance daily klines")
    parser.add_argument("--symbols", nargs="+", help="Symbols to fetch (e.g. BTCUSDT ETHUSDT)")
    parser.add_argument("--top-n", type=int, help="Fetch top N by 24h volume")
    parser.add_argument("--days", type=int, help="Limit to last N days")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    symbols = []
    if args.top_n:
        print(f"Fetching top {args.top_n} USDT pairs by volume...")
        symbols = get_top_symbols(args.top_n)
        print(f"Found: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
    if args.symbols:
        for s in args.symbols:
            s_upper = s.upper()
            if not s_upper.endswith("USDT"):
                s_upper += "USDT"
            if s_upper not in symbols:
                symbols.append(s_upper)

    if not symbols:
        parser.error("Provide --symbols or --top-n")

    for sym in symbols:
        try:
            fetch_and_save(sym, data_dir=data_dir, days=args.days)
        except Exception as e:
            print(f"ERROR: {sym}: {e}")
        time.sleep(0.2)


if __name__ == "__main__":
    main()
