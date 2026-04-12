#!/usr/bin/env python3
"""
Download daily OHLCV bars from Alpaca for all symbols in a symbol list.

Reads from stock_info.csv (5782 NYSE/NASDAQ tickers from dhhagan/stocks) or
a custom symbols file. Saves to trainingdata/ in the same format as existing
CSVs: timestamp,open,high,low,close,volume,trade_count,vwap,symbol

Usage:
    # All 5782 symbols from stock_info.csv
    python scripts/download_alpaca_stocks.py

    # Custom symbol list
    python scripts/download_alpaca_stocks.py --symbols-file symbol_lists/stocks_1000_v1.txt

    # Specific symbols
    python scripts/download_alpaca_stocks.py --symbols AAPL MSFT TSLA

    # Refresh existing (re-download last N rows)
    python scripts/download_alpaca_stocks.py --refresh-days 10

    # Dry run
    python scripts/download_alpaca_stocks.py --dry-run --limit 5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from alpaca.data import StockBarsRequest, StockHistoricalDataClient, TimeFrame
from alpaca.data.enums import DataFeed

# Load keys from env_real — same source as alpaca_wrapper
from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

DEFAULT_OUTPUT_DIR = REPO_ROOT / "trainingdata"
DEFAULT_START_DATE = "2020-01-01"
STOCK_INFO_CSV = REPO_ROOT.parent.parent / "code/stock-prediction/stock_info.csv"
# Also try sdb-disk location
STOCK_INFO_CSV_SDB = Path("/sdb-disk/code/stock-prediction/stock_info.csv")

# Alpaca rate limits: ~200 req/min for free tier, ~unlimited for paid
BATCH_SIZE = 10          # symbols per request (Alpaca supports multi-symbol)
BATCH_DELAY = 0.4        # seconds between batches (stay well under rate limit)
RETRY_DELAY = 5.0        # seconds before retrying a failed batch
MAX_RETRIES = 3


def load_stock_info_symbols() -> list[str]:
    """Load tickers from the dhhagan/stocks stock_info.csv."""
    for path in [STOCK_INFO_CSV_SDB, STOCK_INFO_CSV, REPO_ROOT / "stock_info.csv"]:
        if path.exists():
            with path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            tickers = [r["Ticker"].strip().upper() for r in rows if r.get("Ticker", "").strip()]
            print(f"Loaded {len(tickers)} tickers from {path}")
            return tickers
    raise FileNotFoundError("stock_info.csv not found — run: curl -sL https://raw.githubusercontent.com/dhhagan/stocks/master/scripts/stock_info.csv > stock_info.csv")


def load_symbols_from_file(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip().upper() for line in lines if line.strip() and not line.startswith("#")]


def normalize_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize Alpaca bar dataframe to the standard trainingdata CSV format."""
    df = df.reset_index()

    # Alpaca multi-symbol response has a 'symbol' level in the index
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].copy()

    # Alpaca column names: open, high, low, close, volume, trade_count, vwap, timestamp
    rename = {}
    for col in df.columns:
        if col.lower() == "timestamp":
            rename[col] = "timestamp"

    if rename:
        df = df.rename(columns=rename)

    # Ensure timestamp column exists
    if "timestamp" not in df.columns:
        # Sometimes it's in the index
        df = df.reset_index()
        if "timestamp" in df.columns:
            pass
        else:
            raise ValueError(f"No timestamp column found in Alpaca response for {symbol}")

    # Convert timestamp to UTC string matching existing format
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"(\+\d{2})(\d{2})$", r"\1:\2", regex=True)

    # Add symbol column
    df["symbol"] = symbol

    # Select and order columns matching existing CSVs
    out_cols = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap", "symbol"]
    for col in ["trade_count", "vwap"]:
        if col not in df.columns:
            df[col] = 0.0

    return df[out_cols].copy()


def download_batch(
    client: StockHistoricalDataClient,
    symbols: list[str],
    start_date: str,
    end_date: str,
    output_dir: Path,
    *,
    force: bool = False,
    min_rows: int = 50,
) -> dict[str, str]:
    """Download a batch of symbols. Returns {symbol: status}."""
    # Filter out already-done symbols (unless force)
    if not force:
        to_fetch = []
        for sym in symbols:
            dest = output_dir / f"{sym}.csv"
            if not dest.exists():
                to_fetch.append(sym)
        if not to_fetch:
            return {sym: "skip" for sym in symbols}
    else:
        to_fetch = symbols

    if not to_fetch:
        return {sym: "skip" for sym in symbols}

    request = StockBarsRequest(
        symbol_or_symbols=to_fetch,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
        adjustment="all",       # split + dividend adjusted
        feed=DataFeed.SIP,      # SIP = consolidated feed (all exchanges)
    )

    try:
        bars_df = client.get_stock_bars(request).df
    except Exception as exc:
        return {sym: f"error:{exc}" for sym in to_fetch} | {sym: "skip" for sym in symbols if sym not in to_fetch}

    results = {sym: "skip" for sym in symbols if sym not in to_fetch}
    if bars_df is None or bars_df.empty:
        for sym in to_fetch:
            results[sym] = "empty"
        return results

    # Multi-symbol response: index is (symbol, timestamp)
    bars_df = bars_df.reset_index()

    for sym in to_fetch:
        dest = output_dir / f"{sym}.csv"
        try:
            if "symbol" in bars_df.columns:
                sym_df = bars_df[bars_df["symbol"] == sym].copy()
            else:
                sym_df = bars_df.copy()

            if sym_df.empty or len(sym_df) < min_rows:
                results[sym] = f"thin:{len(sym_df)}"
                continue

            normalized = normalize_bars(sym_df, sym)
            normalized.to_csv(dest, index=False)
            results[sym] = f"ok:{len(normalized)}"
        except Exception as exc:
            results[sym] = f"error:{exc}"

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", help="Explicit symbol list")
    parser.add_argument("--symbols-file", help="Text file with one symbol per line")
    parser.add_argument("--stock-info-csv", help="Path to stock_info.csv (default: auto-detect)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for CSVs")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--force", action="store_true", help="Re-download even if CSV exists")
    parser.add_argument("--refresh-days", type=int, default=0, help="If set, re-download last N days worth")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N symbols (for testing)")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N symbols (for resuming)")
    parser.add_argument("--dry-run", action="store_true", help="Print symbols without downloading")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Symbols per Alpaca request")
    parser.add_argument("--min-rows", type=int, default=50, help="Minimum rows to save (thin=skip)")
    parser.add_argument("--sip", action="store_true", default=True, help="Use SIP consolidated feed (default)")
    parser.add_argument("--iex", action="store_true", help="Use IEX feed instead of SIP")
    args = parser.parse_args()

    # Resolve symbols
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.symbols_file:
        symbols = load_symbols_from_file(args.symbols_file)
    else:
        symbols = load_stock_info_symbols()

    if args.skip:
        symbols = symbols[args.skip:]
    if args.limit:
        symbols = symbols[:args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    end_date = args.end_date or (datetime.now(tz=timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = args.start_date

    print(f"Symbols: {len(symbols)}")
    print(f"Output: {output_dir}")
    print(f"Date range: {start_date} → {end_date}")
    print(f"Batch size: {args.batch_size}")

    if args.dry_run:
        already = sum(1 for s in symbols if (output_dir / f"{s}.csv").exists())
        print(f"Dry run — {already}/{len(symbols)} already have CSVs, {len(symbols)-already} to download")
        print("First 20:", symbols[:20])
        return 0

    client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)

    ok = err = skip = thin = 0
    batches = [symbols[i : i + args.batch_size] for i in range(0, len(symbols), args.batch_size)]
    total = len(symbols)
    done = 0

    for batch_idx, batch in enumerate(batches, 1):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                results = download_batch(
                    client, batch, start_date, end_date, output_dir,
                    force=args.force, min_rows=args.min_rows,
                )
                break
            except Exception as exc:
                print(f"  [WARN] batch {batch_idx} attempt {attempt}/{MAX_RETRIES} failed: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    results = {sym: f"error:{exc}" for sym in batch}

        for sym, status in results.items():
            done += 1
            if status.startswith("ok"):
                ok += 1
                rows = status.split(":")[1]
                if batch_idx % 10 == 0 or done <= 20:
                    print(f"  [{done}/{total}] {sym}: {rows} rows")
            elif status == "skip":
                skip += 1
            elif status.startswith("thin"):
                thin += 1
                print(f"  [{done}/{total}] {sym}: THIN {status}")
            else:
                err += 1
                print(f"  [{done}/{total}] {sym}: ERROR {status}")

        if batch_idx % 50 == 0:
            pct = done / total * 100
            print(f"\n=== Progress: {done}/{total} ({pct:.0f}%) — ok={ok} skip={skip} thin={thin} err={err} ===\n")

        time.sleep(BATCH_DELAY)

    print(f"\nDone: {total} symbols — ok={ok} skip={skip} thin={thin} err={err}")
    print(f"CSVs in {output_dir}: {sum(1 for _ in output_dir.glob('*.csv'))} files")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
