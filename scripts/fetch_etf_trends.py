#!/usr/bin/env python3
"""Fetch daily prices for ETFs and inject trend metrics into trend_summary.json."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}.us&i=d"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


class PriceSourceError(RuntimeError):
    """Raised when a specific data provider cannot deliver usable prices."""


def fetch_prices_stooq(symbol: str, days: int) -> List[Tuple[datetime, float]]:
    resp = requests.get(STOOQ_URL.format(symbol=symbol.lower()), timeout=30)
    resp.raise_for_status()
    rows: List[Tuple[datetime, float]] = []
    for line in resp.text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        date_raw, _open, _high, _low, close_raw = parts[:5]
        if not close_raw or close_raw == "null":
            continue
        try:
            close_val = float(close_raw)
            date_val = datetime.strptime(date_raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        rows.append((date_val, close_val))
    if len(rows) < 2:
        raise PriceSourceError("Not enough price data")
    return rows[-days:]


def fetch_prices_yahoo(symbol: str, days: int) -> List[Tuple[datetime, float]]:
    end_ts = int(datetime.now(tz=timezone.utc).timestamp())
    # Request a small buffer of extra days to account for weekends/holidays.
    start_ts = end_ts - (days + 7) * 24 * 60 * 60
    params = {
        "interval": "1d",
        "includePrePost": "false",
        "events": "history",
        "period1": str(start_ts),
        "period2": str(end_ts),
    }
    resp = requests.get(
        YAHOO_CHART_URL.format(symbol=symbol),
        params=params,
        timeout=30,
        headers={"User-Agent": "marketsimulator-trend-fetch/1.0"},
    )
    resp.raise_for_status()
    payload = resp.json()
    chart = (payload.get("chart") or {}).get("result")
    if not chart:
        raise PriceSourceError("Empty chart result")
    result = chart[0]
    timestamps = result.get("timestamp") or []
    quotes = (result.get("indicators") or {}).get("quote") or [{}]
    closes = quotes[0].get("close") or []
    rows: List[Tuple[datetime, float]] = []
    for ts, close in zip(timestamps, closes):
        if close is None:
            continue
        date_val = datetime.fromtimestamp(ts, tz=timezone.utc)
        rows.append((date_val, float(close)))
    if len(rows) < 2:
        raise PriceSourceError("Not enough price data")
    return rows[-days:]


def fetch_prices(
    symbol: str, days: int, providers: Iterable[str]
) -> Tuple[str, List[Tuple[datetime, float]], float]:
    provider_map = {
        "stooq": fetch_prices_stooq,
        "yahoo": fetch_prices_yahoo,
    }
    errors = []
    for provider in providers:
        handler = provider_map.get(provider)
        if handler is None:
            errors.append(f"unknown provider '{provider}'")
            continue
        try:
            start = time.perf_counter()
            rows = handler(symbol, days)
            latency = time.perf_counter() - start
            return provider, rows, latency
        except PriceSourceError as exc:
            errors.append(f"{provider}: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{provider}: {exc}")
    raise ValueError(f"No data for {symbol.upper()}; attempted providers -> {', '.join(errors)}")


def compute_metrics(prices: List[Tuple[datetime, float]], window: int) -> Dict[str, float]:
    closes = [price for _, price in prices]
    if len(closes) < 2:
        raise ValueError("Not enough price data")
    latest_close = closes[-1]
    start_close = closes[0]
    pnl = latest_close - start_close
    pct_change = (latest_close - start_close) / start_close if start_close else 0.0
    try:
        sma_series = closes[-window:]
    except IndexError:
        sma_series = closes
    sma = sum(sma_series) / len(sma_series)
    std = statistics.pstdev(closes) if len(closes) > 1 else 0.0
    return {
        "latest": latest_close,
        "pnl": pnl,
        "sma": sma,
        "std": std,
        "observations": len(closes),
        "pct_change": pct_change,
    }


def update_summary(
    summary_path: Path,
    metrics: Dict[str, Dict[str, float]],
    providers: Dict[str, str],
) -> None:
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        data = {}
    for symbol, stats in metrics.items():
        entry = data.get(symbol.upper(), {})
        entry.update(
            {
                "avg_entries": entry.get("avg_entries", 0.0),
                "entry_limit": entry.get("entry_limit", 0.0),
                "entry_runs": entry.get("entry_runs", 0.0),
                "fees": entry.get("fees", 0.0),
                "latest": stats["latest"],
                "observations": stats["observations"],
                "pnl": stats["pnl"],
                "pct_change": stats["pct_change"],
                "sma": stats["sma"],
                "std": stats["std"],
                "trades": entry.get("trades", 0.0),
                "provider": providers.get(symbol.upper(), entry.get("provider")),
            }
        )
        data[symbol.upper()] = entry
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ETF trends and update summary JSON.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["XLF", "XLI", "XLE", "XLP", "XLY", "XLV", "XLK", "SMH", "QQQ", "IWM"],
        help="Symbols to fetch (default major sector ETFs).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of days of history to request (default 180).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Window size for SMA calculation (default 50).",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("marketsimulator/run_logs/trend_summary.json"),
        help="Path to trend summary JSON to update.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Optional newline-delimited file listing symbols to fetch.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["stooq", "yahoo"],
        choices=("stooq", "yahoo"),
        help="Ordered list of data providers to attempt (default: stooq yahoo).",
    )
    parser.add_argument(
        "--provider-log",
        type=Path,
        default=None,
        help="Optional CSV file to append provider usage counts (timestamp,provider,count).",
    )
    parser.add_argument(
        "--provider-switch-log",
        type=Path,
        default=None,
        help="Optional CSV file to append provider switch events (timestamp,symbol,previous,current).",
    )
    parser.add_argument(
        "--latency-log",
        type=Path,
        default=None,
        help="Optional CSV file to append provider latency observations (timestamp,symbol,provider,latency_ms).",
    )
    args = parser.parse_args()

    symbols = args.symbols
    if args.symbols_file and args.symbols_file.exists():
        file_symbols = [
            line.strip()
            for line in args.symbols_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if file_symbols:
            symbols = file_symbols

    previous_providers: Dict[str, str] = {}
    if args.summary_path.exists():
        try:
            existing = json.loads(args.summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        for key, value in existing.items():
            if isinstance(value, dict) and "provider" in value:
                previous_providers[key.upper()] = str(value["provider"])

    collected: Dict[str, Dict[str, float]] = {}
    provider_usage: Dict[str, str] = {}
    latency_records: List[Tuple[str, str, float]] = []
    failures: Dict[str, str] = {}
    for symbol in symbols:
        try:
            provider_used, prices, latency = fetch_prices(symbol, args.days, args.providers)
            stats = compute_metrics(prices, args.window)
            collected[symbol.upper()] = stats
            provider_usage[symbol.upper()] = provider_used
            latency_records.append((symbol.upper(), provider_used, latency))
            print(
                f"[info] {symbol.upper()}@{provider_used}: "
                f"SMA={stats['sma']:.2f} pnl={stats['pnl']:.2f} latency={latency*1000:.1f}ms"
            )
        except Exception as exc:  # noqa: BLE001
            failures[symbol.upper()] = str(exc)
            print(f"[warn] Failed to fetch {symbol.upper()}: {exc}")
    if collected:
        update_summary(args.summary_path, collected, provider_usage)
        print(f"[info] Updated {args.summary_path} with {len(collected)} symbols.")
        if provider_usage:
            counts = Counter(provider_usage.values())
            fragments = [f"{provider}:{counts[provider]}" for provider in sorted(counts)]
            print(f"[info] Provider usage this run: {'; '.join(fragments)}")
            timestamp = datetime.now(timezone.utc).isoformat()
            if args.latency_log and latency_records:
                args.latency_log.parent.mkdir(parents=True, exist_ok=True)
                write_header = not args.latency_log.exists()
                with args.latency_log.open("a", encoding="utf-8") as handle:
                    if write_header:
                        handle.write("timestamp,symbol,provider,latency_ms\n")
                    for symbol, provider, latency in latency_records:
                        handle.write(
                            f"{timestamp},{symbol},{provider},{latency*1000:.3f}\n"
                        )
            if latency_records:
                latency_by_provider: Dict[str, List[float]] = {}
                for symbol, provider, latency in latency_records:
                    latency_by_provider.setdefault(provider, []).append(latency)
                for provider, values in sorted(latency_by_provider.items()):
                    avg_ms = sum(values) / len(values) * 1000
                    max_ms = max(values) * 1000
                    print(
                        f"[info] Latency[{provider}]: avg={avg_ms:.1f}ms "
                        f"max={max_ms:.1f}ms samples={len(values)}"
                    )
            if args.provider_log:
                args.provider_log.parent.mkdir(parents=True, exist_ok=True)
                write_header = not args.provider_log.exists()
                with args.provider_log.open("a", encoding="utf-8") as handle:
                    if write_header:
                        handle.write("timestamp,provider,count\n")
                    for provider, count in sorted(counts.items()):
                        handle.write(f"{timestamp},{provider},{count}\n")
            switch_events: List[Tuple[str, str, str]] = []
            for symbol, provider in provider_usage.items():
                previous = previous_providers.get(symbol)
                if previous and previous != provider:
                    print(
                        f"[notice] Provider change for {symbol}: "
                        f"{previous} -> {provider}"
                    )
                    switch_events.append((symbol, previous, provider))
            if switch_events and args.provider_switch_log:
                args.provider_switch_log.parent.mkdir(parents=True, exist_ok=True)
                write_header = not args.provider_switch_log.exists()
                with args.provider_switch_log.open("a", encoding="utf-8") as handle:
                    if write_header:
                        handle.write("timestamp,symbol,previous,current\n")
                    for symbol, previous, current in sorted(switch_events):
                        handle.write(f"{timestamp},{symbol},{previous},{current}\n")
    if failures:
        print("[warn] Some symbols failed:", failures)


if __name__ == "__main__":
    main()
