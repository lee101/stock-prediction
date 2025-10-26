#!/usr/bin/env python3
"""Append trend_summary.json entries to a historical CSV."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append trend summary metrics to CSV history.")
    parser.add_argument("trend_json", type=Path, help="Path to trend_summary.json.")
    parser.add_argument("history_csv", type=Path, help="CSV file to append records to.")
    parser.add_argument(
        "--timestamp",
        default=datetime.now(timezone.utc).isoformat(),
        help="Timestamp to record (default: current UTC ISO).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Optional comma-separated list of symbols to include (default: all).",
    )
    return parser.parse_args()


def load_trend(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_history(trend: dict, csv_path: Path, timestamp: str, symbols: list[str] | None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = [
        "timestamp",
        "symbol",
        "pnl",
        "fees",
        "trades",
        "sma",
        "std",
        "latest",
        "observations",
        "avg_entries",
        "entry_limit",
        "entry_utilization",
    ]
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for symbol, stats in trend.items():
            if symbol == "__overall__":
                continue
            if symbols and symbol not in symbols:
                continue
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "pnl": stats.get("pnl", 0.0),
                    "fees": stats.get("fees", 0.0),
                    "trades": stats.get("trades", 0.0),
                    "sma": stats.get("sma", 0.0),
                    "std": stats.get("std", 0.0),
                    "latest": stats.get("latest", 0.0),
                    "observations": stats.get("observations", 0.0),
                    "avg_entries": stats.get("avg_entries", 0.0),
                    "entry_limit": stats.get("entry_limit", 0.0),
                    "entry_utilization": (
                        (stats.get("avg_entries", 0.0) / stats.get("entry_limit", 1.0))
                        if stats.get("entry_limit")
                        else 0.0
                    ),
                }
            )


def main() -> int:
    args = parse_args()
    if not args.trend_json.exists():
        print(f"[warn] Trend file not found: {args.trend_json}")
        return 0
    trend = load_trend(args.trend_json)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    append_history(trend, args.history_csv, args.timestamp, symbols)
    print(
        f"[info] Appended trend stats for {', '.join(symbols) if symbols else 'all symbols'}"
        f" at {args.timestamp}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
