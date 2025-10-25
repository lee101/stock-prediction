#!/usr/bin/env python3
"""Aggregate trade summaries to inspect per-symbol PnL trends.

Usage:
    python scripts/trend_analyze_trade_summaries.py marketsimulator/run_logs/*_trades_summary.json

The script prints a table showing per-symbol totals along with the latest
observation and simple moving averages (window configurable).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
import math
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate trading summaries for trend analysis.")
    parser.add_argument(
        "summary_glob",
        nargs="+",
        help="One or more glob patterns pointing to *_trades_summary.json files.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Window size for simple moving average (default: 5).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show only the top/bottom N symbols by cumulative PnL (default: 10).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write aggregated stats as JSON.",
    )
    return parser.parse_args()


def expand_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        found = list(Path().glob(pattern))
        if not found:
            print(f"[warn] no files matched glob '{pattern}'")
        paths.extend(found)
    return sorted(paths)


def load_summary(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data.pop("__overall__", None)
    return data


def _resolve_metrics_path(summary_path: Path) -> Path:
    name = summary_path.name.replace("_trades_summary.json", "_metrics.json")
    return summary_path.with_name(name)


def load_entry_snapshot(summary_path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    metrics_path = _resolve_metrics_path(summary_path)
    if not metrics_path.exists():
        return {}
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"[warn] Failed to parse metrics file {metrics_path}: {exc}")
        return {}
    entry_limits = metrics.get("entry_limits", {})
    per_symbol = entry_limits.get("per_symbol", {})
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for symbol, info in per_symbol.items():
        try:
            entries = float(info.get("entries", 0.0))
        except (TypeError, ValueError):
            entries = 0.0
        entry_limit = info.get("entry_limit")
        try:
            entry_limit_val = float(entry_limit) if entry_limit is not None else None
        except (TypeError, ValueError):
            entry_limit_val = None
        result[symbol.upper()] = {
            "entries": entries,
            "entry_limit": entry_limit_val,
        }
    return result


def aggregate(
    summaries: List[Tuple[Path, Dict[str, Dict[str, float]]]],
    window: int,
) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))
    full_history: Dict[str, List[float]] = defaultdict(list)
    entry_totals: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"entries": 0.0, "runs": 0.0, "limits": []}
    )

    for path, summary in summaries:
        entry_snapshot = load_entry_snapshot(path)
        for symbol, stats in summary.items():
            pnl = float(stats.get("cash_delta", 0.0))
            fees = float(stats.get("fees", 0.0))
            totals[symbol]["pnl"] += pnl
            totals[symbol]["fees"] += fees
            totals[symbol]["trades"] += float(stats.get("trades", 0.0))
            history[symbol].append(pnl)
            full_history[symbol].append(pnl)
            totals[symbol]["latest"] = pnl
            entry_info = entry_snapshot.get(symbol.upper())
            if entry_info:
                entries = float(entry_info.get("entries") or 0.0)
                entry_totals[symbol]["entries"] += entries
                entry_totals[symbol]["runs"] += 1.0
                limit_val = entry_info.get("entry_limit")
                if limit_val is not None:
                    entry_totals[symbol]["limits"].append(float(limit_val))

    for symbol, pnl_values in history.items():
        if pnl_values:
            totals[symbol]["sma"] = sum(pnl_values) / len(pnl_values)
        else:
            totals[symbol]["sma"] = 0.0
    for symbol, values in full_history.items():
        if len(values) > 1:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            totals[symbol]["std"] = math.sqrt(variance)
        else:
            totals[symbol]["std"] = 0.0
        totals[symbol]["observations"] = len(values)

    for symbol, info in entry_totals.items():
        if info["runs"] > 0:
            totals[symbol]["avg_entries"] = info["entries"] / info["runs"]
            totals[symbol]["entry_runs"] = info["runs"]
        if info["limits"]:
            totals[symbol]["entry_limit"] = min(info["limits"])
        elif "entry_limit" not in totals[symbol]:
            totals[symbol]["entry_limit"] = None

    return totals


def display(totals: Dict[str, Dict[str, float]], top_n: int) -> None:
    if not totals:
        print("No trade summaries loaded.")
        return
    sorted_symbols = sorted(totals.items(), key=lambda item: item[1]["pnl"], reverse=True)
    head = sorted_symbols[:top_n]
    tail = sorted_symbols[-top_n:] if len(sorted_symbols) > top_n else []

    def fmt_entry(symbol: str, stats: Dict[str, float]) -> str:
        entry_limit = stats.get("entry_limit")
        avg_entries = stats.get("avg_entries")
        entry_str = ""
        if avg_entries is not None:
            if entry_limit is not None and entry_limit > 0:
                utilization = (avg_entries / entry_limit) * 100.0
                entry_str = f" | Entries {avg_entries:.1f}/{entry_limit:.0f} ({utilization:5.1f}%)"
            else:
                entry_str = f" | Entries {avg_entries:.1f}"
        return (
            f"{symbol:>8} | "
            f"P&L {stats['pnl']:>9.2f} | "
            f"Fees {stats['fees']:>8.2f} | "
            f"SMA {stats.get('sma', 0.0):>8.2f} | "
            f"Std {stats.get('std', 0.0):>8.2f} | "
            f"Latest {stats.get('latest', 0.0):>8.2f} | "
            f"Trades {int(stats.get('trades', 0.0)):>4d}"
            f"{entry_str}"
        )

    print("=== Top Symbols ===")
    for symbol, stats in head:
        print(fmt_entry(symbol, stats))
    if tail:
        print("\n=== Bottom Symbols ===")
        for symbol, stats in tail:
            print(fmt_entry(symbol, stats))


def main() -> None:
    args = parse_args()
    paths = expand_paths(args.summary_glob)
    if not paths:
        return

    summaries = [(path, load_summary(path)) for path in paths]
    totals = aggregate(summaries, window=args.window)
    display(totals, top_n=args.top)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump({k: dict(v) for k, v in totals.items()}, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
