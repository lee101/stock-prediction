#!/usr/bin/env python3
"""Validate trend_summary.json against alert thresholds."""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, Optional

from trade_limit_utils import (
    apply_trend_threshold_defaults,
    entry_limit_to_trade_limit,
    parse_entry_limit_map,
    parse_trade_limit_map,
    resolve_entry_limit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check trend summary for anomalies.")
    parser.add_argument("summary", type=Path, help="Path to trend_summary.json.")
    parser.add_argument(
        "--min-sma",
        type=float,
        default=None,
        help="Alert if any symbol SMA falls below this value.",
    )
    parser.add_argument(
        "--max-std",
        type=float,
        default=None,
        help="Alert if any symbol standard deviation exceeds this value.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols to monitor (default: all).",
    )
    parser.add_argument(
        "--trades-glob",
        type=str,
        default=None,
        help="Optional glob pointing to trades_summary.json files for the latest run.",
    )
    parser.add_argument(
        "--max-trades",
        type=float,
        default=None,
        help="Alert if any symbol executes more trades than this limit.",
    )
    parser.add_argument(
        "--max-trades-map",
        type=str,
        default=None,
        help="Per-symbol overrides for maximum trades (e.g., 'NVDA@maxdiff:10,AAPL@maxdiff:22').",
    )
    parser.add_argument(
        "--entry-util-threshold",
        type=float,
        default=0.8,
        help="Emit a soft warning when entries consume at least this fraction of the configured limit (default: 0.8).",
    )
    return parser.parse_args()


def load_summary(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def find_latest_trade_summary(pattern: str) -> Optional[Path]:
    matched = [Path(p) for p in glob(pattern)]
    if not matched:
        return None
    matched.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matched[0]


def _metrics_path_for(trade_summary_path: Path) -> Path:
    return trade_summary_path.with_name(
        trade_summary_path.name.replace("_trades_summary.json", "_metrics.json")
    )


def load_entry_metrics(summary_path: Path) -> Dict[str, Dict[str, float]]:
    metrics_path = _metrics_path_for(summary_path)
    if not metrics_path.exists():
        return {}
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"[warn] Unable to parse metrics file {metrics_path}: {exc}")
        return {}
    entry_limits = payload.get("entry_limits", {})
    per_symbol = entry_limits.get("per_symbol", {})
    result: Dict[str, Dict[str, float]] = {}
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


def main() -> int:
    args = parse_args()
    if not args.summary.exists():
        print(f"[warn] trend summary not found: {args.summary}")
        return 0

    summary = load_summary(args.summary)
    min_sma, max_std = apply_trend_threshold_defaults(args.min_sma, args.max_std)
    symbols = set(sym.strip() for sym in args.symbols.split(",")) if args.symbols else None
    hard_alerts = []
    soft_alerts = []
    for symbol, stats in summary.items():
        if symbol == "__overall__":
            continue
        if symbols and symbol not in symbols:
            continue
        sma = float(stats.get("sma", 0.0))
        std = float(stats.get("std", 0.0))
        if sma < min_sma:
            hard_alerts.append(f"{symbol}: SMA {sma:.2f} < {min_sma:.2f}")
        if std > max_std:
            hard_alerts.append(f"{symbol}: std {std:.2f} > {max_std:.2f}")
    max_trades_map = parse_trade_limit_map(args.max_trades_map, verbose=False)
    entry_limit_map = parse_entry_limit_map(os.getenv("MARKETSIM_SYMBOL_MAX_ENTRIES_MAP"))
    if args.trades_glob and (args.max_trades is not None or max_trades_map):
        latest = find_latest_trade_summary(args.trades_glob)
        if latest is None:
            print(f"[warn] No trades summary matched glob '{args.trades_glob}'")
        elif not latest.exists():
            print(f"[warn] trades summary {latest} not found on disk")
        else:
            trade_summary = load_summary(latest)
            entry_metrics = load_entry_metrics(latest)
            for symbol, stats in trade_summary.items():
                if symbol == "__overall__":
                    continue
                if symbols and symbol not in symbols:
                    continue
                limit = max_trades_map.get(symbol, args.max_trades)
                entry_limit = resolve_entry_limit(entry_limit_map, symbol)
                entry_limit_trades = entry_limit_to_trade_limit(entry_limit)
                if entry_limit_trades is not None:
                    limit = entry_limit_trades if limit is None else min(limit, entry_limit_trades)
                if limit is None:
                    continue
                trades = float(stats.get("trades", 0.0))
                if trades > limit:
                    hard_alerts.append(
                        f"{symbol}: trade count {trades:.0f} exceeds limit {limit:.0f} (source {latest.name})"
                    )
                if entry_metrics and args.entry_util_threshold is not None and args.entry_util_threshold > 0:
                    metrics_info = entry_metrics.get(symbol.upper())
                    if metrics_info:
                        entries = metrics_info.get("entries", 0.0)
                        entry_cap = metrics_info.get("entry_limit")
                        if entry_cap and entry_cap > 0:
                            utilization = entries / entry_cap
                            if utilization >= args.entry_util_threshold:
                                soft_alerts.append(
                                    f"{symbol}: entries {entries:.0f}/{entry_cap:.0f} "
                                    f"({utilization*100:.1f}%) nearing cap (source {latest.name})"
                                )
    if hard_alerts:
        print("[warn] Trend alerts triggered:")
        for alert in hard_alerts:
            print(f"  - {alert}")
        return 1
    print("[info] Trend summary within configured thresholds.")
    if soft_alerts:
        print("[info] Soft utilization warnings:")
        for alert in soft_alerts:
            print(f"  - {alert}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
