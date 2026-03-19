#!/usr/bin/env python3
"""Summarize paused-streak escalations from trend_paused_escalations.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Paused escalation log not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
    return rows


def summarize(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, float]]]:
    rows_sorted = sorted(rows, key=lambda r: r.get("timestamp", ""))
    latest: Dict[str, Dict[str, float]] = {}
    aggregate_counts: Dict[str, int] = defaultdict(int)
    max_streaks: Dict[str, int] = defaultdict(int)

    for row in rows_sorted:
        symbol = row.get("symbol", "").upper()
        streak = int(row.get("streak") or 0)
        pnl_raw = row.get("pnl")
        pnl = float(pnl_raw) if pnl_raw not in (None, "",) else float("nan")
        aggregate_counts[symbol] += 1
        max_streaks[symbol] = max(max_streaks[symbol], streak)
        latest[symbol] = {"timestamp": row.get("timestamp", ""), "streak": streak, "pnl": pnl}

    summary: Dict[str, Dict[str, float]] = {}
    for symbol, info in latest.items():
        summary[symbol] = {
            "latest_timestamp": info["timestamp"],
            "latest_streak": info["streak"],
            "max_streak": max_streaks[symbol],
            "events": aggregate_counts[symbol],
            "latest_trend_pnl": info["pnl"],
        }
    return rows_sorted, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Report paused escalation streaks.")
    parser.add_argument(
        "log",
        type=Path,
        nargs="?",
        default=Path("marketsimulator/run_logs/trend_paused_escalations.csv"),
        help="Path to the paused escalation CSV log.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show at most N symbols in the summary table (default 10).",
    )
    args = parser.parse_args()

    rows = load_rows(args.log)
    if not rows:
        print("[info] Paused escalation log is empty.")
        return

    _, summary = summarize(rows)
    print("Symbol | Events | Latest Streak | Max Streak | Last Timestamp           | Last Trend PnL")
    print("-------|--------|---------------|------------|--------------------------|---------------")
    ordered = sorted(
        summary.items(),
        key=lambda item: (item[1]["latest_streak"], item[1]["max_streak"], item[0]),
        reverse=True,
    )
    for idx, (symbol, info) in enumerate(ordered):
        if idx >= args.top:
            break
        pnl = info["latest_trend_pnl"]
        pnl_str = "nan" if pnl != pnl else f"{pnl:.2f}"
        print(
            f"{symbol:>6} | {int(info['events']):>6} | {int(info['latest_streak']):>13} | "
            f"{int(info['max_streak']):>10} | {info['latest_timestamp']:<24} | {pnl_str:>13}"
        )


if __name__ == "__main__":
    main()
