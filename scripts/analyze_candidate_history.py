#!/usr/bin/env python3
"""Summarize candidate readiness history to highlight consistent momentum."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def load_history(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Candidate readiness history not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
    if not rows:
        raise ValueError("History file is empty.")
    return rows


def summarize(rows: List[Dict[str, str]], trailing: int) -> List[Tuple[str, Dict[str, float]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        symbol = row["symbol"].upper()
        grouped[symbol].append(row)

    leaders: List[Tuple[str, Dict[str, float]]] = []
    for symbol, entries in grouped.items():
        # Sort by timestamp ascending
        entries_sorted = sorted(entries, key=lambda r: r["timestamp"])
        recent = entries_sorted[-trailing:] if trailing > 0 else entries_sorted
        pct_values = [float(entry["pct_change"]) for entry in recent]
        sma_values = [float(entry["sma"]) for entry in recent]
        pnl_values = [float(entry["trend_pnl"]) for entry in recent]
        latest = recent[-1]
        leaders.append(
            (
                symbol,
                {
                    "observations": len(entries),
                    "trailing_count": len(recent),
                    "avg_pct": sum(pct_values) / len(pct_values),
                    "avg_sma": sum(sma_values) / len(sma_values),
                    "avg_pnl": sum(pnl_values) / len(pnl_values),
                    "latest_pct": float(latest["pct_change"]),
                    "latest_sma": float(latest["sma"]),
                    "latest_pnl": float(latest["trend_pnl"]),
                    "latest_timestamp": latest["timestamp"],
                },
            )
        )
    # Rank by average pct change then average SMA
    leaders.sort(key=lambda item: (item[1]["avg_pct"], item[1]["avg_sma"]), reverse=True)
    return leaders


def render_markdown(summary: List[Tuple[str, Dict[str, float]]], trailing: int) -> str:
    now = datetime.now(timezone.utc).isoformat()
    lines: List[str] = []
    lines.append(f"# Candidate Momentum Summary — {now}")
    lines.append("")
    lines.append(f"- Trailing window: {trailing if trailing else 'all'} observations\n")
    lines.append("| Symbol | Avg % (trail) | Latest % | Avg SMA | Latest SMA | Latest ΔPnL | Runs | Last Seen |")
    lines.append("|--------|----------------|----------|---------|------------|-------------|------|-----------|")
    for symbol, stats in summary:
        lines.append(
            f"| {symbol} | {stats['avg_pct']*100:>9.2f}% | "
            f"{stats['latest_pct']*100:>7.2f}% | {stats['avg_sma']:>9.2f} | "
            f"{stats['latest_sma']:>11.2f} | {stats['latest_pnl']:>11.2f} | "
            f"{stats['observations']:>4} | {stats['latest_timestamp']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze candidate readiness history.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("marketsimulator/run_logs/candidate_readiness_history.csv"),
        help="Path to candidate readiness history CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marketsimulator/run_logs/candidate_momentum.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--trailing",
        type=int,
        default=5,
        help="Trailing number of observations per symbol (0 = all).",
    )
    args = parser.parse_args()

    rows = load_history(args.history)
    summary = summarize(rows, args.trailing)
    markdown = render_markdown(summary, args.trailing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"[info] Candidate momentum report written to {args.output}")


if __name__ == "__main__":
    main()
