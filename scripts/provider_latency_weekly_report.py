#!/usr/bin/env python3
"""Generate a weekly latency trend report highlighting CRIT/WARN changes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def load_history(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    entries: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def aggregate(entries: List[Dict[str, object]]) -> Dict[str, Counter]:
    counts: Dict[str, Counter] = {}
    for entry in entries:
        provider_map = entry.get("provider_severity", {})
        for provider, data in provider_map.items():
            key = provider.upper()
            counter = counts.setdefault(key, Counter())
            for severity, value in data.items():
                counter[severity.upper()] += value
    return counts


def compute_trend(entries: List[Dict[str, object]], window: int, compare_window: int) -> Dict[str, Dict[str, int]]:
    if len(entries) < window + compare_window:
        return {}
    current_entries = entries[-window:]
    previous_entries = entries[-(window + compare_window) : -window]

    current = aggregate(current_entries)
    previous = aggregate(previous_entries)

    deltas: Dict[str, Dict[str, int]] = {}
    for provider in current.keys() | previous.keys():
        deltas[provider.upper()] = {
            "CRIT": current.get(provider, {}).get("CRIT", 0)
            - previous.get(provider, {}).get("CRIT", 0),
            "WARN": current.get(provider, {}).get("WARN", 0)
            - previous.get(provider, {}).get("WARN", 0),
        }
    return deltas


def build_report(entries: List[Dict[str, object]], window: int, compare_window: int, min_delta: float) -> str:
    deltas = compute_trend(entries, window, compare_window)
    if not deltas:
        return "# Weekly Latency Trend Report\n\nNot enough history to compute trends.\n"

    lines: List[str] = []
    lines.append("# Weekly Latency Trend Report")
    lines.append("")
    lines.append(f"Current window: {window} snapshots; previous window: {compare_window} snapshots")
    lines.append("")
    lines.append("| Provider | CRIT Δ | WARN Δ |")
    lines.append("|----------|---------|---------|")

    flagged = False
    for provider, delta_map in sorted(deltas.items()):
        crit_delta = delta_map["CRIT"]
        warn_delta = delta_map["WARN"]
        if abs(crit_delta) >= min_delta or abs(warn_delta) >= min_delta:
            flagged = True
            lines.append(f"| {provider} | {crit_delta:+} | {warn_delta:+} |")

    if not flagged:
        lines.append("| (none) | 0 | 0 |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weekly latency trend report.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_alert_history.jsonl"),
        help="History JSONL produced by provider_latency_alert_digest.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_weekly_trends.md"),
        help="Markdown report output.",
    )
    parser.add_argument("--window", type=int, default=7, help="Snapshots in the current window (default 7).")
    parser.add_argument("--compare-window", type=int, default=7, help="Snapshots in the comparison window (default 7).")
    parser.add_argument("--min-delta", type=float, default=1.0, help="Minimum delta to flag (default 1 alert).")
    args = parser.parse_args()

    entries = load_history(args.history)
    report = build_report(entries, args.window, args.compare_window, args.min_delta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"[info] Wrote weekly trend report to {args.output}")


if __name__ == "__main__":
    main()
