#!/usr/bin/env python3
"""Render a leaderboard from provider_latency_alert_history.jsonl."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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


def _aggregate(entries: List[Dict[str, object]]) -> Dict[str, Counter]:
    aggregate: Dict[str, Counter] = defaultdict(Counter)
    for entry in entries:
        provider_map = entry.get("provider_severity", {})
        for provider, data in provider_map.items():
            for severity, value in data.items():
                aggregate[provider.upper()][severity.upper()] += value
    return aggregate


def build_leaderboard(
    entries: List[Dict[str, object]],
    window: int,
    compare_window: int | None = None,
    rate_window: int | None = None,
) -> str:
    if not entries:
        return "# Latency Alert Leaderboard\n\nNo history available.\n"
    tail = entries[-window:] if window > 0 else entries
    compare_window = compare_window or window
    prev_tail = []
    if compare_window and len(entries) > len(tail):
        prev_tail = entries[-(window + compare_window) : -window]
    counts = _aggregate(tail)
    prev_counts = _aggregate(prev_tail)
    rate_window = rate_window or window
    lines: List[str] = []
    lines.append("# Latency Alert Leaderboard")
    lines.append("")
    lines.append(f"Window: last {len(tail)} snapshots")
    lines.append("")
    if prev_tail:
        lines.append("| Provider | INFO | WARN | CRIT | Total | ΔTotal | CRIT% (Δ) | WARN% (Δ) |")
        lines.append("|----------|------|------|------|-------|--------|------------|-------------|")
    else:
        lines.append("| Provider | INFO | WARN | CRIT | Total | CRIT% | WARN% |")
        lines.append("|----------|------|------|------|-------|-------|-------|")
    for provider, counter in sorted(
        counts.items(),
        key=lambda item: (item[1]["CRIT"], item[1]["WARN"], item[1]["INFO"]),
        reverse=True,
    ):
        total = sum(counter.values())
        crit_count = counter.get("CRIT", 0)
        warn_count = counter.get("WARN", 0)
        crit_pct = (crit_count / total * 100) if total else 0.0
        warn_pct = (warn_count / total * 100) if total else 0.0
        if prev_tail:
            prev_total = sum(prev_counts.get(provider, {}).values())
            delta = total - prev_total
            prev_total_nonzero = prev_total if prev_total else 1
            prev_crit_pct = (
                prev_counts.get(provider, {}).get("CRIT", 0) / prev_total_nonzero * 100
            ) if prev_total else 0.0
            prev_warn_pct = (
                prev_counts.get(provider, {}).get("WARN", 0) / prev_total_nonzero * 100
            ) if prev_total else 0.0
            lines.append(
                f"| {provider} | {counter.get('INFO', 0)} | {warn_count} | {crit_count} | {total} | {delta:+d} | {crit_pct:.1f}% ({crit_pct - prev_crit_pct:+.1f}) | {warn_pct:.1f}% ({warn_pct - prev_warn_pct:+.1f}) |"
            )
        else:
            lines.append(
                f"| {provider} | {counter.get('INFO', 0)} | {warn_count} | {crit_count} | {total} | {crit_pct:.1f}% | {warn_pct:.1f}% |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render latency alert leaderboard.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_alert_history.jsonl"),
        help="JSONL history produced by provider_latency_alert_digest.py --history",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_leaderboard.md"),
        help="Markdown output path",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Number of snapshots to include (0=all)",
    )
    parser.add_argument(
        "--compare-window",
        type=int,
        default=None,
        help="Snapshots to use for delta comparison (default = window)",
    )
    args = parser.parse_args()

    entries = load_history(args.history)
    leaderboard = build_leaderboard(entries, args.window, compare_window=args.compare_window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(leaderboard, encoding="utf-8")
    print(f"[info] Wrote leaderboard to {args.output}")


if __name__ == "__main__":
    main()
