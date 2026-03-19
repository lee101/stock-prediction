#!/usr/bin/env python3
"""Generate rolling latency statistics from provider latency rollup CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Deque, Dict, List


@dataclass
class RollupRow:
    timestamp: str
    provider: str
    avg_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float
    count: int


def load_rollup(path: Path) -> List[RollupRow]:
    if not path.exists():
        raise FileNotFoundError(f"Rollup CSV not found: {path}")
    rows: List[RollupRow] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(
                RollupRow(
                    timestamp=raw["timestamp"],
                    provider=raw["provider"],
                    avg_ms=float(raw["avg_ms"]),
                    p50_ms=float(raw["p50_ms"]),
                    p95_ms=float(raw["p95_ms"]),
                    max_ms=float(raw["max_ms"]),
                    count=int(raw["count"]),
                )
            )
    rows.sort(key=lambda item: item.timestamp)
    return rows


def compute_rolling(rows: List[RollupRow], window: int) -> Dict[str, Dict[str, float | None]]:
    window = max(window, 1)
    buckets: Dict[str, Deque[RollupRow]] = defaultdict(deque)
    aggregates: Dict[str, Dict[str, float | None]] = {}
    previous: Dict[str, Dict[str, float]] = {}
    for row in rows:
        dq = buckets[row.provider]
        dq.append(row)
        if len(dq) > window:
            dq.popleft()
        avg_avg = mean(item.avg_ms for item in dq)
        avg_p95 = mean(item.p95_ms for item in dq)
        prev_stats = previous.get(row.provider)
        delta_avg = avg_avg - prev_stats["avg"] if prev_stats else None
        delta_p95 = avg_p95 - prev_stats["p95"] if prev_stats else None
        aggregates[row.provider] = {
            "window": len(dq),
            "avg_ms": avg_avg,
            "p95_ms": avg_p95,
            "latest_timestamp": row.timestamp,
            "delta_avg_ms": delta_avg,
            "delta_p95_ms": delta_p95,
        }
        previous[row.provider] = {"avg": avg_avg, "p95": avg_p95}
    return aggregates


def render_markdown(aggregates: Dict[str, Dict[str, float | None]], window: int) -> str:
    if not aggregates:
        return "# Rolling Provider Latency\n\n_No rollup data available._\n"
    lines: List[str] = []
    lines.append("# Rolling Provider Latency")
    lines.append("")
    lines.append(f"Window size: {window}")
    lines.append("")
    lines.append(
        "| Provider | Samples | Avg Latency (ms) | ΔAvg (ms) | P95 Latency (ms) | ΔP95 (ms) | Last Timestamp |"
    )
    lines.append(
        "|----------|---------|------------------|----------|------------------|----------|----------------|"
    )
    for provider, stats in sorted(aggregates.items()):
        delta_avg = stats.get("delta_avg_ms")
        delta_p95 = stats.get("delta_p95_ms")
        delta_avg_str = f"{delta_avg:+.2f}" if delta_avg is not None else "–"
        delta_p95_str = f"{delta_p95:+.2f}" if delta_p95 is not None else "–"
        lines.append(
            f"| {provider} | {stats['window']} | {stats['avg_ms']:.2f} | {delta_avg_str} | "
            f"{stats['p95_ms']:.2f} | {delta_p95_str} | {stats['latest_timestamp']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute rolling provider latency stats.")
    parser.add_argument(
        "--rollup",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_rollup.csv"),
        help="Latency rollup CSV produced by provider_latency_report.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_rolling.md"),
        help="Markdown output file.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Rolling window size (number of runs).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write aggregates as JSON for downstream comparisons.",
    )
    parser.add_argument(
        "--history-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file to append rolling snapshots (timestamp + aggregates).",
    )
    args = parser.parse_args()

    rows = load_rollup(args.rollup)
    aggregates = compute_rolling(rows, args.window)
    markdown = render_markdown(aggregates, args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(markdown)

    timestamp = None
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            provider: {k: v for k, v in stats.items()} for provider, stats in aggregates.items()
        }
        args.json_output.write_text(json.dumps(serialisable, indent=2, sort_keys=True), encoding="utf-8")
        timestamp = datetime.now(timezone.utc).isoformat()

    if args.history_jsonl:
        args.history_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        payload = {
            "timestamp": timestamp,
            "window": args.window,
            "aggregates": {
                provider: {k: v for k, v in stats.items()} for provider, stats in aggregates.items()
            },
        }
        with args.history_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
