#!/usr/bin/env python3
"""Render latency history trends from provider_latency_rolling_history.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def load_history(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"latency history not found: {path}")
    entries: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row: {line[:80]}") from exc
    entries.sort(key=lambda item: item["timestamp"])
    return entries


def normalize(values: List[float]) -> List[int]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if v_max == v_min:
        return [len(SPARK_CHARS) // 2 for _ in values]
    return [int((val - v_min) / (v_max - v_min) * (len(SPARK_CHARS) - 1)) for val in values]


def render_history(entries: List[Dict[str, object]], window: int) -> str:
    if not entries:
        return "# Provider Latency History\n\n_No history available._\n"
    providers = sorted(entries[-1]["aggregates"].keys())
    lines: List[str] = []
    lines.append("# Provider Latency History")
    lines.append("")
    lines.append(f"Window: last {window} snapshots")
    lines.append("")
    lines.append("| Provider | Sparkline | Latest Avg (ms) | Latest ΔAvg (ms) | Latest P95 (ms) | Latest ΔP95 (ms) |")
    lines.append("|----------|-----------|-----------------|------------------|-----------------|------------------|")
    tail = entries[-window:] if window > 0 else entries
    for provider in providers:
        series = [snap["aggregates"].get(provider, {}).get("avg_ms") for snap in tail]
        series = [val for val in series if val is not None]
        if not series:
            continue
        norm = normalize(series)
        spark = "".join(SPARK_CHARS[idx] for idx in norm)
        latest = tail[-1]["aggregates"].get(provider, {})
        lines.append(
            f"| {provider} | {spark or 'n/a'} | "
            f"{latest.get('avg_ms', float('nan')):.2f} | "
            f"{latest.get('delta_avg_ms', 0.0):+,.2f} | "
            f"{latest.get('p95_ms', float('nan')):.2f} | "
            f"{latest.get('delta_p95_ms', 0.0):+,.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render provider latency history trends.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_rolling_history.jsonl"),
        help="JSONL file populated by provider_latency_rolling.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_history.md"),
        help="Markdown output file.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Number of snapshots to include (0 = all).",
    )
    args = parser.parse_args()

    entries = load_history(args.history)
    markdown = render_history(entries, args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
