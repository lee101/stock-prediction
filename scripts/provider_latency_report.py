#!/usr/bin/env python3
"""Summarise provider latency observations from trend fetch runs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


@dataclass
class LatencySample:
    timestamp: datetime
    symbol: str
    provider: str
    latency_ms: float


def load_latency(path: Path) -> List[LatencySample]:
    if not path.exists():
        raise FileNotFoundError(f"latency log not found: {path}")
    rows: List[LatencySample] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            try:
                timestamp = datetime.fromisoformat(raw["timestamp"])
                symbol = raw["symbol"].upper()
                provider = raw["provider"]
                latency_ms = float(raw["latency_ms"])
            except (ValueError, KeyError) as exc:
                raise ValueError(f"invalid row in latency log: {raw}") from exc
            rows.append(
                LatencySample(
                    timestamp=timestamp,
                    symbol=symbol,
                    provider=provider,
                    latency_ms=latency_ms,
                )
            )
    rows.sort(key=lambda item: item.timestamp)
    return rows


def render_summary(
    samples: List[LatencySample],
    p95_threshold: float | None = None,
) -> str:
    if not samples:
        return "No latency samples available."
    per_provider: Dict[str, List[float]] = defaultdict(list)
    for sample in samples:
        per_provider[sample.provider].append(sample.latency_ms)

    lines: List[str] = []
    lines.append(f"Total samples: {len(samples)}")
    lines.append("Provider latency stats (ms):")
    alerts: List[Tuple[str, float]] = []
    for provider, values in sorted(per_provider.items()):
        lines.append(
            f"- {provider}: avg={mean(values):.2f} ms, "
            f"p50={percentile(values, 50):.2f} ms, "
            f"p95={percentile(values, 95):.2f} ms, "
            f"max={max(values):.2f} ms (n={len(values)})"
        )
        if p95_threshold is not None:
            p95 = percentile(values, 95)
            if p95 > p95_threshold:
                alerts.append((provider, p95))
    latest = samples[-1]
    lines.append(
        f"Latest sample: {latest.timestamp.isoformat()} {latest.symbol}@{latest.provider} "
        f"latency={latest.latency_ms:.2f} ms"
    )
    if alerts:
        lines.append("Alerts:")
        for provider, p95 in alerts:
            lines.append(
                f"[alert] {provider} p95 latency {p95:.2f} ms exceeds threshold {p95_threshold:.2f} ms"
            )
    return "\n".join(lines)


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise provider latency log.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency.csv"),
        help="Latency log CSV (timestamp,symbol,provider,latency_ms).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_summary.txt"),
        help="Where to write summary text.",
    )
    parser.add_argument(
        "--rollup-csv",
        type=Path,
        default=None,
        help="Optional CSV file to append aggregated latency stats per provider per run.",
    )
    parser.add_argument(
        "--p95-threshold",
        type=float,
        default=None,
        help="Emit alerts when provider p95 latency exceeds this threshold (ms).",
    )
    args = parser.parse_args()

    samples = load_latency(args.log)
    summary = render_summary(samples, args.p95_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")
    print(summary)

    if args.rollup_csv:
        args.rollup_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.rollup_csv.exists()
        per_provider: Dict[str, List[float]] = defaultdict(list)
        for sample in samples:
            per_provider[sample.provider].append(sample.latency_ms)
        timestamp = samples[-1].timestamp.isoformat()
        with args.rollup_csv.open("a", encoding="utf-8") as handle:
            if write_header:
                handle.write("timestamp,provider,avg_ms,p50_ms,p95_ms,max_ms,count\n")
            for provider, values in sorted(per_provider.items()):
                handle.write(
                    f"{timestamp},{provider},{mean(values):.3f},{percentile(values,50):.3f},"
                    f"{percentile(values,95):.3f},{max(values):.3f},{len(values)}\n"
                )


if __name__ == "__main__":
    main()
