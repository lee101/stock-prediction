#!/usr/bin/env python3
"""Summarise provider usage history for ETF trend fetches."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


@dataclass
class ProviderUsage:
    timestamp: datetime
    provider: str
    count: int


def load_usage(path: Path) -> List[ProviderUsage]:
    if not path.exists():
        raise FileNotFoundError(f"provider usage log not found: {path}")
    rows: List[ProviderUsage] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            try:
                ts = datetime.fromisoformat(raw["timestamp"])
                provider = (raw["provider"] or "").strip()
                count = int(raw["count"])
            except (ValueError, KeyError) as exc:
                raise ValueError(f"invalid row in provider usage log: {raw}") from exc
            rows.append(ProviderUsage(timestamp=ts, provider=provider, count=count))
    rows.sort(key=lambda item: item.timestamp)
    return rows


def build_timeline(rows: Iterable[ProviderUsage], window: int) -> str:
    tail = list(rows)[-window:] if window > 0 else list(rows)
    return "".join(entry.provider[:1].upper() or "?" for entry in tail)


def render_report(rows: List[ProviderUsage], timeline_window: int, sparkline: bool) -> str:
    lines: List[str] = []
    lines.append(f"Total runs: {len(rows)}")
    counts = Counter(entry.provider for entry in rows)
    if counts:
        lines.append("Provider totals:")
        for provider, total in counts.most_common():
            last_seen = max(entry.timestamp for entry in rows if entry.provider == provider)
            lines.append(f"- {provider or 'unknown'}: {total} runs (last {last_seen.isoformat()})")
    if sparkline and rows:
        timeline = build_timeline(rows, timeline_window)
        lines.append(f"Timeline (last {timeline_window if timeline_window else 'all'}): {timeline}")
    if rows:
        latest = rows[-1]
        lines.append(
            "Latest run: "
            f"{latest.timestamp.isoformat()} provider={latest.provider or 'unknown'} count={latest.count}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise provider usage history.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_usage.csv"),
        help="Provider usage CSV produced by fetch_etf_trends.py",
    )
    parser.add_argument(
        "--timeline-window",
        type=int,
        default=20,
        help="Number of rows to include in the timeline (0 = all).",
    )
    parser.add_argument(
        "--no-sparkline",
        action="store_true",
        help="Disable timeline sparkline output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the summary text to disk.",
    )
    args = parser.parse_args()

    rows = load_usage(args.log)
    report = render_report(rows, args.timeline_window, not args.no_sparkline)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
