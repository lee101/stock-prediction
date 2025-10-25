#!/usr/bin/env python3
"""Summarise latency alert log into a markdown digest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def load_alerts(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        return []
    alerts: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            timestamp_str, message = line.split(" ", 1)
        except ValueError:
            continue
        alerts.append((timestamp_str, message.strip()))
    return alerts


def summarise_details(alerts: List[Tuple[str, str]]) -> Tuple[str, Dict[str, Counter], Counter]:
    if not alerts:
        return "# Latency Alert Digest\n\nNo alerts recorded.\n", defaultdict(Counter), Counter()

    per_day: Dict[str, Counter] = defaultdict(Counter)
    providers: Counter = Counter()
    severity_counter: Counter = Counter()
    provider_severity: Dict[str, Counter] = defaultdict(Counter)
    for timestamp_str, message in alerts:
        try:
            ts = datetime.fromisoformat(timestamp_str)
            day = ts.date().isoformat()
        except ValueError:
            day = "unknown"
        lower = message.lower()
        for token in lower.replace("\n", " ").split():
            if token.endswith("ms") and "latency" in lower:
                continue
        per_day[day][message] += 1
        words_raw = [w.strip(",.!:;()") for w in message.split()]
        uppercase_words = [word for word in words_raw if word.isupper() and len(word) >= 2]
        for word in uppercase_words:
            providers[word] += 1
        if "δ≥" in lower or "threshold" in lower:
            severity = "CRIT"
        elif "warn" in lower:
            severity = "WARN"
        else:
            severity = "INFO"
        severity_counter[severity] += 1
        targets = uppercase_words if uppercase_words else ["unknown"]
        for provider in targets:
            provider_severity[provider][severity] += 1
    lines: List[str] = []
    lines.append("# Latency Alert Digest")
    lines.append("")
    lines.append(f"Total alerts: {len(alerts)}")
    lines.append("")
    if severity_counter:
        lines.append("## Severity Counts")
        lines.append("")
        for sev, count in severity_counter.most_common():
            lines.append(f"- {sev}: {count}")
        lines.append("")
    lines.append("## Alerts by Day")
    lines.append("")
    for day in sorted(per_day.keys()):
        lines.append(f"### {day}")
        for message, count in per_day[day].most_common():
            lines.append(f"- {count} × {message}")
        lines.append("")
    if providers:
        lines.append("## Providers Mentioned")
        lines.append("")
        for provider, count in providers.most_common():
            lines.append(f"- {provider}: {count}")
        lines.append("")
        lines.append("### Provider Severity Breakdown")
        lines.append("")
        lines.append("| Provider | INFO | WARN | CRIT |")
        lines.append("|----------|------|------|------|")
        for provider, counts in sorted(provider_severity.items()):
            lines.append(
                f"| {provider} | {counts.get('INFO', 0)} | {counts.get('WARN', 0)} | {counts.get('CRIT', 0)} |"
            )
        lines.append("")
    return "\n".join(lines), provider_severity, severity_counter


def summarise(alerts: List[Tuple[str, str]]) -> str:
    text, _, _ = summarise_details(alerts)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise latency alert log.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_alerts.log"),
        help="Latency alert log path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_alert_digest.md"),
        help="Markdown digest output path.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Optional JSONL history file to append severity tallies.",
    )
    args = parser.parse_args()

    alerts = load_alerts(args.log)
    digest, provider_severity, severity_counter = summarise_details(alerts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(digest, encoding="utf-8")
    print(f"[info] Latency alert digest written to {args.output}")

    if args.history:
        args.history.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            "provider_severity": {p: dict(counts) for p, counts in provider_severity.items()},
            "severity_totals": dict(severity_counter),
        }
        with args.history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
