#!/usr/bin/env python3
"""Generate a compact Markdown sparkline for provider usage history."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from scripts.provider_usage_report import load_usage


def build_tokens(providers: List[str], token_map: Dict[str, str]) -> str:
    return "".join(token_map.get(provider, token_map.get("__default__", "?")) for provider in providers)


def default_token_map() -> Dict[str, str]:
    return {
        "yahoo": "🟦",
        "stooq": "🟥",
        "__default__": "⬛",
    }


def render_markdown(log_path: Path, window: int, token_map: Dict[str, str]) -> str:
    rows = load_usage(log_path)
    if not rows:
        return "# Provider Usage Sparkline\n\n_No provider usage data available._\n"
    tail = rows[-window:] if window > 0 else rows
    providers = [entry.provider for entry in tail]
    tokens = build_tokens(providers, token_map)
    timestamps = [entry.timestamp for entry in tail]
    latest = tail[-1]
    lines: List[str] = []
    lines.append("# Provider Usage Sparkline")
    lines.append("")
    lines.append(f"Window: last {window if window else len(rows)} runs")
    lines.append("")
    lines.append(f"Sparkline: {tokens}")
    lines.append("")
    lines.append("| Run | Timestamp (UTC) | Provider | Count | Token |")
    lines.append("|-----|-----------------|----------|-------|-------|")
    for idx, entry in enumerate(tail, start=max(len(rows) - len(tail) + 1, 1)):
        token = token_map.get(entry.provider, token_map.get("__default__", "?"))
        lines.append(
            f"| {idx} | {entry.timestamp.isoformat()} | {entry.provider or 'unknown'} | "
            f"{entry.count} | {token} |"
        )
    lines.append("")
    lines.append(
        f"Latest: {latest.timestamp.isoformat()} provider={latest.provider or 'unknown'} count={latest.count}"
    )
    lines.append("")
    legend_tokens = {
        provider: token for provider, token in token_map.items() if provider != "__default__"
    }
    if legend_tokens:
        lines.append("Legend:")
        for provider, token in legend_tokens.items():
            lines.append(f"- {token} = {provider}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Markdown sparkline for provider usage.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_usage.csv"),
        help="Provider usage CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_usage_sparkline.md"),
        help="Markdown output file.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Number of runs to include (0 = all).",
    )
    args = parser.parse_args()

    token_map = default_token_map()
    markdown = render_markdown(args.log, args.window, token_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
