#!/usr/bin/env python3
"""Generate a markdown leaderboard of top trending symbols from trend_summary.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def load_summary(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Trend summary not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {k.upper(): v for k, v in data.items() if k.upper() != "__OVERALL__"}


def pick_leaders(summary: Dict[str, Dict[str, float]], top_n: int) -> List[Tuple[str, float, float, float]]:
    rows: List[Tuple[str, float, float, float]] = []
    for symbol, stats in summary.items():
        sma = float(stats.get("sma", 0.0) or 0.0)
        pnl = float(stats.get("pnl", 0.0) or 0.0)
        pct = float(stats.get("pct_change", 0.0) or 0.0)
        rows.append((symbol, sma, pnl, pct))
    rows.sort(key=lambda item: (item[1], item[3], item[2]), reverse=True)
    return rows[:top_n]


def generate_markdown(leaders: List[Tuple[str, float, float, float]]) -> str:
    timestamp = datetime.now(timezone.utc).isoformat()
    lines: List[str] = []
    lines.append(f"# Trend Leaderboard — {timestamp}")
    lines.append("")
    if not leaders:
        lines.append("No symbols in trend summary.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Rank | Symbol | SMA | Trend ΔPnL | % Change |")
    lines.append("|------|--------|-----|------------|----------|")
    for idx, (symbol, sma, pnl, pct) in enumerate(leaders, start=1):
        lines.append(f"| {idx} | {symbol} | {sma:>8.2f} | {pnl:>10.2f} | {pct*100:>8.2f}% |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a trend leaderboard markdown table.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("marketsimulator/run_logs/trend_summary.json"),
        help="Path to trend summary JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marketsimulator/run_logs/trend_leaderboard.md"),
        help="Markdown output path.",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of symbols to show (default 10).")
    args = parser.parse_args()

    summary = load_summary(args.summary_path)
    leaders = pick_leaders(summary, args.top)
    markdown = generate_markdown(leaders)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"[info] Leaderboard written to {args.output}")


if __name__ == "__main__":
    main()
