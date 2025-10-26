#!/usr/bin/env python3
"""Summarize ETF readiness for onboarding based on trend metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def load_summary(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Trend summary not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {k.upper(): v for k, v in data.items() if k.upper() != "__OVERALL__"}


def filter_candidates(
    summary: Dict[str, Dict[str, float]], min_sma: float, min_pct: float
) -> List[Dict[str, float]]:
    candidates: List[Dict[str, float]] = []
    for symbol, stats in summary.items():
        sma = float(stats.get("sma", 0.0) or 0.0)
        pct = float(stats.get("pct_change", 0.0) or 0.0)
        pnl = float(stats.get("pnl", 0.0) or 0.0)
        if sma >= min_sma and pct >= min_pct:
            candidates.append(
                {
                    "symbol": symbol,
                    "sma": sma,
                    "pct_change": pct,
                    "pnl": pnl,
                }
            )
    candidates.sort(key=lambda item: (item["pct_change"], item["sma"]), reverse=True)
    return candidates


def render_markdown(candidates: List[Dict[str, float]], min_sma: float, min_pct: float) -> str:
    timestamp = datetime.now(timezone.utc).isoformat()
    lines: List[str] = []
    lines.append(f"# Candidate Readiness — {timestamp}")
    lines.append("")
    lines.append(f"- Filters: SMA ≥ {min_sma:.1f}, % change ≥ {min_pct * 100:.1f}%")
    lines.append("")
    if not candidates:
        lines.append("No symbols meet the readiness filters. Keep monitoring trend updates.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Symbol | SMA | Trend ΔPnL | % Change | Note |")
    lines.append("|--------|-----|------------|----------|------|")
    for item in candidates:
        symbol = item["symbol"]
        sma = item["sma"]
        pnl = item["pnl"]
        pct = item["pct_change"]
        note = "High SMA + positive momentum"
        lines.append(f"| {symbol} | {sma:>8.2f} | {pnl:>10.2f} | {pct*100:>8.2f}% | {note} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a candidate readiness report.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("marketsimulator/run_logs/trend_summary.json"),
        help="Path to trend summary JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marketsimulator/run_logs/candidate_readiness.md"),
        help="Markdown output path.",
    )
    parser.add_argument("--min-sma", type=float, default=200.0, help="Minimum SMA threshold.")
    parser.add_argument(
        "--min-pct", type=float, default=0.0, help="Minimum fractional pct change (e.g., 0.1 = 10%)."
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV file to append per-run readiness rows.",
    )
    args = parser.parse_args()

    summary = load_summary(args.summary_path)
    candidates = filter_candidates(summary, args.min_sma, args.min_pct)
    markdown = render_markdown(candidates, args.min_sma, args.min_pct)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"[info] Candidate readiness written to {args.output}")

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.csv_output.exists()
        timestamp = datetime.now(timezone.utc).isoformat()
        with args.csv_output.open("a", encoding="utf-8") as handle:
            if write_header:
                handle.write("timestamp,symbol,sma,pct_change,trend_pnl\n")
            for item in candidates:
                handle.write(
                    f"{timestamp},{item['symbol']},{item['sma']:.4f},{item['pct_change']:.6f},{item['pnl']:.4f}\n"
                )


if __name__ == "__main__":
    main()
