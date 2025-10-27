#!/usr/bin/env python3
"""Render rotation recommendations into a markdown summary."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from scripts.provider_latency_status import evaluate


def load_log(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Rotation recommendations log not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def render_markdown(
    rows: List[Dict[str, str]],
    streak_threshold: int,
    latency_snapshot: Dict[str, Dict[str, float]] | None,
    latency_png: Path | None,
    latency_digest: Path | None,
    latency_leaderboard: Path | None,
) -> str:
    removal_rows = [row for row in rows if row.get("type") == "removal"]
    candidate_rows = [row for row in rows if row.get("type") == "candidate"]

    latest_per_symbol: Dict[str, Dict[str, str]] = {}
    for row in rows:
        symbol = row.get("symbol", "").upper()
        latest_per_symbol[symbol] = row

    lines: List[str] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    lines.append(f"# Rotation Summary — {now_iso}")
    lines.append("")

    if removal_rows:
        lines.append("## Recommended Removals")
        lines.append("")
        lines.append("| Symbol | Streak | Trend PnL | Last Escalation | Note |")
        lines.append("|--------|--------|-----------|-----------------|------|")
        for row in removal_rows:
            symbol = row.get("symbol", "").upper()
            detail = row.get("detail", "")
            parts = dict(part.split("=", 1) for part in detail.split(";") if "=" in part)
            streak = parts.get("streak", "?")
            pnl = parts.get("trend_pnl", "?")
            last = parts.get("last_escalation", row.get("timestamp", ""))
            note = "Exceeded threshold" if int(streak) >= streak_threshold else ""
            lines.append(f"| {symbol} | {streak} | {pnl} | {last} | {note} |")
        lines.append("")
    else:
        lines.append("## Recommended Removals")
        lines.append("")
        lines.append("None at the moment.")
        lines.append("")

    if candidate_rows:
        lines.append("## Candidate Additions")
        lines.append("")
        lines.append("| Symbol | SMA | Trend PnL | Logged At |")
        lines.append("|--------|-----|-----------|-----------|")
        # keep last unique per symbol
        seen: Dict[str, Dict[str, str]] = {}
        for row in candidate_rows:
            symbol = row.get("symbol", "").upper()
            seen[symbol] = row
        for symbol, row in seen.items():
            detail = row.get("detail", "")
            parts = dict(part.split("=", 1) for part in detail.split(";") if "=" in part)
            sma = parts.get("sma", "?")
            pnl = parts.get("trend_pnl", "?")
            ts = row.get("timestamp", "")
            lines.append(f"| {symbol} | {sma} | {pnl} | {ts} |")
        lines.append("")
    else:
        lines.append("## Candidate Additions")
        lines.append("")
        lines.append("No candidates currently exceed the SMA threshold.")
    lines.append("")

    if latency_snapshot:
        status, _status_details = evaluate(latency_snapshot, warn_threshold=20.0, crit_threshold=40.0)
        lines.append(f"**Latency Status:** {status}")
        lines.append("")
        lines.append("## Data Feed Health")
        lines.append("")
        lines.append("| Provider | Rolling Avg (ms) | ΔAvg (ms) | Rolling P95 (ms) | ΔP95 (ms) |")
        lines.append("|----------|------------------|-----------|------------------|-----------|")
        for provider, stats in sorted(latency_snapshot.items()):
            avg = stats.get("avg_ms", float("nan"))
            delta_avg = stats.get("delta_avg_ms")
            p95 = stats.get("p95_ms", float("nan"))
            delta_p95 = stats.get("delta_p95_ms")
            lines.append(
                f"| {provider} | {avg:.2f} | "
                f"{delta_avg:+.2f} | {p95:.2f} | {delta_p95:+.2f} |"
            )
        lines.append("")
        if any(abs(stats.get("delta_avg_ms", 0.0)) >= 40.0 for stats in latency_snapshot.values()):
            lines.append("- ⚠️ Rolling latency shift exceeds 40 ms for at least one provider. Investigate data feed stability.")
        else:
            lines.append("- ✅ Rolling latency shifts are within the 40 ms tolerance.")
        lines.append("")
        if latency_png:
            lines.append(f"![Latency History Thumbnail]({latency_png.as_posix()})")
            lines.append("")

    if latency_digest and latency_digest.exists():
        digest_text = latency_digest.read_text(encoding="utf-8").strip()
        if digest_text:
            lines.append("## Recent Latency Alerts")
            lines.append("")
            preview_lines = digest_text.splitlines()
            snippet = preview_lines[:20]
            lines.extend(snippet)
            if len(preview_lines) > len(snippet):
                lines.append("...")
            lines.append("")
    if latency_leaderboard and latency_leaderboard.exists():
        board_lines = latency_leaderboard.read_text(encoding="utf-8").splitlines()
        header_idx = next((i for i, line in enumerate(board_lines) if line.startswith("| Provider")), None)
        if header_idx is not None:
            lines.append("## Latency Offenders Leaderboard")
            lines.append("")
            preview = board_lines[header_idx : header_idx + 5]
            lines.extend(preview)
            lines.append("")

    lines.append("## Next Actions")
    lines.append("")
    if removal_rows:
        symbols = ", ".join(sorted({row.get("symbol", "").upper() for row in removal_rows}))
        lines.append(f"- Consider removing **{symbols}** from active rosters until trend PnL recovers.")
    else:
        lines.append("- Continue monitoring trends; no removals are recommended.")

    if candidate_rows:
        lines.append(
            "- Review candidates above the SMA threshold and stage onboarding experiments with directional gating."
        )
    else:
        lines.append("- No SMA-qualified candidates; continue scanning the universe for replacements.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render rotation log into markdown.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("marketsimulator/run_logs/rotation_recommendations.log"),
        help="Rotation recommendations log file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marketsimulator/run_logs/rotation_summary.md"),
        help="Markdown output file.",
    )
    parser.add_argument(
        "--streak-threshold",
        type=int,
        default=8,
        help="Paused streak threshold (for annotations).",
    )
    parser.add_argument(
        "--latency-json",
        type=Path,
        default=None,
        help="Optional provider latency snapshot JSON (from provider_latency_rolling.json).",
    )
    parser.add_argument(
        "--latency-png",
        type=Path,
        default=None,
        help="Optional latency thumbnail PNG to embed (from provider_latency_history.png).",
    )
    parser.add_argument(
        "--latency-digest",
        type=Path,
        default=None,
        help="Optional latency alert digest markdown to embed preview from.",
    )
    parser.add_argument(
        "--latency-leaderboard",
        type=Path,
        default=None,
        help="Optional latency leaderboard markdown to embed.",
    )
    args = parser.parse_args()

    rows = load_log(args.input)
    if not rows:
        print("[info] rotation log is empty; nothing to summarise.")
        return

    snapshot: Dict[str, Dict[str, float]] | None = None
    if args.latency_json and args.latency_json.exists():
        try:
            snapshot = json.loads(args.latency_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[warn] Failed to parse latency snapshot {args.latency_json}: {exc}")
            snapshot = None

    markdown = render_markdown(
        rows,
        args.streak_threshold,
        snapshot,
        args.latency_png,
        args.latency_digest,
        args.latency_leaderboard,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"[info] rotation summary written to {args.output}")


if __name__ == "__main__":
    main()
