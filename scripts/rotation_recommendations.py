#!/usr/bin/env python3
"""Recommend symbol rotations based on paused streaks and trend summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_paused_log(path: Path) -> Dict[str, Dict[str, object]]:
    latest: Dict[str, Dict[str, object]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symbol = row.get("symbol", "").upper()
            streak = int(row.get("streak") or 0)
            timestamp = row.get("timestamp", "")
            pnl_raw = row.get("pnl")
            pnl = float(pnl_raw) if pnl_raw not in (None, "",) else float("nan")
            latest[symbol] = {
                "streak": streak,
                "timestamp": timestamp,
                "pnl": pnl,
            }
    return latest


def load_trend_summary(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Trend summary not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {k.upper(): v for k, v in data.items() if k.upper() != "__OVERALL__"}


def pick_candidates(summary: Dict[str, Dict[str, float]], min_sma: float, limit: int = 5) -> List[Tuple[str, float, float]]:
    eligible = []
    for symbol, stats in summary.items():
        sma = float(stats.get("sma", 0.0) or 0.0)
        pnl = float(stats.get("pnl", 0.0) or 0.0)
        if sma >= min_sma:
            eligible.append((symbol, sma, pnl))
    eligible.sort(key=lambda item: item[1], reverse=True)
    return eligible[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate symbol rotation suggestions.")
    parser.add_argument(
        "--paused-log",
        type=Path,
        default=Path("marketsimulator/run_logs/trend_paused_escalations.csv"),
        help="Path to paused escalation CSV log.",
    )
    parser.add_argument(
        "--trend-summary",
        type=Path,
        default=Path("marketsimulator/run_logs/trend_summary.json"),
        help="Path to trend summary JSON.",
    )
    parser.add_argument(
        "--streak-threshold",
        type=int,
        default=8,
        help="Minimum paused streak to recommend removal (default 8).",
    )
    parser.add_argument(
        "--candidate-sma",
        type=float,
        default=500.0,
        help="Minimum SMA to surface candidate additions (default 500).",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=None,
        help="Optional path to append recommendations as text (for audit trail).",
    )
    args = parser.parse_args()

    paused_info = load_paused_log(args.paused_log)
    trend_summary = load_trend_summary(args.trend_summary)

    removals = [
        (symbol, info["streak"], info["pnl"], info["timestamp"])
        for symbol, info in paused_info.items()
        if info.get("streak", 0) >= args.streak_threshold
    ]
    removals.sort(key=lambda item: item[1], reverse=True)

    candidates = pick_candidates(trend_summary, args.candidate_sma)

    if removals:
        print("Recommended removals (paused streak ≥ threshold):")
        print("Symbol | Streak | Trend PnL | Last Escalation")
        print("-------|--------|-----------|-------------------------")
        for symbol, streak, pnl, timestamp in removals:
            pnl_str = "nan" if pnl != pnl else f"{pnl:.2f}"
            print(f"{symbol:>6} | {streak:>6} | {pnl_str:>9} | {timestamp}")
    else:
        print("[info] No symbols exceeded the paused streak threshold.")

    if candidates:
        print("\nCandidate additions (SMA ≥ %.1f):" % args.candidate_sma)
        print("Symbol | SMA      | Trend PnL | % Change")
        print("-------|----------|-----------|----------")
        for symbol, sma, pnl in candidates:
            pct = trend_summary.get(symbol.upper(), {}).get("pct_change", float("nan"))
            pct_str = f"{pct*100:>8.2f}%" if pct == pct else "   n/a   "
            print(f"{symbol:>6} | {sma:>8.2f} | {pnl:>9.2f} | {pct_str}")
    else:
        print("\n[info] No candidate symbols meet the SMA threshold (%.1f)." % args.candidate_sma)

    if args.log_output:
        from datetime import datetime, timezone

        args.log_output.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.log_output.exists()
        now_iso = datetime.now(timezone.utc).isoformat()
        with args.log_output.open("a", encoding="utf-8") as handle:
            if write_header:
                handle.write("timestamp,symbol,type,detail\n")
            if removals:
                for symbol, streak, pnl, timestamp in removals:
                    pnl_str = "" if pnl != pnl else f"{pnl:.2f}"
                    handle.write(
                        f"{now_iso},{symbol},removal,streak={streak};trend_pnl={pnl_str};last_escalation={timestamp}\n"
                    )
            if candidates:
                for symbol, sma, pnl in candidates:
                    pct = trend_summary.get(symbol.upper(), {}).get("pct_change", float("nan"))
                    detail = f"sma={sma:.2f};trend_pnl={pnl:.2f}"
                    if pct == pct:
                        detail += f";pct_change={pct*100:.2f}%"
                    handle.write(f"{now_iso},{symbol},candidate,{detail}\n")

if __name__ == "__main__":
    main()
