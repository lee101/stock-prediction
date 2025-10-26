#!/usr/bin/env python3
"""Report trend-based gating status for each symbol."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple


def parse_threshold_map(env_name: str) -> Dict[str, float]:
    raw = os.getenv(env_name)
    thresholds: Dict[str, float] = {}
    if not raw:
        return thresholds
    for item in raw.split(","):
        entry = item.strip()
        if not entry or ":" not in entry:
            continue
        key, value = entry.split(":", 1)
        try:
            thresholds[key.strip().upper()] = float(value)
        except ValueError:
            continue
    return thresholds


def load_summary(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Trend summary not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def evaluate(symbol: str, pnl: float, suspend: Dict[str, float], resume: Dict[str, float]) -> Tuple[str, str]:
    symbol_key = symbol.upper()
    suspend_threshold = suspend.get(symbol_key)
    resume_threshold = resume.get(symbol_key)
    suspended = suspend_threshold is not None and pnl <= suspend_threshold
    if suspended:
        return symbol, "suspended"

    resume_ready = resume_threshold is not None and pnl > resume_threshold
    if resume_ready:
        return symbol, "resume_ready"

    if resume_threshold is not None and pnl <= resume_threshold:
        return symbol, "paused"

    return symbol, "neutral"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report trend gating status")
    parser.add_argument(
        "summary",
        type=Path,
        nargs="?",
        default=Path("marketsimulator/run_logs/trend_summary.json"),
        help="Path to trend_summary.json (default: marketsimulator/run_logs/trend_summary.json)",
    )
    parser.add_argument(
        "--suspend-map",
        dest="suspend_map",
        default=None,
        help="Override suspend thresholds (format SYMBOL:value,...)",
    )
    parser.add_argument(
        "--resume-map",
        dest="resume_map",
        default=None,
        help="Override resume thresholds (format SYMBOL:value,...)",
    )
    parser.add_argument(
        "--alert",
        action="store_true",
        help="Emit resume alerts for symbols ready to trade",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="Optional JSON file to persist paused streak counts between runs.",
    )
    parser.add_argument(
        "--paused-threshold",
        type=int,
        default=5,
        help="Emit an escalation when a symbol remains paused for at least this many consecutive runs (default: 5).",
    )
    parser.add_argument(
        "--paused-log",
        type=Path,
        default=None,
        help="Optional CSV file to append paused-streak escalations (timestamp,symbol,streak,pnl).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print aggregate counts by status after listing symbols",
    )
    args = parser.parse_args()

    summary = load_summary(args.summary)
    if args.suspend_map:
        os.environ["MARKETSIM_TREND_PNL_SUSPEND_MAP"] = args.suspend_map
    if args.resume_map:
        os.environ["MARKETSIM_TREND_PNL_RESUME_MAP"] = args.resume_map
    suspend_map = parse_threshold_map("MARKETSIM_TREND_PNL_SUSPEND_MAP")
    resume_map = parse_threshold_map("MARKETSIM_TREND_PNL_RESUME_MAP")

    if not summary:
        print("[warn] No trend data available")
        return

    history_records: Dict[str, Dict[str, object]] = {}
    if args.history and args.history.exists():
        with args.history.open("r", encoding="utf-8") as handle:
            try:
                history_records = json.load(handle)
            except json.JSONDecodeError:
                history_records = {}

    print("Symbol  | Trend PnL |     Status | Paused Streak | Resume Streak")
    print("--------|-----------|------------|---------------|---------------")
    resume_alerts = []
    paused_alerts = []
    paused_streak_alerts = []
    resume_streak_alerts = []
    status_counts: Dict[str, int] = {}
    for symbol, stats in summary.items():
        if symbol.upper() == "__OVERALL__":
            continue
        pnl = float(stats.get("pnl", 0.0))
        _, status = evaluate(symbol, pnl, suspend_map, resume_map)
        symbol_key = symbol.upper()
        record = history_records.get(symbol_key, {})
        paused_streak = int(record.get("paused_streak", 0))
        resume_streak = int(record.get("resume_streak", 0))
        if status == "paused":
            paused_streak += 1
            paused_streak_alerts.append((symbol, paused_streak))
        else:
            paused_streak = 0
        if status == "resume_ready":
            resume_streak += 1
            resume_streak_alerts.append((symbol, resume_streak))
        else:
            resume_streak = 0
        history_records[symbol_key] = {
            "paused_streak": paused_streak,
            "resume_streak": resume_streak,
            "last_status": status,
        }

        paused_display = str(paused_streak) if paused_streak else "-"
        resume_display = str(resume_streak) if resume_streak else "-"
        print(
            f"{symbol:>6} | {pnl:>9.2f} | {status:>10} | {paused_display:>13} | {resume_display:>13}"
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "resume_ready":
            resume_alerts.append((symbol, pnl))
        elif status == "paused":
            paused_alerts.append((symbol, pnl))

    if args.alert and resume_alerts:
        print("[resume-alert] Symbols ready to resume:")
        for symbol, pnl in resume_alerts:
            print(f"  - {symbol}: trend pnl {pnl:.2f}")
    if args.alert and paused_alerts:
        print("[paused-alert] Symbols above suspend but below resume:")
        for symbol, pnl in paused_alerts:
            print(f"  - {symbol}: trend pnl {pnl:.2f}")
    log_rows = []
    now_iso = datetime.now(timezone.utc).isoformat()

    if args.alert and paused_streak_alerts:
        print("[paused-streak] Paused streak lengths:")
        for symbol, streak in paused_streak_alerts:
            print(f"  - {symbol}: {streak} consecutive runs")
        threshold = max(args.paused_threshold, 1)
        over_threshold = [(symbol, streak) for symbol, streak in paused_streak_alerts if streak >= threshold]
        if over_threshold:
            print(f"[paused-escalation] Symbols paused for ≥{threshold} runs:")
            for symbol, streak in over_threshold:
                print(f"  - {symbol}: {streak} consecutive runs (trend still below resume floor)")
                log_rows.append(
                    {
                        "timestamp": now_iso,
                        "symbol": symbol,
                        "streak": streak,
                        "status": "paused",
                        "pnl": next(
                            (stats.get("pnl", 0.0) for sym, stats in summary.items() if sym.upper() == symbol.upper()),
                            None,
                        ),
                    }
                )
    if args.alert and resume_streak_alerts:
        print("[resume-streak] Resume-ready streak lengths:")
        for symbol, streak in resume_streak_alerts:
            print(f"  - {symbol}: {streak} consecutive runs")

    if args.summary:
        total_tracked = sum(status_counts.values())
        if total_tracked:
            summary_parts = [
                f"{label}={status_counts.get(label, 0)}"
                for label in ("resume_ready", "paused", "suspended", "neutral")
            ]
            print(f"[trend-summary] tracked={total_tracked} " + ", ".join(summary_parts))

    if args.history:
        args.history.parent.mkdir(parents=True, exist_ok=True)
        with args.history.open("w", encoding="utf-8") as handle:
            json.dump(history_records, handle, indent=2, sort_keys=True)

    if args.paused_log and log_rows:
        args.paused_log.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.paused_log.exists()
        with args.paused_log.open("a", encoding="utf-8") as handle:
            if write_header:
                handle.write("timestamp,symbol,status,streak,pnl\n")
            for row in log_rows:
                pnl_val = "" if row["pnl"] is None else f"{row['pnl']:.2f}"
                handle.write(
                    f"{row['timestamp']},{row['symbol']},{row['status']},{row['streak']},{pnl_val}\n"
                )


if __name__ == "__main__":
    main()
