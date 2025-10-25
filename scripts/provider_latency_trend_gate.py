#!/usr/bin/env python3
"""Exit with CRIT if weekly CRIT/WARN delta exceeds thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from scripts.provider_latency_weekly_report import compute_trend, load_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate pipeline on weekly latency deltas.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_alert_history.jsonl"),
    )
    parser.add_argument("--window", type=int, default=7, help="Snapshots in current window (default 7).")
    parser.add_argument("--compare-window", type=int, default=7, help="Snapshots in compare window (default 7).")
    parser.add_argument("--crit-limit", type=int, default=2, help="Max allowed CRIT delta before failing.")
    parser.add_argument("--warn-limit", type=int, default=5, help="Max allowed WARN delta before failing.")
    args = parser.parse_args()

    entries = load_history(args.history)
    deltas = compute_trend(entries, args.window, args.compare_window)
    if not deltas:
        print("[warn] Not enough history for trend gating; skipping.")
        return

    violations: Dict[str, Dict[str, int]] = {}
    for provider, delta_map in deltas.items():
        crit_delta = delta_map["CRIT"]
        warn_delta = delta_map["WARN"]
        if crit_delta >= args.crit_limit or warn_delta >= args.warn_limit:
            violations[provider] = {"CRIT": crit_delta, "WARN": warn_delta}

    if violations:
        print("[error] Weekly latency trend gate failed:")
        for provider, delta_map in violations.items():
            print(f"  {provider}: CRIT Δ={delta_map['CRIT']} WARN Δ={delta_map['WARN']}")
        raise SystemExit(2)

    print("[info] Weekly latency trend within thresholds.")


if __name__ == "__main__":
    main()
