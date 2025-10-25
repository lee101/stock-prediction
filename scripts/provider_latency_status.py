#!/usr/bin/env python3
"""Evaluate provider latency snapshot and emit OK/WARN/CRIT status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def load_snapshot(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Latency snapshot not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(
    snapshot: Dict[str, Dict[str, float]],
    warn_threshold: float,
    crit_threshold: float,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    status = "OK"
    details: Dict[str, Dict[str, float]] = {}
    for provider, stats in snapshot.items():
        delta = abs(stats.get("delta_avg_ms", 0.0))
        severity = "ok"
        if delta >= crit_threshold:
            status = "CRIT"
            severity = "crit"
        elif delta >= warn_threshold and status != "CRIT":
            status = "WARN"
            severity = "warn"
        details[provider] = {
            "avg_ms": stats.get("avg_ms", 0.0),
            "delta_avg_ms": delta,
            "p95_ms": stats.get("p95_ms", 0.0),
            "severity": severity,
        }
    return status, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise provider latency health.")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("marketsimulator/run_logs/provider_latency_rolling.json"),
        help="Rolling latency snapshot JSON path.",
    )
    parser.add_argument(
        "--warn",
        type=float,
        default=20.0,
        help="Absolute delta threshold (ms) producing WARN status (default 20).",
    )
    parser.add_argument(
        "--crit",
        type=float,
        default=40.0,
        help="Absolute delta threshold (ms) producing CRIT status (default 40).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output instead of text.")
    args = parser.parse_args()

    snapshot = load_snapshot(args.snapshot)
    status, details = evaluate(snapshot, warn_threshold=args.warn, crit_threshold=args.crit)

    if args.json:
        output = {
            "status": status,
            "warn_threshold": args.warn,
            "crit_threshold": args.crit,
            "providers": details,
        }
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(f"status={status} warn={args.warn}ms crit={args.crit}ms")
        for provider, stats in sorted(details.items()):
            print(
                f"  - {provider}: avg={stats['avg_ms']:.2f}ms Δavg={stats['delta_avg_ms']:.2f}ms "
                f"p95={stats['p95_ms']:.2f}ms severity={stats['severity']}"
            )

    exit_code = {"OK": 0, "WARN": 1, "CRIT": 2}[status]
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
