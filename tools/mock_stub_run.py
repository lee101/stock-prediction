#!/usr/bin/env python3
"""Generate stubbed simulator outputs for tooling tests."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path


def build_stub_metrics(seed: int | None = None) -> dict[str, float | int | list[str]]:
    rng = random.Random(seed)
    return {
        "return": round(rng.uniform(-0.02, 0.03), 6),
        "sharpe": round(rng.uniform(-1.0, 1.5), 6),
        "pnl": round(rng.uniform(-5000, 8000), 2),
        "balance": round(100_000 + rng.uniform(-10_000, 15_000), 2),
        "steps": rng.randint(10, 50),
        "symbols": rng.sample(["AAPL", "MSFT", "NVDA", "GOOG", "TSLA", "AMZN"], 3),
    }


def write_log(log_path: Path, metrics: dict[str, float | int | list[str]]) -> None:
    timestamp = datetime.utcnow().isoformat()
    text = [
        f"[{timestamp}] Stub simulator run",
        "Starting trading loop (stub mode)…",
        f"Final return: {metrics['return']}",
        f"Final Sharpe: {metrics['sharpe']}",
        f"Final PnL: {metrics['pnl']}",
        f"Ending balance: {metrics['balance']}",
        f"Steps executed: {metrics['steps']}",
        f"Symbols traded: {', '.join(metrics['symbols'])}",
        "Run complete.",
    ]
    log_path.write_text("\n".join(text) + "\n", encoding="utf-8")


def write_summary(summary_path: Path, metrics: dict[str, float | int | list[str]]) -> None:
    summary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="Destination stub log file.")
    parser.add_argument(
        "--summary", required=True, type=Path, help="Destination JSON summary file."
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    args = parser.parse_args()

    metrics = build_stub_metrics(seed=args.seed)
    write_log(args.log, metrics)
    write_summary(args.summary, metrics)


if __name__ == "__main__":
    main()
