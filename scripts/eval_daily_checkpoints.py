#!/usr/bin/env python3
"""Evaluate multiple neuraldaily checkpoints over a fixed window.

Usage:
  python scripts/eval_daily_checkpoints.py --suite daily_confidence_40ep_full --start 2025-11-11 --days 10 --crypto-fee 0.0008 --stock-fee 0.0005 --top-k 8

Outputs JSON with per-checkpoint PnL/Sortino and writes to runs/<suite>_eval_<date>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuraldailytraining import DailyTradingRuntime
from neuraldailymarketsimulator.simulator import NeuralDailyMarketSimulator


def eval_checkpoint(path: Path, symbols: List[str], start: str, days: int, stock_fee: float, crypto_fee: float):
    runtime = DailyTradingRuntime(path)
    sim = NeuralDailyMarketSimulator(runtime, symbols, stock_fee=stock_fee, crypto_fee=crypto_fee)
    _, summary = sim.run(start_date=start, days=days)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="Checkpoint directory under neuraldailytraining/checkpoints")
    ap.add_argument("--start", required=True, help="ISO start date")
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--stock-fee", type=float, default=0.0005)
    ap.add_argument("--crypto-fee", type=float, default=0.0008)
    ap.add_argument("--top-k", type=int, default=5, help="How many checkpoints from manifest to score (sorted by val_loss)")
    ap.add_argument("--output", help="Optional output path")
    args = ap.parse_args()

    suite_dir = Path("neuraldailytraining/checkpoints") / args.suite
    manifest = suite_dir / "manifest.json"
    data = json.loads(manifest.read_text())
    ckpts = data["checkpoints"]
    ckpts = sorted(ckpts, key=lambda x: x["val_loss"])[: args.top_k]
    symbols = data["config"]["dataset"]["symbols"]

    results = []
    for row in ckpts:
        path = suite_dir / row["path"]
        summary = eval_checkpoint(path, symbols, args.start, args.days, args.stock_fee, args.crypto_fee)
        results.append({"path": str(path), **summary})
        print(path.name, summary)

    out_path = (
        Path(args.output)
        if args.output
        else Path("runs") / f"{args.suite}_eval_{args.start.replace('-', '')}_{args.days}d.json"
    )
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} results -> {out_path}")


if __name__ == "__main__":
    main()
