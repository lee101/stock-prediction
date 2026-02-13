#!/usr/bin/env python3
"""
Sortino optimization sweep for SUI/ETH.
Varies return_weight to find optimal Sortino.
Based on binanceprogress.md findings: lower return_weight -> higher Sortino
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent


@dataclass
class SweepConfig:
    symbol: str
    return_weight: float
    epochs: int = 15
    sequence_length: int = 72
    horizons: str = "1,4,24"
    batch_size: int = 16
    learning_rate: float = 3e-4


def run_experiment(cfg: SweepConfig) -> dict:
    run_name = f"{cfg.symbol.lower()}_sortino_rw{str(cfg.return_weight).replace('.', '')}"

    # Use existing forecast cache
    if cfg.symbol == "SUIUSDT":
        cache_root = "binancechronossolexperiment/forecast_cache_sui_10bp"
    else:
        cache_root = "binanceneural/forecast_cache"

    cmd = [
        sys.executable, "-m", "binancechronossolexperiment.run_experiment",
        "--symbol", cfg.symbol,
        "--return-weight", str(cfg.return_weight),
        "--epochs", str(cfg.epochs),
        "--sequence-length", str(cfg.sequence_length),
        "--horizons", cfg.horizons,
        "--batch-size", str(cfg.batch_size),
        "--learning-rate", str(cfg.learning_rate),
        "--forecast-cache-root", cache_root,
        "--cache-only",
        "--no-compile",
        "--run-name", run_name,
    ]

    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=str(REPO_ROOT))

        metrics_path = EXP_DIR / "results" / run_name / "simulation_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            return {
                "name": run_name,
                "config": {
                    "symbol": cfg.symbol,
                    "return_weight": cfg.return_weight,
                    "epochs": cfg.epochs,
                },
                "test_return": metrics.get("metrics", {}).get("test", {}).get("total_return", 0),
                "test_sortino": metrics.get("metrics", {}).get("test", {}).get("sortino", 0),
                "test_trades": metrics.get("metrics", {}).get("test", {}).get("num_trades", 0),
                "final_equity": metrics.get("metrics", {}).get("test", {}).get("final_equity", 10000),
            }
        else:
            return {"name": run_name, "error": "No metrics file", "stdout": result.stdout[-500:], "stderr": result.stderr[-500:]}
    except Exception as e:
        return {"name": run_name, "error": str(e)}


def run_sweep(symbol: str, output_path: Path):
    configs = [
        SweepConfig(symbol=symbol, return_weight=0.005, epochs=15),
        SweepConfig(symbol=symbol, return_weight=0.01, epochs=15),
        SweepConfig(symbol=symbol, return_weight=0.02, epochs=15),
        SweepConfig(symbol=symbol, return_weight=0.04, epochs=15),
        SweepConfig(symbol=symbol, return_weight=0.01, epochs=30),
        SweepConfig(symbol=symbol, return_weight=0.01, epochs=60),
        SweepConfig(symbol=symbol, return_weight=0.01, epochs=15, sequence_length=96),
    ]

    results = []
    for cfg in configs:
        result = run_experiment(cfg)
        results.append(result)

        if "error" not in result:
            print(f"  -> return={result['test_return']:.4f}, sortino={result['test_sortino']:.2f}")
        else:
            print(f"  -> ERROR: {result['error']}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE: {symbol}")
    print(f"{'='*60}")

    sorted_results = sorted([r for r in results if "error" not in r], key=lambda x: x.get("test_sortino", 0), reverse=True)
    for r in sorted_results:
        print(f"{r['name']}: sortino={r['test_sortino']:.2f}, return={r['test_return']:.4f}")

    if sorted_results:
        best = sorted_results[0]
        print(f"\nBest: {best['name']} (Sortino {best['test_sortino']:.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SUIUSDT", help="Symbol to sweep")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else EXP_DIR / f"sortino_sweep_{args.symbol.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    run_sweep(args.symbol, output_path)


if __name__ == "__main__":
    main()
