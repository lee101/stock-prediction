#!/usr/bin/env python3
"""Sweep preaug strategies for stocks and find best config."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

PREAUG_STRATEGIES = ["percent_change", "log_returns", "differencing", "robust_scaling"]


def train_and_get_mae(
    symbol: str,
    preaug: str,
    data_root: Path,
    context_length: int = 128,
    learning_rate: float = 5e-5,
    num_steps: int = 500,
) -> Optional[Dict]:
    """Train a model and return MAE metrics."""
    cmd = [
        sys.executable, "scripts/train_crypto_lora_sweep.py",
        "--symbol", symbol,
        "--data-root", str(data_root),
        "--context-length", str(context_length),
        "--learning-rate", str(learning_rate),
        "--num-steps", str(num_steps),
        "--preaug", preaug,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=Path(__file__).parent.parent)
        if result.returncode != 0:
            logger.warning("Training failed for {} preaug={}: {}", symbol, preaug, result.stderr[-200:])
            return None

        # Parse the JSON result file
        for line in result.stdout.split("\n"):
            if "Results:" in line:
                results_path = line.split("Results:")[1].strip()
                with open(results_path) as f:
                    data = json.load(f)
                    return {
                        "symbol": symbol,
                        "preaug": preaug,
                        "context_length": context_length,
                        "learning_rate": learning_rate,
                        "val_mae_percent": data["val"]["mae_percent_mean"],
                        "val_consistency": data["val_consistency_score"],
                        "model_path": data["output_dir"],
                    }
        return None
    except Exception as e:
        logger.error("Error training {} preaug={}: {}", symbol, preaug, e)
        return None


def sweep_symbol(symbol: str, data_root: Path) -> Optional[Dict]:
    """Sweep all preaug strategies for a symbol."""
    logger.info("Sweeping preaug strategies for {}", symbol)
    best = None

    for preaug in PREAUG_STRATEGIES:
        logger.info("  {} - {}", symbol, preaug)
        result = train_and_get_mae(symbol, preaug, data_root)
        if result:
            logger.info("    MAE: {:.4f}%", result["val_mae_percent"])
            if best is None or result["val_mae_percent"] < best["val_mae_percent"]:
                best = result
        else:
            logger.warning("    Failed")

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--stock-data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--crypto-data-root", type=Path, default=Path("trainingdatahourly/crypto"))
    parser.add_argument("--output", type=Path, default=Path("unified_hourly_experiment/best_configs.json"))
    args = parser.parse_args()

    from src.symbol_utils import is_crypto_symbol

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    results = {}

    for symbol in symbols:
        data_root = args.crypto_data_root if is_crypto_symbol(symbol) else args.stock_data_root
        best = sweep_symbol(symbol, data_root)
        if best:
            results[symbol] = best
            logger.success("{}: Best = {} with {:.4f}% MAE", symbol, best["preaug"], best["val_mae_percent"])

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\nBest configs saved to {}", args.output)
    logger.info("\nSummary:")
    for sym, data in sorted(results.items(), key=lambda x: x[1]["val_mae_percent"]):
        logger.info("  {}: {:.4f}% ({})", sym, data["val_mae_percent"], data["preaug"])


if __name__ == "__main__":
    main()
