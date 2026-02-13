#!/usr/bin/env python3
"""Sweep preaug strategies and hyperparams, retrain with best, build trading bot."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from loguru import logger

PREAUG_STRATEGIES = ["percent_change", "log_returns", "differencing", "robust_scaling"]
CONTEXT_LENGTHS = [64, 128, 256]
LEARNING_RATES = [1e-5, 5e-5, 1e-4]


@dataclass
class SweepConfig:
    symbol: str
    preaug: str
    context_length: int
    learning_rate: float
    num_steps: int = 1000


@dataclass
class SweepResult:
    config: SweepConfig
    val_mae_percent: float
    val_consistency: float
    model_path: str
    success: bool
    error: Optional[str] = None


def run_single_train(cfg: SweepConfig, data_root: Path, output_root: Path) -> SweepResult:
    """Run a single training config and return result."""
    cmd = [
        sys.executable,
        "scripts/train_crypto_lora_sweep.py",
        "--symbol", cfg.symbol,
        "--data-root", str(data_root),
        "--output-root", str(output_root),
        "--context-length", str(cfg.context_length),
        "--learning-rate", str(cfg.learning_rate),
        "--num-steps", str(cfg.num_steps),
        "--preaug", cfg.preaug,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=Path(__file__).parent.parent)
        if result.returncode != 0:
            return SweepResult(cfg, float("inf"), float("inf"), "", False, result.stderr[-500:])

        val_mae = float("inf")
        val_cons = float("inf")
        model_path = ""

        for line in result.stderr.split("\n"):
            if "Results:" in line:
                results_path = line.split("Results:")[1].strip()
                with open(results_path) as f:
                    data = json.load(f)
                    model_path = data.get("output_dir", "")
                    val_mae = data["val"]["mae_percent_mean"]
                    val_cons = data["val_consistency_score"]
                break

        if val_mae == float("inf"):
            return SweepResult(cfg, val_mae, val_cons, model_path, False, "No results file found")
        return SweepResult(cfg, val_mae, val_cons, model_path, True)
    except Exception as e:
        return SweepResult(cfg, float("inf"), float("inf"), "", False, str(e))


def sweep_symbol(
    symbol: str,
    data_root: Path,
    output_root: Path,
    quick: bool = False,
) -> Tuple[SweepConfig, SweepResult]:
    """Sweep all configs for a symbol and return best."""
    configs = []

    if quick:
        # Quick sweep: just try different preaug with default hyperparams
        for preaug in PREAUG_STRATEGIES:
            configs.append(SweepConfig(symbol, preaug, 128, 5e-5, 500))
    else:
        # Full sweep
        for preaug in PREAUG_STRATEGIES:
            for ctx in CONTEXT_LENGTHS:
                for lr in LEARNING_RATES:
                    configs.append(SweepConfig(symbol, preaug, ctx, lr, 1000))

    logger.info("Sweeping {} configs for {}", len(configs), symbol)

    best_result = None
    best_config = None

    for cfg in configs:
        logger.info("  Testing {}: preaug={} ctx={} lr={:.0e}", symbol, cfg.preaug, cfg.context_length, cfg.learning_rate)
        result = run_single_train(cfg, data_root, output_root)

        if result.success:
            logger.info("    Val MAE: {:.4f}%", result.val_mae_percent)
            if best_result is None or result.val_mae_percent < best_result.val_mae_percent:
                best_result = result
                best_config = cfg
        else:
            logger.warning("    Failed: {}", result.error[:100] if result.error else "Unknown")

    return best_config, best_result


def build_forecast_cache(
    symbol: str,
    model_path: str,
    cache_root: Path,
    data_root: Path,
    horizons: str = "1,24",
    lookback_hours: int = 2000,
) -> bool:
    """Build forecast cache for a trained model."""
    cmd = [
        sys.executable, "-m", "alpacanewccrosslearning.build_forecasts",
        "--symbols", symbol,
        "--finetuned-model", f"{model_path}/finetuned-ckpt",
        "--forecast-cache-root", str(cache_root),
        "--stock-data-root", str(data_root),
        "--horizons", horizons,
        "--lookback-hours", str(lookback_hours),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0
    except Exception as e:
        logger.error("Failed to build cache for {}: {}", symbol, e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--stock-data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--crypto-data-root", type=Path, default=Path("trainingdatahourly/crypto"))
    parser.add_argument("--output-root", type=Path, default=Path("chronos2_finetuned"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--quick", action="store_true", help="Quick sweep (fewer configs)")
    parser.add_argument("--skip-cache", action="store_true", help="Skip building forecast caches")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Determine data root per symbol
    from src.symbol_utils import is_crypto_symbol

    results = {}

    for symbol in symbols:
        data_root = args.crypto_data_root if is_crypto_symbol(symbol) else args.stock_data_root

        logger.info("=" * 60)
        logger.info("Sweeping {}", symbol)
        logger.info("=" * 60)

        best_config, best_result = sweep_symbol(symbol, data_root, args.output_root, args.quick)

        if best_result and best_result.success:
            results[symbol] = {
                "config": asdict(best_config),
                "val_mae_percent": best_result.val_mae_percent,
                "val_consistency": best_result.val_consistency,
                "model_path": best_result.model_path,
            }
            logger.success("{}: Best config = {} (MAE {:.4f}%)", symbol, best_config.preaug, best_result.val_mae_percent)

            if not args.skip_cache:
                logger.info("Building forecast cache for {}", symbol)
                if build_forecast_cache(symbol, best_result.model_path, args.cache_root, data_root):
                    logger.success("Cache built for {}", symbol)
                else:
                    logger.error("Failed to build cache for {}", symbol)
        else:
            logger.error("No successful config found for {}", symbol)

    # Save results
    results_path = args.cache_root / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to {}", results_path)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SWEEP SUMMARY")
    logger.info("=" * 60)
    for symbol, data in sorted(results.items(), key=lambda x: x[1]["val_mae_percent"]):
        logger.info("{}: {:.4f}% MAE (preaug={}, ctx={}, lr={:.0e})",
            symbol,
            data["val_mae_percent"],
            data["config"]["preaug"],
            data["config"]["context_length"],
            data["config"]["learning_rate"],
        )


if __name__ == "__main__":
    main()
