#!/usr/bin/env python3
"""Rebuild forecast caches using best swept models."""
import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

SWEEP_RESULTS = Path("unified_hourly_experiment/forecast_cache/sweep_results.json")
CACHE_ROOT = Path("unified_hourly_experiment/forecast_cache")
DATA_ROOT = Path("trainingdatahourly/stocks")


def main():
    with open(SWEEP_RESULTS) as f:
        results = json.load(f)

    for symbol, data in results.items():
        model_path = data["model_path"]
        finetuned_ckpt = f"{model_path}/finetuned-ckpt"

        if not Path(finetuned_ckpt).exists():
            logger.warning("Missing checkpoint for {}: {}", symbol, finetuned_ckpt)
            continue

        logger.info("Building cache for {} using {}", symbol, model_path)

        cmd = [
            sys.executable, "-m", "alpacanewccrosslearning.build_forecasts",
            "--symbols", symbol,
            "--finetuned-model", finetuned_ckpt,
            "--forecast-cache-root", str(CACHE_ROOT),
            "--stock-data-root", str(DATA_ROOT),
            "--horizons", "1,24",
            "--lookback-hours", "8000",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            logger.success("Built cache for {}", symbol)
        else:
            logger.error("Failed {}: {}", symbol, result.stderr[-200:])


if __name__ == "__main__":
    main()
