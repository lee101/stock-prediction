#!/usr/bin/env python3
"""Rebuild forecast caches for all stocks using best LoRA models."""
import subprocess
import sys
from pathlib import Path
from loguru import logger

BEST_MODELS = {
    "NVDA": "NVDA_lora_log_returns_ctx128_lr5e-05_r16_20260213_171522",
    "MSFT": "MSFT_lora_percent_change_ctx128_lr5e-05_r16_20260213_171918",
    "META": "META_lora_differencing_ctx128_lr5e-05_r16_20260213_172640",
    "GOOG": "GOOG_lora_differencing_ctx128_lr5e-05_r16_20260213_173150",
    "NET": "NET_lora_differencing_ctx128_lr5e-05_r16_20260213_173740",
    "PLTR": "PLTR_lora_robust_scaling_ctx128_lr5e-05_r16_20260213_174410",
    "NYT": "NYT_lora_differencing_ctx128_lr5e-05_r16_20260213_174756",
    "YELP": "YELP_lora_percent_change_ctx128_lr5e-05_r16_20260213_175101",
    "DBX": "DBX_lora_differencing_ctx128_lr5e-05_r16_20260213_175824",
    "TRIP": "TRIP_lora_percent_change_ctx128_lr5e-05_r16_20260213_180128",
    "KIND": "KIND_lora_differencing_ctx128_lr5e-05_r16_20260215_223432",
    "EBAY": "EBAY_lora_differencing_ctx128_lr5e-05_r16_20260215_223717",
    "MTCH": "MTCH_lora_differencing_ctx128_lr5e-05_r16_20260215_224012",
    "ANGI": "ANGI_lora_differencing_ctx128_lr5e-05_r16_20260215_224308",
    "Z": "Z_lora_differencing_ctx128_lr5e-05_r16_20260215_224604",
    "EXPE": "EXPE_lora_differencing_ctx128_lr5e-05_r16_20260215_224859",
    "BKNG": "BKNG_lora_differencing_ctx128_lr5e-05_r16_20260215_225148",
    "NWSA": "NWSA_lora_differencing_ctx128_lr5e-05_r16_20260215_225432",
}

CACHE_ROOT = Path("unified_hourly_experiment/forecast_cache")
DATA_ROOT = Path("trainingdatahourly/stocks")
MODEL_ROOT = Path("chronos2_finetuned")

def build_cache(symbol: str, model_name: str):
    """Build forecast cache for a symbol."""
    model_path = MODEL_ROOT / model_name / "finetuned-ckpt"
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        return False

    cmd = [
        sys.executable, "-m", "alpacanewccrosslearning.build_forecasts",
        "--symbols", symbol,
        "--finetuned-model", str(model_path),
        "--forecast-cache-root", str(CACHE_ROOT),
        "--stock-data-root", str(DATA_ROOT),
        "--horizons", "1,24",
        "--lookback-hours", "8000",
    ]

    logger.info(f"Building cache for {symbol}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        logger.success(f"Built cache for {symbol}")
        return True
    else:
        logger.error(f"Failed {symbol}: {result.stderr[-200:]}")
        return False

def main():
    logger.info("=" * 60)
    logger.info("Rebuilding forecast caches")
    logger.info("=" * 60)

    success = 0
    failed = []

    for symbol, model in BEST_MODELS.items():
        if build_cache(symbol, model):
            success += 1
        else:
            failed.append(symbol)

    logger.info("=" * 60)
    logger.info(f"Complete: {success}/{len(BEST_MODELS)} successful")
    if failed:
        logger.warning(f"Failed: {failed}")

if __name__ == "__main__":
    main()
