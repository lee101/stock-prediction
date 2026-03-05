#!/usr/bin/env python3
"""Rebuild forecast caches for all stocks using best LoRA models."""
import subprocess
import sys
from pathlib import Path

from loguru import logger


BEST_MODELS = {
    "NVDA": "NVDA_lora_nonreg_20260304_213434_ctx512_lr0p00005_st400_r16_a32_d0",
    "MSFT": "MSFT_lora_percent_change_ctx128_lr5e-05_r16_20260213_171918",
    "META": "META_lora_differencing_ctx128_lr5e-05_r16_20260213_172640",
    "GOOG": "GOOG_lora_nonreg_20260304_213434_ctx512_lr0p00005_st400_r16_a32_d0p05",
    "NET": "NET_lora_differencing_ctx128_lr5e-5_r16_20260221_080636",
    "PLTR": "PLTR_lora_nonreg_20260304_213434_ctx512_lr0p00007_st400_r32_a32_d0p05",
    "NYT": "NYT_lora_differencing_ctx128_lr5e-5_r16_20260221_084237",
    "YELP": "YELP_lora_percent_change_ctx128_lr5e-05_r16_20260213_175101",
    "DBX": "DBX_lora_nonreg_20260304_213434_ctx512_lr0p00007_st400_r16_a32_d0",
    "TRIP": "TRIP_lora_nonreg_20260304_213434_ctx512_lr0p00005_st400_r32_a32_d0",
    "KIND": "KIND_lora_differencing_ctx128_lr5e-05_r16_20260215_225721",
    "EBAY": "EBAY_lora_robust_scaling_ctx128_lr5e-5_r16_20260221_083308",
    "MTCH": "MTCH_lora_nonreg_20260304_213434_ctx512_lr0p00007_st400_r32_a32_d0p05",
    "ANGI": "ANGI_lora_differencing_ctx128_lr5e-05_r16_20260215_224308",
    "Z": "Z_lora_differencing_ctx128_lr5e-05_r16_20260215_224604",
    "EXPE": "EXPE_lora_differencing_ctx128_lr5e-05_r16_20260215_224859",
    "BKNG": "BKNG_lora_differencing_ctx128_lr5e-05_r16_20260215_225148",
    "NWSA": "NWSA_lora_differencing_ctx128_lr5e-05_r16_20260215_225432",
    "TSLA": "TSLA_lora_differencing_ctx128_lr5e-05_r16_20260218_102822",
    "AAPL": "AAPL_lora_differencing_ctx128_lr5e-05_r16_20260218_102824",
    "QUBT": "QUBT_lora_percent_change_ctx128_lr5e-05_r16_20260218_224547",
}

CACHE_ROOT = Path("unified_hourly_experiment/forecast_cache")
DATA_ROOT = Path("trainingdatahourly/stocks")
MODEL_ROOT = Path("chronos2_finetuned")

def _resolve_model_path(symbol: str, model_name: str) -> Path | None:
    """Resolve a usable finetuned checkpoint path for a symbol.

    Prefers the configured model name, but falls back to the most recently
    updated local checkpoint for the symbol when mappings become stale.
    """
    preferred = MODEL_ROOT / model_name / "finetuned-ckpt"
    if preferred.exists():
        return preferred

    fallback_candidates: list[tuple[float, Path]] = []
    for candidate_dir in MODEL_ROOT.glob(f"{symbol}_*"):
        ckpt = candidate_dir / "finetuned-ckpt"
        if ckpt.exists():
            try:
                mtime = ckpt.stat().st_mtime
            except OSError:
                continue
            fallback_candidates.append((mtime, ckpt))

    if not fallback_candidates:
        logger.warning("Model not found and no fallback candidates: {}", preferred)
        return None

    fallback_candidates.sort(key=lambda row: row[0], reverse=True)
    chosen = fallback_candidates[0][1]
    logger.warning(
        "Model not found for {} -> {}. Falling back to latest available checkpoint: {}",
        symbol,
        preferred,
        chosen,
    )
    return chosen

def build_cache(symbol: str, model_name: str):
    """Build forecast cache for a symbol."""
    model_path = _resolve_model_path(symbol, model_name)
    if model_path is None:
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
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
