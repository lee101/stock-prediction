#!/usr/bin/env python3
"""Train LoRA models for new stocks using differencing (best for most stocks)."""
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

NEW_STOCKS = ["EBAY", "MTCH", "ANGI", "Z", "EXPE", "BKNG", "NWSA", "KIND"]
DATA_ROOT = Path("trainingdatahourly/stocks")
CACHE_ROOT = Path("unified_hourly_experiment/forecast_cache")

def train_lora(symbol: str):
    """Train LoRA for a single symbol."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"{symbol}_lora_differencing_ctx128_lr5e-05_r16_{timestamp}"

    cmd = [
        sys.executable, "chronos2_trainer.py",
        "--symbol", symbol,
        "--data-root", str(DATA_ROOT),
        "--output-root", "chronos2_finetuned",
        "--save-name", save_name,
        "--context-length", "128",
        "--learning-rate", "5e-5",
        "--lora-r", "16",
        "--finetune-mode", "lora",
        "--num-steps", "1000",
    ]

    logger.info(f"Training LoRA for {symbol}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode == 0:
        logger.success(f"Trained LoRA for {symbol} -> chronos2_finetuned/{save_name}")
        return f"chronos2_finetuned/{save_name}"
    else:
        logger.error(f"Failed {symbol}: {result.stderr[-300:]}")
        return None

def build_cache(symbol: str, model_dir: str):
    """Build forecast cache for a symbol."""
    model_path = f"{model_dir}/finetuned-ckpt"
    if not Path(model_path).exists():
        logger.warning(f"Model not found: {model_path}")
        return False

    cmd = [
        sys.executable, "-m", "alpacanewccrosslearning.build_forecasts",
        "--symbols", symbol,
        "--finetuned-model", model_path,
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
        logger.error(f"Cache build failed {symbol}: {result.stderr[-200:]}")
        return False

def main():
    logger.info("=" * 60)
    logger.info("Training LoRA + Building Caches for New Stocks")
    logger.info("=" * 60)

    success = 0
    for symbol in NEW_STOCKS:
        model_dir = train_lora(symbol)
        if model_dir:
            if build_cache(symbol, model_dir):
                success += 1

    logger.info("=" * 60)
    logger.info(f"Complete: {success}/{len(NEW_STOCKS)} successful")

if __name__ == "__main__":
    main()
