#!/usr/bin/env python3
"""Rebuild OP/ARB forecast caches with full price history coverage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gc
import pandas as pd
import torch
from loguru import logger

from binanceneural.forecasts import ChronosForecastManager, ForecastConfig
from src.models.chronos2_wrapper import Chronos2OHLCWrapper

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "trainingdatahourlybinance"
CACHE_DIR = REPO / "binanceneural/forecast_cache/h1"

SYMBOLS = {
    "OPUSDT": "chronos2_finetuned/OPUSDT_lora_percent_change_ctx128_lr5e-5_r16/finetuned-ckpt",
    "ARBUSDT": "chronos2_finetuned/ARBUSDT_lora_percent_change_ctx128_lr5e-5_r16/finetuned-ckpt",
}


def rebuild(symbol: str, model_id: str):
    csv = DATA_ROOT / f"{symbol}.csv"
    if not csv.exists():
        logger.error(f"No CSV: {csv}")
        return

    price = pd.read_csv(csv, parse_dates=["timestamp"])
    if price["timestamp"].dt.tz is None:
        price["timestamp"] = price["timestamp"].dt.tz_localize("UTC")

    start_ts = price["timestamp"].min()
    end_ts = pd.Timestamp.now(tz="UTC")
    logger.info(f"{symbol}: {len(price)} rows, {start_ts} to {end_ts}")

    # check if LoRA exists, fallback to base
    lora_path = REPO / model_id
    if not lora_path.exists():
        logger.warning(f"LoRA not found: {lora_path}, using base model")
        model_id = "amazon/chronos-2"

    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id=model_id,
        device_map="cuda",
        default_context_length=512,
        default_batch_size=64,  # smaller to avoid OOM with concurrent GPU use
        quantile_levels=(0.1, 0.5, 0.9),
    )

    cfg = ForecastConfig(
        symbol=symbol,
        data_root=csv.parent,
        context_hours=512,
        prediction_horizon_hours=1,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=64,
        cache_dir=CACHE_DIR,
    )
    manager = ChronosForecastManager(cfg, wrapper_factory=lambda: wrapper)
    manager.ensure_latest(start=start_ts, end=end_ts, cache_only=False, force_rebuild=True)

    # verify
    pq = CACHE_DIR / f"{symbol}.parquet"
    if pq.exists():
        fc = pd.read_parquet(pq)
        coverage = len(fc) / len(price) * 100
        logger.info(f"  Rebuilt: {len(fc)} rows ({coverage:.1f}% coverage)")
    else:
        logger.error(f"  Cache file not created!")

    del wrapper
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for sym, model in SYMBOLS.items():
        rebuild(sym, model)
        logger.info("")
