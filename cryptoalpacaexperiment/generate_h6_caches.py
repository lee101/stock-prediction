#!/usr/bin/env python3
"""Generate h6 Chronos2 forecast caches for BTC and ETH."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from binanceneural.forecasts import ChronosForecastManager, ForecastConfig
from src.models.chronos2_wrapper import Chronos2OHLCWrapper

SYMBOLS = {
    "BTCUSD": "chronos2_finetuned/BTCUSD_lora_percent_change_ctx128_lr5e-5_r16/finetuned-ckpt",
    "ETHUSD": "chronos2_finetuned/ETHUSD_lora_percent_change_ctx128_lr5e-5_r16/finetuned-ckpt",
}

CACHE_ROOT = Path("cryptoalpacaexperiment/forecast_cache")
DATA_ROOT = Path("trainingdatahourly/crypto")

def main():
    for horizon in [1, 6]:
        for symbol, model_path in SYMBOLS.items():
            cache_dir = CACHE_ROOT / f"h{horizon}"
            existing = cache_dir / f"{symbol}.parquet"
            if existing.exists() and horizon == 1:
                logger.info(f"h{horizon} cache for {symbol} already exists, skipping")
                continue

            logger.info(f"Generating h{horizon} cache for {symbol} using {model_path}")
            wrapper = Chronos2OHLCWrapper.from_pretrained(
                model_id=str(Path(model_path)),
                device_map="cuda",
                default_context_length=24 * 14,
                default_batch_size=128,
                quantile_levels=(0.1, 0.5, 0.9),
            )

            def _factory():
                return wrapper

            cfg = ForecastConfig(
                symbol=symbol,
                data_root=DATA_ROOT,
                context_hours=24 * 14,
                prediction_horizon_hours=horizon,
                quantile_levels=(0.1, 0.5, 0.9),
                batch_size=128,
                cache_dir=cache_dir,
            )
            manager = ChronosForecastManager(cfg, wrapper_factory=_factory)
            manager.ensure_latest(cache_only=False)
            logger.success(f"Done: h{horizon} {symbol}")

if __name__ == "__main__":
    main()
