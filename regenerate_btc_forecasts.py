#!/usr/bin/env python3
"""Regenerate BTC forecast cache using finetuned Chronos2 LoRA model."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def find_best_btc_lora_model() -> tuple[Path, dict]:
    """Find the best BTC LoRA model from training."""
    base_dirs = [
        Path("chronos2_finetuned"),
        Path("/home/lee/code/btcmarketsbot/chronos2_finetuned"),
    ]

    best_model = None
    best_config = None
    best_score = float("inf")

    for base_dir in base_dirs:
        if not base_dir.exists():
            continue
        for model_dir in base_dir.glob("BTC_lora_*"):
            ckpt_dir = model_dir / "finetuned-ckpt"
            config_file = model_dir / "config.json"
            if not ckpt_dir.exists() or not config_file.exists():
                continue
            try:
                config = json.loads(config_file.read_text())
                metrics = config.get("final_metrics", {})
                consistency = metrics.get("consistency_score") or metrics.get("val_consistency_score")
                if consistency is None:
                    continue
                if consistency < best_score:
                    best_score = consistency
                    best_model = ckpt_dir
                    best_config = config
            except Exception as e:
                logger.warning(f"Failed to read {config_file}: {e}")
                continue

    return best_model, best_config


def update_hyperparam_config(model_path: Path, config: dict):
    """Update hyperparams to use finetuned model for BTC."""
    hyperparam_dir = Path("hyperparams/chronos2/hourly")
    hyperparam_dir.mkdir(parents=True, exist_ok=True)

    btc_config = {
        "model_id": str(model_path),
        "context_length": config.get("context_length", 512),
        "prediction_length": config.get("prediction_length", 24),
        "quantile_levels": [0.1, 0.5, 0.9],
        "batch_size": 128,
        "device_map": "cuda",
        "name": f"finetuned_btc_lora_{model_path.parent.name}",
    }

    config_path = hyperparam_dir / "BTCUSD.json"
    config_path.write_text(json.dumps(btc_config, indent=2))
    logger.info(f"Updated {config_path}")
    return config_path


def regenerate_forecasts(symbol: str, horizons: list[int], force: bool = True):
    """Regenerate forecast cache for symbol."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from binanceneural.forecasts import ChronosForecastManager
    from binanceneural.config import ForecastConfig

    data_root = Path("trainingdatahourly/crypto")
    cache_root = Path("forecast_cache_hourly")

    for horizon in horizons:
        horizon_dir = cache_root / f"h{horizon}"
        config = ForecastConfig(
            symbol=symbol,
            data_root=data_root,
            context_hours=512,
            prediction_horizon_hours=horizon,
            quantile_levels=(0.1, 0.5, 0.9),
            batch_size=128,
            cache_dir=horizon_dir,
        )
        manager = ChronosForecastManager(config)
        logger.info(f"Regenerating {symbol} forecasts for horizon={horizon}h...")
        frame = manager.ensure_latest(force_rebuild=force)
        logger.info(f"Generated {len(frame)} forecast rows for {symbol} h{horizon}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSD", help="Symbol to regenerate forecasts for")
    parser.add_argument("--horizons", default="1,4,8,24", help="Comma-separated forecast horizons")
    parser.add_argument("--model-path", type=Path, help="Path to finetuned model (auto-detect if not provided)")
    parser.add_argument("--no-force", action="store_true", help="Don't force rebuild existing forecasts")
    parser.add_argument("--update-config-only", action="store_true", help="Only update hyperparams, don't regenerate")
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]

    if args.model_path:
        model_path = args.model_path
        config = {"context_length": 512, "prediction_length": 24}
    else:
        model_path, config = find_best_btc_lora_model()
        if model_path is None:
            logger.error("No finetuned BTC LoRA model found")
            sys.exit(1)
        logger.info(f"Found best model: {model_path}")
        logger.info(f"Consistency score: {config.get('final_metrics', {}).get('consistency_score', 'N/A')}")

    update_hyperparam_config(model_path, config)

    if args.update_config_only:
        logger.info("Config updated. Skipping forecast regeneration.")
        return

    os.environ["CHRONOS2_MODEL_ID_OVERRIDE"] = str(model_path)
    regenerate_forecasts(args.symbol, horizons, force=not args.no_force)
    logger.info("Done regenerating forecasts")


if __name__ == "__main__":
    main()
