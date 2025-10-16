#!/usr/bin/env python3
"""CLI utility to build and cache GymRL feature cubes."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gymrl import FeatureBuilder, FeatureBuilderConfig
from gymrl.cache_utils import save_feature_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and cache GymRL feature cubes.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing per-symbol CSV history.")
    parser.add_argument("--output", type=Path, required=True, help="Destination NPZ path for the feature cache.")
    parser.add_argument("--forecast-backend", type=str, default="toto", choices=["toto", "chronos", "bootstrap"], help="Forecast backend to use.")
    parser.add_argument("--num-samples", type=int, default=2048, help="Number of forecast samples (must satisfy backend requirements).")
    parser.add_argument("--context-window", type=int, default=192, help="History length provided to the forecaster.")
    parser.add_argument("--prediction-length", type=int, default=1, help="Forecast horizon in steps.")
    parser.add_argument("--realized-horizon", type=int, default=1, help="Realised return horizon for rewards.")
    parser.add_argument("--fill-method", type=str, default="ffill", help="Optional fill method when aligning timestamps (e.g., ffill, bfill, none).")
    parser.add_argument("--device-map", type=str, default=None, help="Device override for Toto/Chronos (e.g., 'cuda', 'cpu').")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("gymrl.build_features")

    backend_kwargs = {"device_map": args.device_map} if args.device_map else {}
    fill_method = None if args.fill_method and args.fill_method.lower() == "none" else args.fill_method

    config = FeatureBuilderConfig(
        forecast_backend=args.forecast_backend,
        num_samples=args.num_samples,
        context_window=args.context_window,
        prediction_length=args.prediction_length,
        realized_horizon=args.realized_horizon,
        fill_method=fill_method,
    )

    logger.info("Building feature cube (backend=%s, samples=%d) ...", args.forecast_backend, args.num_samples)
    builder = FeatureBuilder(config=config, backend_kwargs=backend_kwargs)
    cube = builder.build_from_directory(args.data_dir)
    save_feature_cache(args.output, cube, extra_metadata={"builder_config": config.__dict__})
    logger.info("Saved feature cache to %s", args.output)


if __name__ == "__main__":
    main()

