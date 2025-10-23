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
    parser.add_argument(
        "--forecast-backend",
        type=str,
        default="toto",
        choices=["toto", "kronos", "chronos", "bootstrap"],
        help="Forecast backend to use.",
    )
    parser.add_argument("--num-samples", type=int, default=2048, help="Number of forecast samples (must satisfy backend requirements).")
    parser.add_argument("--context-window", type=int, default=192, help="History length provided to the forecaster.")
    parser.add_argument("--prediction-length", type=int, default=1, help="Forecast horizon in steps.")
    parser.add_argument("--realized-horizon", type=int, default=1, help="Realised return horizon for rewards.")
    parser.add_argument("--fill-method", type=str, default="ffill", help="Optional fill method when aligning timestamps (e.g., ffill, bfill, none).")
    parser.add_argument("--resample-rule", type=str, default=None, help="Optional pandas offset alias (e.g., '1H') to resample inputs before feature extraction.")
    parser.add_argument("--device-map", type=str, default=None, help="Device override for Toto/Kronos (e.g., 'cuda', 'cpu').")
    parser.add_argument("--kronos-device", type=str, default=None, help="Device override specifically for Kronos forecasts.")
    parser.add_argument(
        "--enforce-common-index",
        action="store_true",
        help="Require identical timestamps across all symbols (defaults to union with forward-fill).",
    )
    parser.add_argument("--kronos-temperature", type=float, default=None, help="Sampling temperature passed to Kronos.")
    parser.add_argument("--kronos-top-p", type=float, default=None, help="Top-p nucleus sampling parameter for Kronos.")
    parser.add_argument("--kronos-top-k", type=int, default=None, help="Top-k sampling parameter for Kronos.")
    parser.add_argument("--kronos-sample-count", type=int, default=None, help="Number of autoregressive samples Kronos draws before averaging.")
    parser.add_argument("--kronos-max-context", type=int, default=None, help="Maximum context tokens provided to Kronos.")
    parser.add_argument("--kronos-clip", type=float, default=None, help="Value clip applied to Kronos inputs.")
    parser.add_argument("--kronos-oom-retries", type=int, default=None, help="OOM retry attempts for Kronos.")
    parser.add_argument("--kronos-jitter-std", type=float, default=None, help="Optional Gaussian jitter (std) applied to Kronos outputs for feature dispersion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("gymrl.build_features")

    backend_kwargs = {}
    if args.device_map:
        backend_kwargs["device_map"] = args.device_map
    if args.kronos_device:
        backend_kwargs["kronos_device"] = args.kronos_device
    if args.kronos_temperature is not None:
        backend_kwargs["kronos_temperature"] = args.kronos_temperature
    if args.kronos_top_p is not None:
        backend_kwargs["kronos_top_p"] = args.kronos_top_p
    if args.kronos_top_k is not None:
        backend_kwargs["kronos_top_k"] = args.kronos_top_k
    if args.kronos_sample_count is not None:
        backend_kwargs["kronos_sample_count"] = args.kronos_sample_count
    if args.kronos_max_context is not None:
        backend_kwargs["kronos_max_context"] = args.kronos_max_context
    if args.kronos_clip is not None:
        backend_kwargs["kronos_clip"] = args.kronos_clip
    if args.kronos_oom_retries is not None:
        backend_kwargs["kronos_oom_retries"] = args.kronos_oom_retries
    if args.kronos_jitter_std is not None:
        backend_kwargs["kronos_jitter_std"] = args.kronos_jitter_std
    fill_method = None if args.fill_method and args.fill_method.lower() == "none" else args.fill_method

    config = FeatureBuilderConfig(
        forecast_backend=args.forecast_backend,
        num_samples=args.num_samples,
        context_window=args.context_window,
        prediction_length=args.prediction_length,
        realized_horizon=args.realized_horizon,
        fill_method=fill_method,
        enforce_common_index=args.enforce_common_index,
        resample_rule=args.resample_rule,
    )

    logger.info("Building feature cube (backend=%s, samples=%d) ...", args.forecast_backend, args.num_samples)
    builder = FeatureBuilder(config=config, backend_kwargs=backend_kwargs)
    cube = builder.build_from_directory(args.data_dir)
    save_feature_cache(args.output, cube, extra_metadata={"builder_config": config.__dict__})
    logger.info("Saved feature cache to %s", args.output)


if __name__ == "__main__":
    main()
