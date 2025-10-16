#!/usr/bin/env python3
"""Evaluate a saved PPO checkpoint on a cached feature cube."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from stable_baselines3 import PPO

from gymrl.cache_utils import load_feature_cache
from gymrl.config import PortfolioEnvConfig
from gymrl.eval_utils import evaluate_trained_policy
from gymrl.portfolio_env import PortfolioEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO allocator on cached features.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a Stable-Baselines3 checkpoint (.zip).")
    parser.add_argument("--features-cache", type=Path, required=True, help="Feature cache produced via build_features or training run.")
    parser.add_argument("--validation-days", type=int, default=21, help="Number of trailing daily steps used for evaluation.")
    parser.add_argument("--start-index", type=int, default=None, help="Optional explicit start index for evaluation window.")
    parser.add_argument("--turnover-penalty", type=float, default=5e-4, help="Turnover penalty used in the environment.")
    parser.add_argument("--drawdown-penalty", type=float, default=0.0, help="Drawdown penalty weight.")
    parser.add_argument("--cvar-penalty", type=float, default=0.0, help="CVaR penalty weight.")
    parser.add_argument("--uncertainty-penalty", type=float, default=0.0, help="Forecast uncertainty penalty weight.")
    parser.add_argument("--weight-cap", type=float, default=0.35, help="Per-asset weight cap for long-only allocation.")
    parser.add_argument("--no-cash", action="store_true", help="Disable the synthetic cash asset in the evaluation environment.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (INFO/DEBUG/...).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("gymrl.evaluate_policy")

    cube, extra_meta = load_feature_cache(args.features_cache)
    total_steps = cube.features.shape[0]
    validation_steps = max(2, min(args.validation_days, total_steps - 1))
    if args.start_index is not None:
        start_index = max(0, min(args.start_index, total_steps - validation_steps - 1))
    else:
        start_index = max(0, total_steps - validation_steps - 1)

    env_config = PortfolioEnvConfig(
        turnover_penalty=args.turnover_penalty,
        drawdown_penalty=args.drawdown_penalty,
        cvar_penalty=args.cvar_penalty,
        uncertainty_penalty=args.uncertainty_penalty,
        weight_cap=args.weight_cap,
        include_cash=not args.no_cash,
    )

    env = PortfolioEnv(
        cube.features,
        cube.realized_returns,
        config=env_config,
        feature_names=cube.feature_names,
        symbols=cube.symbols,
        timestamps=cube.timestamps,
        forecast_cvar=cube.forecast_cvar,
        forecast_uncertainty=cube.forecast_uncertainty,
        start_index=start_index,
        episode_length=validation_steps,
    )

    logger.info(
        "Evaluating checkpoint %s from start index %d over %d steps (features=%s)",
        args.checkpoint,
        start_index,
        validation_steps,
        args.features_cache,
    )

    model = PPO.load(str(args.checkpoint))
    metrics = evaluate_trained_policy(model, env)

    for key, value in metrics.items():
        logger.info("%s: %s", key, value)

    if extra_meta:
        logger.info("Feature cache metadata: %s", extra_meta)


if __name__ == "__main__":
    main()
