#!/usr/bin/env python3
"""
GymRL PPO allocator training script.

This script wires the feature builder, offline dataset tooling, and the
PortfolioEnv into a Stable-Baselines3 training loop. It is intentionally
modular so it can be invoked from CI, notebooks, or future
``predict_stock_gymrl.py`` integrations.

Usage example:

    uv pip install stable-baselines3 gymnasium torch pandas chronos
    python -m gymrl.train_ppo_allocator \
        --data-dir tototraining/trainingdata/train \
        --output-dir gymrl/artifacts \
        --num-timesteps 200000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from gymrl import (
    FeatureBuilder,
    FeatureBuilderConfig,
    PortfolioEnv,
    PortfolioEnvConfig,
    build_offline_dataset,
)
from gymrl.cache_utils import load_feature_cache, save_feature_cache
from gymrl.config import OfflineDatasetConfig
from gymrl.eval_utils import evaluate_trained_policy

logger = logging.getLogger("gymrl.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO allocator with GymRL.")
    parser.add_argument("--data-dir", type=Path, default=Path("tototraining/trainingdata/train"), help="Directory of per-symbol CSV files.")
    parser.add_argument("--forecast-backend", type=str, default="auto", choices=["auto", "toto", "chronos", "bootstrap"], help="Forecasting backend used to build features.")
    parser.add_argument("--num-samples", type=int, default=2048, help="Number of forecast samples per step (2048 recommended for Toto/Chronos).")
    parser.add_argument("--context-window", type=int, default=192, help="History length provided to the forecaster.")
    parser.add_argument("--prediction-length", type=int, default=1, help="Forecast horizon in steps.")
    parser.add_argument("--realized-horizon", type=int, default=1, help="Realised return horizon for rewards.")
    parser.add_argument("--train-fraction", type=float, default=0.75, help="Fraction of timeline used for PPO training before reserving validation steps.")
    parser.add_argument("--num-timesteps", type=int, default=500_000, help="Total PPO timesteps.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="PPO learning rate.")
    parser.add_argument("--batch-size", type=int, default=512, help="PPO minibatch size.")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps to run per environment update.")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy regularisation coefficient.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument("--tensorboard-log", type=Path, default=Path("gymrl/runs"), help="TensorBoard log directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("gymrl/artifacts"), help="Directory for checkpoints and artefacts.")
    parser.add_argument("--save-frequency", type=int, default=50_000, help="Checkpoint frequency (timesteps).")
    parser.add_argument("--behaviour-dataset", type=Path, default=None, help="Optional path to save offline behaviour dataset (.npz).")
    parser.add_argument("--behaviour-policy", type=str, default="topk", choices=["topk", "kelly", "blended"], help="Behaviour policy flavour for offline dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device-map", type=str, default=None, help="Device override for Toto/Chronos (e.g., 'cuda', 'cpu').")
    parser.add_argument("--enforce-common-index", action="store_true", help="Require identical timestamps across all symbols (default: union with forward-fill).")
    parser.add_argument("--fill-method", type=str, default="ffill", help="Optional pandas fillna method when aligning timestamps (default: ffill).")
    parser.add_argument("--topk-checkpoints", type=int, default=3, help="Number of top evaluation models to keep during training.")
    parser.add_argument("--topk-eval-freq", type=int, default=4096, help="Frequency (timesteps) for saving top-k evaluation checkpoints.")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Number of evaluation episodes per checkpoint (long episodes recommended).")
    parser.add_argument("--validation-days", type=int, default=21, help="Number of trailing daily steps reserved for validation-only evaluation.")
    parser.add_argument("--features-cache", type=Path, default=None, help="Path to a cached feature NPZ file to load instead of rebuilding.")
    parser.add_argument("--cache-features-to", type=Path, default=None, help="Optional path to persist the generated feature cube for reuse.")
    return parser.parse_args()


def slice_indices(total_steps: int, train_fraction: float, validation_steps: int) -> Tuple[int, int, int]:
    validation_steps = max(2, min(validation_steps, total_steps - 32))
    desired_train = max(32, int(total_steps * train_fraction))
    max_train = max(32, total_steps - validation_steps)
    train_steps = min(desired_train, max_train)
    eval_start = train_steps
    return train_steps, eval_start, validation_steps


def make_env_factory(
    features: np.ndarray,
    realized_returns: np.ndarray,
    cube_meta: Dict[str, object],
    env_config: PortfolioEnvConfig,
    *,
    start_index: int,
    episode_length: int,
) -> Callable[[], PortfolioEnv]:
    def _factory() -> PortfolioEnv:
        return PortfolioEnv(
            features=features,
            realized_returns=realized_returns,
            config=env_config,
            feature_names=cube_meta["feature_names"],
            symbols=cube_meta["symbols"],
            timestamps=cube_meta["timestamps"],
            forecast_cvar=cube_meta.get("forecast_cvar"),
            forecast_uncertainty=cube_meta.get("forecast_uncertainty"),
            append_portfolio_state=True,
            start_index=start_index,
            episode_length=episode_length,
        )

    return _factory


class TopKCheckpointCallback(BaseCallback):
    """Save top-k checkpoints based on evaluation reward."""

    def __init__(
        self,
        eval_env: DummyVecEnv,
        save_dir: Path,
        *,
        top_k: int = 3,
        eval_freq: int = 4096,
        n_eval_episodes: int = 1,
    ) -> None:
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.save_dir = Path(save_dir)
        self.top_k = max(1, top_k)
        self.eval_freq = max(1, eval_freq)
        self.n_eval_episodes = max(1, n_eval_episodes)
        self._leaderboard: List[Dict[str, float]] = []

    def _init_callback(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps == 0 or (self.num_timesteps % self.eval_freq) != 0:
            return True

        mean_reward, _ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            warn=False,
        )

        save_candidate = False
        if len(self._leaderboard) < self.top_k:
            save_candidate = True
        else:
            worst_reward = min(entry["reward"] for entry in self._leaderboard)
            save_candidate = mean_reward > worst_reward

        if save_candidate:
            checkpoint_path = self.save_dir / f"step_{self.num_timesteps}_reward_{mean_reward:.4f}.zip"
            self.model.save(str(checkpoint_path))
            self._leaderboard.append({"reward": float(mean_reward), "path": str(checkpoint_path)})
            self._leaderboard.sort(key=lambda item: item["reward"], reverse=True)

            while len(self._leaderboard) > self.top_k:
                removed = self._leaderboard.pop()
                try:
                    Path(removed["path"]).unlink(missing_ok=True)
                except Exception:
                    pass

            if self.verbose:
                rewards = ", ".join(f"{entry['reward']:.4f}" for entry in self._leaderboard)
                logger.info("Top-%d checkpoint update (rewards=%s)", self.top_k, rewards)

        return True


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_log.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    backend_kwargs = {"device_map": args.device_map} if args.device_map else {}
    fill_method = None
    if args.fill_method:
        fill_method = None if args.fill_method.lower() == "none" else args.fill_method

    builder_config = FeatureBuilderConfig(
        forecast_backend=args.forecast_backend,
        num_samples=args.num_samples,
        context_window=args.context_window,
        prediction_length=args.prediction_length,
        realized_horizon=args.realized_horizon,
        enforce_common_index=args.enforce_common_index,
        fill_method=fill_method,
    )

    cube_loaded_from_cache = False
    extra_meta: Dict[str, object] = {}
    if args.features_cache:
        cube, extra_meta = load_feature_cache(args.features_cache)
        cube_loaded_from_cache = True
    else:
        builder = FeatureBuilder(config=builder_config, backend_kwargs=backend_kwargs)
        cube = builder.build_from_directory(args.data_dir)
        if args.cache_features_to:
            save_feature_cache(Path(args.cache_features_to), cube, extra_metadata={"builder_config": builder_config.__dict__})

    cube_meta = {
        "feature_names": cube.feature_names,
        "symbols": cube.symbols,
        "timestamps": cube.timestamps,
        "forecast_cvar": cube.forecast_cvar,
        "forecast_uncertainty": cube.forecast_uncertainty,
    }

    if cube_loaded_from_cache and args.features_cache:
        logger.info("Loaded feature cube from cache %s", args.features_cache)
    elif args.cache_features_to:
        logger.info("Saved feature cube to %s", args.cache_features_to)

    if args.behaviour_dataset:
        dataset_config = OfflineDatasetConfig(output_path=str(args.behaviour_dataset), compress=True)
        build_offline_dataset(
            cube,
            env_config=PortfolioEnvConfig(),
            dataset_config=dataset_config,
            behaviour_policy=args.behaviour_policy,
        )
        logger.info("Saved behaviour dataset to %s", args.behaviour_dataset)

    env_config = PortfolioEnvConfig(
        costs_bps=3.0,
        turnover_penalty=5e-4,
        drawdown_penalty=0.0,
        cvar_penalty=0.0,
        uncertainty_penalty=0.0,
        weight_cap=0.35,
        allow_short=False,
        include_cash=True,
    )

    total_steps = cube.features.shape[0]
    train_steps, eval_start, validation_steps = slice_indices(total_steps, args.train_fraction, args.validation_days)
    train_episode_len = max(8, train_steps - 1)
    eval_episode_len = max(8, validation_steps)

    train_env = DummyVecEnv(
        [make_env_factory(cube.features, cube.realized_returns, cube_meta, env_config, start_index=0, episode_length=train_episode_len)]
    )
    eval_env = DummyVecEnv(
        [
            make_env_factory(
                cube.features,
                cube.realized_returns,
                cube_meta,
                env_config,
                start_index=eval_start,
                episode_length=eval_episode_len,
            )
        ]
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(args.tensorboard_log),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        seed=args.seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_frequency // args.n_steps, 1),
        save_path=str(args.output_dir),
        name_prefix="ppo_allocator",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.output_dir / "best"),
        log_path=str(args.output_dir / "eval"),
        eval_freq=args.n_steps,
        deterministic=True,
    )

    topk_callback = TopKCheckpointCallback(
        eval_env,
        save_dir=args.output_dir / "topk",
        top_k=args.topk_checkpoints,
        eval_freq=args.topk_eval_freq,
        n_eval_episodes=args.eval_episodes,
    )

    logger.info(
        "Starting PPO training with %d timesteps (train steps=%d, eval start idx=%d, backend=%s).",
        args.num_timesteps,
        train_steps,
        eval_start,
        builder_config.forecast_backend,
    )

    model.learn(total_timesteps=args.num_timesteps, callback=[checkpoint_callback, eval_callback, topk_callback])
    model_path = args.output_dir / "ppo_allocator_final.zip"
    model.save(str(model_path))
    logger.info("Saved final PPO model to %s", model_path)

    rollout_env = make_env_factory(
        cube.features,
        cube.realized_returns,
        cube_meta,
        env_config,
        start_index=eval_start,
        episode_length=eval_episode_len,
    )()
    validation_metrics = evaluate_trained_policy(model, rollout_env)
    logger.info(
        "Validation (last %d days) -> final value: %.4f, cumulative return: %.2f%%, avg turnover: %.4f, avg trading cost: %.6f",
        validation_steps,
        validation_metrics["final_portfolio_value"],
        validation_metrics["cumulative_return"] * 100.0,
        validation_metrics["average_turnover"],
        validation_metrics["average_trading_cost"],
    )

    topk_records = [
        {"reward": entry["reward"], "path": entry["path"]}
        for entry in topk_callback._leaderboard
    ]
    if topk_records:
        logger.info(
            "Top-%d checkpoints: %s",
            args.topk_checkpoints,
            ", ".join(f"{rec['reward']:.4f}" for rec in topk_records),
        )

    metadata_path = args.output_dir / "training_metadata.json"
    args_serializable = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    metadata_payload = {
        "args": args_serializable,
        "env_config": env_config.__dict__,
        "train_steps": train_steps,
        "eval_start": eval_start,
        "validation_steps": validation_steps,
        "num_features": len(cube.feature_names),
        "num_assets": len(cube.symbols),
        "total_steps": total_steps,
        "validation_metrics": validation_metrics,
        "topk_checkpoints": topk_records,
        "features_cache_loaded": cube_loaded_from_cache,
        "features_cache_path": str(args.features_cache) if args.features_cache else None,
        "features_cache_written": str(args.cache_features_to) if args.cache_features_to else None,
        "feature_extra_metadata": extra_meta,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)
    logger.info("Wrote metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
