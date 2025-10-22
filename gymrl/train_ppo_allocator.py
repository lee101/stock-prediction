#!/usr/bin/env python3
"""
GymRL PPO allocator training script.

This script wires the feature builder, offline dataset tooling, and the
PortfolioEnv into a Stable-Baselines3 training loop. It is intentionally
modular so it can be invoked from CI, notebooks, or future
``predict_stock_gymrl.py`` integrations.

Usage example:

    uv pip install stable-baselines3 gymnasium torch pandas
    uv pip install -e toto  # Kronos users should also install external/kronos requirements
    python -m gymrl.train_ppo_allocator \
        --data-dir tototraining/trainingdata/train \
        --output-dir gymrl/artifacts \
        --num-timesteps 200000
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import types
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
import torch
from gymrl.cache_utils import load_feature_cache, save_feature_cache
from gymrl.config import OfflineDatasetConfig
from gymrl.eval_utils import evaluate_trained_policy

logger = logging.getLogger("gymrl.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def optional_float(value: str) -> Optional[float]:
    """Parse float arguments that may accept 'none'."""
    if isinstance(value, str) and value.strip().lower() in {"none", "null", "nan"}:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO allocator with GymRL.")
    parser.add_argument("--data-dir", type=Path, default=Path("tototraining/trainingdata/train"), help="Directory of per-symbol CSV files.")
    parser.add_argument(
        "--forecast-backend",
        type=str,
        default="auto",
        choices=["auto", "toto", "kronos", "chronos", "bootstrap"],
        help="Forecasting backend used to build features.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2048,
        help="Number of forecast samples per step (Toto recommends >=2048; Kronos uses deterministic tiling).",
    )
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
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Device override for Toto/Kronos (e.g., 'cuda', 'cpu').",
    )
    parser.add_argument("--kronos-device", type=str, default=None, help="Device override for Kronos forecasts (defaults to cuda:0 if available).")
    parser.add_argument("--kronos-temperature", type=float, default=None, help="Sampling temperature passed to Kronos (defaults to wrapper setting).")
    parser.add_argument("--kronos-top-p", type=float, default=None, help="Top-p nucleus sampling parameter for Kronos.")
    parser.add_argument("--kronos-top-k", type=int, default=None, help="Top-k sampling parameter for Kronos.")
    parser.add_argument("--kronos-sample-count", type=int, default=None, help="Number of autoregressive samples Kronos draws before averaging.")
    parser.add_argument("--kronos-max-context", type=int, default=None, help="Maximum context tokens when running Kronos.")
    parser.add_argument("--kronos-clip", type=float, default=None, help="Clipping applied to Kronos inputs (default 5.0).")
    parser.add_argument("--kronos-oom-retries", type=int, default=None, help="OOM retry count when forecasting with Kronos.")
    parser.add_argument("--kronos-jitter-std", type=float, default=None, help="Optional Gaussian noise (std) added to Kronos forecasts before feature stats.")
    parser.add_argument("--enforce-common-index", action="store_true", help="Require identical timestamps across all symbols (default: union with forward-fill).")
    parser.add_argument("--fill-method", type=str, default="ffill", help="Optional pandas fillna method when aligning timestamps (default: ffill).")
    parser.add_argument("--topk-checkpoints", type=int, default=3, help="Number of top evaluation models to keep during training.")
    parser.add_argument("--topk-eval-freq", type=int, default=4096, help="Frequency (timesteps) for saving top-k evaluation checkpoints.")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Number of evaluation episodes per checkpoint (long episodes recommended).")
    parser.add_argument("--validation-days", type=int, default=21, help="Number of trailing daily steps reserved for validation-only evaluation.")
    parser.add_argument("--features-cache", type=Path, default=None, help="Path to a cached feature NPZ file to load instead of rebuilding.")
    parser.add_argument("--cache-features-to", type=Path, default=None, help="Optional path to persist the generated feature cube for reuse.")
    parser.add_argument("--symbols", type=str, default=None, help="Optional comma-separated subset of symbols to load from the dataset.")
    parser.add_argument("--costs-bps", type=float, default=3.0, help="Baseline proportional trading cost in basis points.")
    parser.add_argument("--turnover-penalty", type=float, default=5e-4, help="Penalty applied to portfolio turnover in the reward.")
    parser.add_argument("--drawdown-penalty", type=float, default=0.0, help="Penalty applied to running drawdown.")
    parser.add_argument("--cvar-penalty", type=float, default=0.0, help="Penalty weight for predicted CVaR inputs.")
    parser.add_argument("--uncertainty-penalty", type=float, default=0.0, help="Penalty weight for forecast dispersion inputs.")
    parser.add_argument("--weight-cap", type=optional_float, default=0.35, help="Maximum per-asset allocation; pass 'none' to disable.")
    parser.add_argument("--allow-short", action="store_true", help="Enable long/short allocations with symmetric leverage.")
    parser.add_argument("--leverage-cap", type=float, default=1.0, help="Gross leverage cap when shorting is enabled.")
    parser.add_argument("--include-cash", dest="include_cash", action="store_true", help="Include a synthetic cash asset (default).")
    parser.add_argument("--no-include-cash", dest="include_cash", action="store_false", help="Disable the synthetic cash asset.")
    parser.add_argument("--cash-return", type=float, default=0.0, help="Per-step deterministic return of the synthetic cash asset.")
    parser.add_argument("--enable-loss-shutdown", action="store_true", help="Enable cooldown gating after unprofitable trades.")
    parser.add_argument("--loss-shutdown-probe-weight", type=float, default=0.05, help="Maximum allocation magnitude allowed during cooldown (probe trade size).")
    parser.add_argument("--loss-shutdown-cooldown", type=int, default=3, help="Number of steps to retain the loss shutdown gate after a loss.")
    parser.add_argument("--loss-shutdown-penalty", type=float, default=0.0, help="Penalty multiplier applied to weights kept in cooldown.")
    parser.add_argument("--loss-shutdown-min-position", type=float, default=1e-4, help="Minimum absolute weight considered active for the shutdown logic.")
    parser.add_argument(
        "--loss-shutdown-return-tolerance",
        type=float,
        default=1e-5,
        help="Absolute net return threshold treated as neutral when updating cooldown state.",
    )
    parser.add_argument("--intraday-leverage-cap", type=float, default=None, help="Optional gross exposure cap applied immediately after actions (long-only leverage).")
    parser.add_argument("--closing-leverage-cap", type=float, default=None, help="Gross exposure cap enforced at market close before carrying positions overnight.")
    parser.add_argument("--leverage-interest-rate", type=float, default=0.0, help="Annual interest rate applied to leverage above 1x when held overnight.")
    parser.add_argument("--trading-days-per-year", type=int, default=252, help="Trading days per year used for leverage interest accrual.")
    parser.add_argument(
        "--policy-dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help="Torch dtype used for PPO policy forward/backward passes (bfloat16 enables autocast).",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use for Stable-Baselines3 (e.g., 'cpu', 'cuda', or 'auto').")
    parser.set_defaults(include_cash=True)
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

    backend_kwargs: Dict[str, object] = {}
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

    selected_symbols = None
    if args.symbols:
        selected_symbols = [symbol.strip() for symbol in str(args.symbols).split(",") if symbol.strip()]

    cube_loaded_from_cache = False
    extra_meta: Dict[str, object] = {}
    backend_label: Optional[str] = builder_config.forecast_backend
    backend_errors: List[str] = []

    if args.features_cache:
        cube, extra_meta = load_feature_cache(args.features_cache)
        extra_meta = dict(extra_meta)
        cube_loaded_from_cache = True
        cached_backend = extra_meta.get("backend_name")
        if isinstance(cached_backend, str):
            backend_label = cached_backend
        cached_errors = extra_meta.get("backend_errors")
        if isinstance(cached_errors, list):
            backend_errors = [str(item) for item in cached_errors]
        elif cached_errors is not None:
            backend_errors = [str(cached_errors)]

        if selected_symbols:
            symbol_lookup = {sym.upper(): idx for idx, sym in enumerate(cube.symbols)}
            missing_symbols = [sym for sym in selected_symbols if sym.upper() not in symbol_lookup]
            if missing_symbols:
                raise ValueError(
                    f"Requested symbols {missing_symbols} not present in feature cube (available: {cube.symbols})"
                )
            indices = [symbol_lookup[sym.upper()] for sym in selected_symbols]
            cube.features = cube.features[:, indices, :]
            cube.realized_returns = cube.realized_returns[:, indices]
            if cube.forecast_cvar is not None:
                cube.forecast_cvar = cube.forecast_cvar[:, indices]
            if cube.forecast_uncertainty is not None:
                cube.forecast_uncertainty = cube.forecast_uncertainty[:, indices]
            cube.symbols = [cube.symbols[idx] for idx in indices]
    else:
        builder = FeatureBuilder(config=builder_config, backend_kwargs=backend_kwargs)
        cube = builder.build_from_directory(args.data_dir, symbols=selected_symbols)
        backend_label = builder.backend_name or builder_config.forecast_backend
        backend_errors = builder.backend_errors
        extra_meta = {
            "builder_config": asdict(builder_config),
            "backend_name": backend_label,
            "backend_errors": backend_errors,
            "selected_symbols": selected_symbols,
        }
        if args.cache_features_to:
            save_feature_cache(Path(args.cache_features_to), cube, extra_metadata=extra_meta)

    if backend_label is not None and "backend_name" not in extra_meta:
        extra_meta["backend_name"] = backend_label
    if "backend_errors" not in extra_meta:
        extra_meta["backend_errors"] = backend_errors

    if selected_symbols is not None:
        extra_meta["selected_symbols"] = selected_symbols
    else:
        extra_meta.setdefault("selected_symbols", list(cube.symbols))

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
    else:
        logger.info("Built feature cube from %s", args.data_dir)

    if backend_label:
        logger.info("Feature backend: %s", backend_label)
    if backend_errors:
        logger.warning("Forecast backend issues: %s", "; ".join(backend_errors))

    env_config = PortfolioEnvConfig(
        costs_bps=args.costs_bps,
        turnover_penalty=args.turnover_penalty,
        drawdown_penalty=args.drawdown_penalty,
        cvar_penalty=args.cvar_penalty,
        uncertainty_penalty=args.uncertainty_penalty,
        weight_cap=args.weight_cap,
        allow_short=args.allow_short,
        loss_shutdown_enabled=args.enable_loss_shutdown,
        loss_shutdown_cooldown=args.loss_shutdown_cooldown,
        loss_shutdown_probe_weight=args.loss_shutdown_probe_weight,
        loss_shutdown_penalty=args.loss_shutdown_penalty,
        loss_shutdown_min_position=args.loss_shutdown_min_position,
        loss_shutdown_return_tolerance=args.loss_shutdown_return_tolerance,
        intraday_leverage_cap=args.intraday_leverage_cap,
        closing_leverage_cap=args.closing_leverage_cap,
        leverage_interest_rate=args.leverage_interest_rate,
        trading_days_per_year=args.trading_days_per_year,
        leverage_cap=args.leverage_cap,
        include_cash=args.include_cash,
        cash_return=args.cash_return,
    )

    if args.behaviour_dataset:
        dataset_config = OfflineDatasetConfig(output_path=str(args.behaviour_dataset), compress=True)
        build_offline_dataset(
            cube,
            env_config=env_config,
            dataset_config=dataset_config,
            behaviour_policy=args.behaviour_policy,
        )
        logger.info("Saved behaviour dataset to %s", args.behaviour_dataset)

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
        device=args.device,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        seed=args.seed,
    )

    if args.policy_dtype == "bfloat16":
        try:
            torch.ones(1, device=model.device, dtype=torch.bfloat16)
        except RuntimeError as exc:
            raise RuntimeError(
                f"policy-dtype bfloat16 requested but not supported on device {model.device}"
            ) from exc
        logger.info("Enabling bfloat16 autocast for PPO policy (device=%s).", model.device)
    else:
        logger.info("Using float32 policy dtype (device=%s).", model.device)

    def _autocast_context() -> contextlib.AbstractContextManager:
        if args.policy_dtype == "bfloat16":
            device_type = "cuda" if model.device.type == "cuda" else "cpu"
            return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    if args.policy_dtype == "bfloat16":
        original_forward = model.policy.forward
        original_predict_values = model.policy.predict_values
        original_predict = model.policy._predict
        original_evaluate_actions = model.policy.evaluate_actions

        def _forward_with_cast(self, obs, deterministic: bool = False):
            actions, values, log_prob = original_forward(obs, deterministic=deterministic)
            return actions.to(torch.float32), values.to(torch.float32), log_prob.to(torch.float32)

        def _predict_values_with_cast(self, obs):
            values = original_predict_values(obs)
            return values.to(torch.float32)

        def _evaluate_actions_with_cast(self, obs, actions):
            values, log_prob, entropy = original_evaluate_actions(obs, actions)
            return values.to(torch.float32), log_prob.to(torch.float32), entropy.to(torch.float32)

        def _predict_with_cast(self, obs, deterministic: bool = False):
            actions = original_predict(obs, deterministic=deterministic)
            return actions.to(torch.float32)

        model.policy.forward = types.MethodType(_forward_with_cast, model.policy)
        model.policy.predict_values = types.MethodType(_predict_values_with_cast, model.policy)
        model.policy._predict = types.MethodType(_predict_with_cast, model.policy)
        model.policy.evaluate_actions = types.MethodType(_evaluate_actions_with_cast, model.policy)

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
        backend_label,
    )

    with _autocast_context():
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
    with _autocast_context():
        validation_metrics = evaluate_trained_policy(model, rollout_env)
    logger.info(
        "Validation (last %d days) -> final value: %.4f, cumulative return: %.2f%%, annualized return: %.2f%%, avg turnover: %.4f, avg trading cost: %.6f, avg interest cost: %.6f, avg close gross: %.3f",
        validation_steps,
        validation_metrics["final_portfolio_value"],
        validation_metrics["cumulative_return"] * 100.0,
        validation_metrics["annualized_return"] * 100.0,
        validation_metrics["average_turnover"],
        validation_metrics["average_trading_cost"],
        validation_metrics.get("average_interest_cost", 0.0),
        validation_metrics.get("average_gross_exposure_close", 0.0),
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
        "env_config": asdict(env_config),
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
        "forecast_backend_used": backend_label,
        "forecast_backend_errors": backend_errors,
        "policy_dtype": args.policy_dtype,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)
    logger.info("Wrote metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
