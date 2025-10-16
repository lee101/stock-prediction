"""
Offline dataset generation utilities for GymRL.

This module derives behaviour policy allocations from the feature cube and
simulates them inside ``PortfolioEnv`` to generate (s, a, r, s') tuples suitable
for algorithms such as IQL/CQL via libraries like d3rlpy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .behaviour import blend_policies, kelly_fractional, topk_equal_weight
from .config import OfflineDatasetConfig, PortfolioEnvConfig
from .feature_pipeline import FeatureCube
from .portfolio_env import PortfolioEnv


def generate_behaviour_weights(
    cube: FeatureCube,
    *,
    policy: str = "topk",
    top_k: int = 2,
    threshold: float = 0.0,
    kelly_cap: float = 0.3,
    blend_alpha: float = 0.6,
) -> np.ndarray:
    """
    Compute behaviour policy weights from feature cube.

    Args:
        cube: FeatureCube produced by FeatureBuilder.
        policy: One of {"topk", "kelly", "blended"}.
        top_k: Number of assets for the top-k policy.
        threshold: Minimum forecast score to participate in the allocation.
        kelly_cap: Max per-asset weight for Kelly sizing.
        blend_alpha: Blend coefficient for the blended policy.

    Returns:
        Array of shape (T, N) containing behaviour weights at each time step.
    """

    feature_lookup = {name: idx for idx, name in enumerate(cube.feature_names)}

    def require_feature(name: str) -> np.ndarray:
        if name not in feature_lookup:
            raise KeyError(f"Feature '{name}' is required for behaviour '{policy}'.")
        return cube.features[:, :, feature_lookup[name]]

    if policy == "topk":
        scores = require_feature("forecast_mu")
        weights = topk_equal_weight(scores, k=top_k, threshold=threshold)
    elif policy == "kelly":
        mu = require_feature("forecast_mean_return")
        sigma = require_feature("forecast_sigma")
        weights = kelly_fractional(mu, sigma**2, cap=kelly_cap)
    elif policy == "blended":
        scores = require_feature("forecast_mu")
        mu = require_feature("forecast_mean_return")
        sigma = require_feature("forecast_sigma")
        w_topk = topk_equal_weight(scores, k=top_k, threshold=threshold)
        w_kelly = kelly_fractional(mu, sigma**2, cap=kelly_cap)
        weights = blend_policies(w_topk, w_kelly, alpha=blend_alpha)
    else:
        raise ValueError(f"Unsupported behaviour policy '{policy}'.")

    return weights.astype(np.float32)


def build_offline_dataset(
    cube: FeatureCube,
    *,
    env_config: Optional[PortfolioEnvConfig] = None,
    dataset_config: Optional[OfflineDatasetConfig] = None,
    behaviour_policy: str = "topk",
    behaviour_kwargs: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """
    Simulate behaviour policy inside PortfolioEnv to build offline dataset.

    Returns:
        dataset: Dictionary with arrays ready for serialisation.
        metadata: Ancillary information (config, symbols, timestamps).
    """

    env_config = env_config or PortfolioEnvConfig()
    dataset_config = dataset_config or OfflineDatasetConfig()
    behaviour_kwargs = behaviour_kwargs or {}

    if env_config.allow_short:
        raise NotImplementedError("Offline dataset generation currently supports long-only environments.")

    env = PortfolioEnv(
        features=cube.features,
        realized_returns=cube.realized_returns,
        config=env_config,
        feature_names=cube.feature_names,
        symbols=cube.symbols,
        timestamps=cube.timestamps,
        forecast_cvar=cube.forecast_cvar,
        forecast_uncertainty=cube.forecast_uncertainty,
        append_portfolio_state=True,
    )

    behaviour_weights = generate_behaviour_weights(
        cube,
        policy=behaviour_policy,
        **behaviour_kwargs,
    )

    observations = []
    next_observations = []
    actions = []
    rewards = []
    dones = []
    portfolio_values = []
    turnover_series = []

    obs, _ = env.reset()
    episode_steps = len(cube.timestamps) - 1
    for t in range(episode_steps):
        weight_vector = behaviour_weights[t]
        observations.append(obs)
        obs_next, reward, terminated, truncated, info = env.step_with_weights(weight_vector)

        actions.append(weight_vector.astype(np.float32))
        rewards.append(reward)
        next_observations.append(obs_next)
        dones.append(float(terminated))
        portfolio_values.append(info["portfolio_value"])
        turnover_series.append(info["turnover"])

        obs = obs_next
        if terminated or truncated:
            break

    observations = np.stack(observations).astype(np.float32)
    next_observations = np.stack(next_observations).astype(np.float32)
    actions = np.stack(actions).astype(np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    portfolio_values = np.asarray(portfolio_values, dtype=np.float32)
    turnover_series = np.asarray(turnover_series, dtype=np.float32)

    if dataset_config.normalize_rewards:
        mean = rewards.mean()
        std = rewards.std()
        if std > 1e-6:
            rewards = (rewards - mean) / std

    dataset = {
        "observations": observations,
        "actions_weights": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "dones": dones,
        "portfolio_values": portfolio_values,
        "turnover": turnover_series,
        "timestamps": np.asarray(cube.timestamps[: len(observations)], dtype="datetime64[ns]"),
    }

    metadata: Dict[str, object] = {
        "symbols": cube.symbols,
        "feature_names": cube.feature_names,
        "behaviour_policy": behaviour_policy,
        "behaviour_kwargs": behaviour_kwargs,
        "env_config": env_config.__dict__,
        "dataset_config": dataset_config.__dict__,
        "allow_short": env_config.allow_short,
        "num_steps": int(observations.shape[0]),
        "num_assets": int(len(cube.symbols)),
        "forecast_cvar_available": cube.forecast_cvar is not None,
        "forecast_uncertainty_available": cube.forecast_uncertainty is not None,
    }

    output_path = dataset_config.output_path
    if output_path and not dataset_config.metadata_only:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        saver = np.savez_compressed if dataset_config.compress else np.savez
        saver(output_path, **dataset)

        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        with meta_path.open("w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=2, default=str)

    return dataset, metadata


__all__ = ["generate_behaviour_weights", "build_offline_dataset"]
