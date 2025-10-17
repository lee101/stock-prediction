"""
Helpers to integrate gymrl's PortfolioEnv into hftraining workflows.

This module provides simple factory functions to build a PortfolioEnv from a
FeatureCube and optional wrappers (e.g., observation normalization).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
from pathlib import Path

import numpy as np

from gymrl.feature_pipeline import FeatureBuilder, FeatureCube
from gymrl.config import FeatureBuilderConfig, PortfolioEnvConfig
from gymrl.portfolio_env import PortfolioEnv
from gymrl.wrappers import ObservationNormalizer, NormalizerConfig


def build_feature_cube_from_directory(
    data_dir: str | Path,
    *,
    symbols: Optional[Sequence[str]] = None,
    feature_config: Optional[FeatureBuilderConfig] = None,
    price_column: str = "close",
) -> FeatureCube:
    builder = FeatureBuilder(feature_config or FeatureBuilderConfig())
    return builder.build_from_directory(Path(data_dir), symbols=symbols, price_column=price_column)


def build_env_from_cube(
    cube: FeatureCube,
    env_config: Optional[PortfolioEnvConfig] = None,
    *,
    append_portfolio_state: bool = True,
    start_index: int = 0,
    episode_length: Optional[int] = None,
    normalise_obs: bool = False,
    normaliser_cfg: Optional[NormalizerConfig] = None,
) -> PortfolioEnv:
    env = PortfolioEnv(
        cube.features,
        cube.realized_returns,
        feature_names=cube.feature_names,
        symbols=cube.symbols,
        timestamps=cube.timestamps,
        forecast_cvar=cube.forecast_cvar,
        forecast_uncertainty=cube.forecast_uncertainty,
        config=env_config or PortfolioEnvConfig(),
        append_portfolio_state=append_portfolio_state,
        start_index=start_index,
        episode_length=episode_length,
    )

    if normalise_obs:
        env = ObservationNormalizer(env, config=normaliser_cfg)
    return env


def make_env_factory(
    cube: FeatureCube,
    env_config: Optional[PortfolioEnvConfig] = None,
    *,
    append_portfolio_state: bool = True,
    normalise_obs: bool = False,
    normaliser_cfg: Optional[NormalizerConfig] = None,
) -> Callable[[], PortfolioEnv]:
    def _factory() -> PortfolioEnv:
        return build_env_from_cube(
            cube,
            env_config=env_config,
            append_portfolio_state=append_portfolio_state,
            normalise_obs=normalise_obs,
            normaliser_cfg=normaliser_cfg,
        )

    return _factory


__all__ = [
    "build_feature_cube_from_directory",
    "build_env_from_cube",
    "make_env_factory",
]

