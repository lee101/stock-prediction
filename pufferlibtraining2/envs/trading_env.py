from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import gymnasium as gym
import torch

import pufferlib.emulation
import pufferlib.vector

from pufferlibtraining.envs.stock_env import StockTradingEnv

from ..config import EnvConfig, TrainingPlan, VecConfig


class RewardScaleWrapper(gym.RewardWrapper):
    """Scale rewards emitted by the base environment."""

    def __init__(self, env: gym.Env, scale: float) -> None:
        super().__init__(env)
        self._scale = float(scale)

    def reward(self, reward: float) -> float:
        return float(reward) * self._scale


def resolve_device(plan: TrainingPlan) -> torch.device:
    device_name = plan.env.device or plan.vec.device or "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Set env.device/vec.device to 'cpu'.")
    return device


def build_env_creator(
    *,
    plan: TrainingPlan,
    asset_frames: Dict[str, "pandas.DataFrame"],
    device: torch.device,
) -> Callable[[], gym.Env]:
    env_kwargs = dict(
        asset_frames=asset_frames,
        window_size=plan.data.window_size,
        initial_balance=plan.env.initial_balance,
        leverage_limit=plan.env.leverage_limit,
        borrowing_cost_annual=plan.env.borrowing_cost_annual,
        trading_days_per_year=plan.env.trading_days_per_year,
        transaction_cost_bps=plan.env.transaction_cost_bps,
        spread_bps=plan.env.spread_bps,
        max_intraday_leverage=plan.env.max_intraday_leverage,
        max_overnight_leverage=plan.env.max_overnight_leverage,
        trade_timing=plan.env.trade_timing,
        risk_scale=plan.env.risk_scale,
        feature_columns=plan.data.feature_columns,
        device=device,
    )

    def _base_env() -> gym.Env:
        env = StockTradingEnv(**env_kwargs)
        if plan.env.reward_scale != 1.0:
            env = RewardScaleWrapper(env, plan.env.reward_scale)
        return env

    def _puffer_env() -> gym.Env:
        return pufferlib.emulation.GymnasiumPufferEnv(env_creator=_base_env)

    return _puffer_env


def make_vecenv(plan: TrainingPlan, asset_frames: Dict[str, "pandas.DataFrame"]):
    """
    Construct a vectorised environment using PufferLib's high-performance
    multiprocess backend.
    """

    device = resolve_device(plan)
    env_creator = build_env_creator(plan=plan, asset_frames=asset_frames, device=device)
    env_creators = [env_creator] * plan.vec.num_envs
    env_args = [[] for _ in range(plan.vec.num_envs)]
    env_kwargs = [{} for _ in range(plan.vec.num_envs)]

    vec_kwargs = plan.vec.as_kwargs()
    backend = vec_kwargs.pop("backend")
    vecenv = pufferlib.vector.make(
        env_creators,
        env_args=env_args,
        env_kwargs=env_kwargs,
        backend=backend,
        **vec_kwargs,
    )
    vecenv.device = device
    return vecenv


__all__ = ["make_vecenv", "RewardScaleWrapper", "resolve_device"]
