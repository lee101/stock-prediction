from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from pathlib import Path

from hftraining.data_utils import load_local_stock_data
from pufferlib import emulation, postprocess


@dataclass(frozen=True)
class MarketEnvConfig:
    data_dir: str = "trainingdata"
    tickers: Optional[Sequence[str]] = None
    context_len: int = 128
    episode_len: int = 256
    horizon: int = 1
    fee_bps: float = 0.5
    slippage_bps: float = 1.5
    leverage_limit: float = 1.5
    device: str = "auto"
    precision: str = "bf16"
    seed: Optional[int] = None


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_dtype(key: str) -> torch.dtype:
    lower = key.lower()
    if lower in {"bf16", "bfloat16"}:
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if lower in {"fp16", "float16"}:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return torch.float32


def _smooth_abs(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(x * x + eps)


def _load_market_frames(config: MarketEnvConfig) -> Dict[str, np.ndarray]:
    if config.tickers:
        symbols = list(config.tickers)
    else:
        data_path = Path(config.data_dir)
        csvs = sorted(data_path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV files found under '{config.data_dir}'."
            )
        symbols = [path.stem for path in csvs]

    frames = load_local_stock_data(symbols, data_dir=config.data_dir)
    if not frames:
        raise FileNotFoundError(
            f"No price CSVs found under '{config.data_dir}'. Provide tickers with data."
        )
    aligned = {}
    # Align on minimum available length across tickers to avoid ragged tensors.
    min_len = min(len(df) for df in frames.values())
    feature_cols = ["open", "high", "low", "close", "volume"]
    for symbol, df in frames.items():
        df = df.sort_values("timestamps" if "timestamps" in df.columns else df.columns[0]).tail(min_len)
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Ticker {symbol} missing required columns: {missing}")
        arr = df[feature_cols].to_numpy(np.float32)
        aligned[symbol] = arr
    return aligned


class MarketEnv(gym.Env):
    """Single-asset trading environment with differentiable torch rewards."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        data_dir: str = "trainingdata",
        tickers: Optional[Sequence[str]] = None,
        context_len: int = 128,
        episode_len: int = 256,
        horizon: int = 1,
        fee_bps: float = 0.5,
        slippage_bps: float = 1.5,
        leverage_limit: float = 1.5,
        device: str = "auto",
        precision: str = "bf16",
        seed: Optional[int] = None,
    ):
        super().__init__()
        if context_len < 8:
            raise ValueError("context_len must be >= 8 to provide meaningful history.")
        if episode_len < 1:
            raise ValueError("episode_len must be >= 1.")

        config = MarketEnvConfig(
            data_dir=os.path.expanduser(data_dir),
            tickers=tickers,
            context_len=context_len,
            episode_len=episode_len,
            horizon=horizon,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            leverage_limit=leverage_limit,
            device=device,
            precision=precision,
            seed=seed,
        )

        self.device = _resolve_device(config.device)
        self.tensor_dtype = _resolve_dtype(config.precision)
        self.rng = np.random.default_rng(config.seed)

        raw_frames = _load_market_frames(config)
        self.symbols = sorted(raw_frames.keys())
        stacked = np.stack([raw_frames[s] for s in self.symbols], axis=0)
        prices = stacked[..., 3]  # close prices

        # Compute log prices/returns and normalised volume features.
        log_close = np.log(np.clip(prices, 1e-6, None))
        returns = np.diff(log_close, axis=1, prepend=log_close[:, :1])
        volume = stacked[..., 4]
        volume_z = (volume - volume.mean(axis=1, keepdims=True)) / (volume.std(axis=1, keepdims=True) + 1e-6)

        feature_stack = np.stack(
            [
                (log_close - log_close.mean(axis=1, keepdims=True)),
                returns,
                volume_z,
            ],
            axis=-1,
        ).astype(np.float32)

        self.feature_tensor = torch.from_numpy(feature_stack).to(self.device, torch.float32)
        self.return_tensor = torch.from_numpy(np.exp(returns) - 1.0).to(self.device, torch.float32)
        self.close_prices = torch.from_numpy(prices).to(self.device, torch.float32)

        self.num_assets = len(self.symbols)
        self.sequence_len = self.feature_tensor.shape[1]
        self.context_len = context_len
        self.episode_len = episode_len
        self.horizon = horizon
        self.leverage_limit = float(leverage_limit)
        self.transaction_cost = float(fee_bps) / 10_000.0
        self.slippage_cost = float(slippage_bps) / 10_000.0

        if self.sequence_len <= self.context_len + 2:
            raise ValueError("Not enough timesteps after alignment for the requested context length.")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        extra_feats = 3  # position, log value, leverage
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.context_len, feature_stack.shape[-1] + extra_feats),
            dtype=np.float32,
        )

        self._current_symbol: int = 0
        self._cursor: int = self.context_len
        self._episode_steps: int = 0
        self._episode_end: int = self.sequence_len - self.horizon - 1
        self._position = torch.zeros(1, device=self.device, dtype=torch.float32)
        self._portfolio_value = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        self._reward_trace = torch.zeros(self.episode_len, device=self.device, dtype=torch.float32)
        self._reward_index = 0

    # ---------------------------------------------------------------------- #
    # Gym API                                                                #
    # ---------------------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._current_symbol = int(self.rng.integers(0, self.num_assets))
        symbol_len = self.sequence_len - self.horizon - 2
        start = int(self.rng.integers(self.context_len, symbol_len - self.episode_len))
        self._cursor = start
        self._episode_end = min(start + self.episode_len, symbol_len)
        self._episode_steps = 0
        self._position.zero_()
        self._portfolio_value.fill_(1.0)
        self._reward_trace.zero_()
        self._reward_index = 0

        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        if action.shape != (1,):
            raise ValueError(f"Expected action shape (1,), received {action.shape}")

        action_tensor = torch.as_tensor(action, dtype=self.tensor_dtype, device=self.device)
        target_position = torch.tanh(action_tensor).to(torch.float32)
        prev_position = self._position

        # Compute trading costs with smooth penalties to keep gradients stable.
        delta = target_position - prev_position
        turnover = _smooth_abs(delta)
        fee = turnover * self.transaction_cost
        slip = turnover * (prev_position.abs() + target_position.abs()) * (self.slippage_cost * 0.5)

        returns = self.return_tensor[self._current_symbol, self._cursor]
        pnl = prev_position * returns
        reward_tensor = pnl - fee - slip

        self._portfolio_value = self._portfolio_value * (1.0 + reward_tensor)
        self._position = target_position.clamp(-self.leverage_limit, self.leverage_limit)

        self._reward_trace[self._reward_index] = reward_tensor.squeeze()
        self._reward_index = (self._reward_index + 1) % self._reward_trace.numel()

        self._cursor += 1
        self._episode_steps += 1
        terminated = self._cursor >= self._episode_end

        # New observation is pulled from the updated cursor.
        observation = self._get_observation()
        info = {
            "symbol": self.symbols[self._current_symbol],
            "position": float(self._position.item()),
            "portfolio_value": float(self._portfolio_value.item()),
            "reward_tensor": reward_tensor.detach(),
        }

        return observation, float(reward_tensor.item()), terminated, False, info

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _get_observation(self) -> np.ndarray:
        start = self._cursor - self.context_len
        end = self._cursor
        core = self.feature_tensor[self._current_symbol, start:end]

        # Broadcast portfolio diagnostics across the context window.
        position_track = self._position.expand(self.context_len, 1)
        value_track = self._portfolio_value.log().expand(self.context_len, 1)
        leverage_track = torch.clamp(self._position.abs(), max=self.leverage_limit).expand(self.context_len, 1)

        obs = torch.cat([core, position_track, value_track, leverage_track], dim=-1)
        return obs.to(torch.float32).cpu().numpy()


def make_market_env(buf=None, **kwargs):
    env = MarketEnv(**kwargs)
    env = postprocess.EpisodeStats(env)
    return emulation.GymnasiumPufferEnv(env=env, buf=buf)


__all__ = ["MarketEnv", "MarketEnvConfig", "make_market_env"]
