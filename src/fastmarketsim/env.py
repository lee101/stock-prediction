from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas optional for synthetic runs
    pd = None

from .config import DEFAULTS as SIM_DEFAULTS, build_sim_config
from .module import load_extension


WRAPPER_DEFAULTS: dict[str, Any] = {
    "action_space": "continuous",
    "data_root": None,
    "device": "cpu",
    "end_date": None,
    "episode_length": None,
    "inv_penalty": 0.0,
    "is_crypto": False,
    "price_key": "close",
    "random_reset": False,
    "reward_scale": 1.0,
    "start_date": None,
    "start_index": None,
    "symbol": None,
    "synth_T": 200_000,
    "synth_mu": 0.0,
    "synth_sigma": 0.02,
}


def _cfg_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"Unsupported config type {type(cfg)!r}; expected mapping or dataclass")


def _resolve_device(name: str) -> torch.device:
    requested = torch.device(name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


class FastMarketEnv(gym.Env):
    """Gym wrapper around the accelerated C++/LibTorch market simulator."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: Optional[torch.Tensor] = None,
        exog: Optional[torch.Tensor] = None,
        price_columns: Optional[Tuple[str, ...]] = None,
        cfg: Any = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        merged = {**WRAPPER_DEFAULTS, **SIM_DEFAULTS, **_cfg_dict(cfg)}
        if device is not None:
            merged["device"] = device
        self.cfg = merged
        self.device = _resolve_device(str(self.cfg["device"]))
        self._g = torch.Generator(device="cpu").manual_seed(int(self.cfg["seed"]))

        if str(self.cfg["mode"]).lower() == "maxdiff":
            raise ValueError(
                "FastMarketEnv does not support maxdiff limit-parameterized actions; use the python backend."
            )
        if float(self.cfg.get("inv_penalty", 0.0)) != 0.0:
            raise ValueError("FastMarketEnv does not support inv_penalty; use the python backend.")

        features, date_index, resolved_columns = self._resolve_data(prices, exog, price_columns)
        self.features = features.to(torch.float32).contiguous()
        self.date_index = date_index
        self.feature_columns = resolved_columns
        self.T, self.F = self.features.shape
        if self.T <= int(self.cfg["context_len"]) + max(1, int(self.cfg["horizon"])):
            raise ValueError("Insufficient history to satisfy context and horizon requirements.")

        self._extension = load_extension()
        sim_cfg = build_sim_config(self.cfg)
        crypto_mask = torch.tensor([bool(self.cfg["is_crypto"])], dtype=torch.bool)
        self._sim = self._extension.MarketSimulator(
            sim_cfg,
            self.features.unsqueeze(0),
            crypto_mask,
            str(self.device),
        )

        self._init_spaces()
        self._reset_state()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._g.manual_seed(int(seed))
        self._reset_state(options=options)
        raw_obs = self._sim.reset(int(self.start_index))
        observation = self._augment_observation(raw_obs[0])
        info = {
            "start_index": int(self.start_index),
            "episode_end": int(self.episode_end),
            "timestamp": self._timestamp(),
        }
        return observation, info

    def step(self, action: Any):
        action_tensor = self._prepare_action(action)
        result = self._sim.step(action_tensor)
        self.cursor += 1

        self.position = float(result["position"][0].item())
        self.equity = float(result["equity"][0].item())

        terminated = bool(result["done"][0].item() or self.cursor >= self.episode_end or self.equity <= 0.0)
        truncated = False
        reward = float(result["reward"][0].item()) * float(self.cfg.get("reward_scale", 1.0))
        observation = self._augment_observation(result["obs"][0])
        info = {
            "gross_pnl": float(result["gross"][0].item()),
            "trading_cost": float(result["trade_cost"][0].item()),
            "financing_cost": float(result["financing_cost"][0].item()),
            "deleverage_cost": float(result["deleverage_cost"][0].item()),
            "deleverage_notional": float(result["deleverage_notional"][0].item()),
            "position": self.position,
            "equity": self.equity,
            "timestamp": self._timestamp(),
            "mode": str(self.cfg["mode"]).lower(),
        }
        return observation, reward, terminated, truncated, info

    def render(self) -> None:  # pragma: no cover - rendering not implemented
        return None

    def close(self) -> None:  # pragma: no cover - no external resources
        return None

    def _resolve_data(
        self,
        prices: Optional[torch.Tensor],
        exog: Optional[torch.Tensor],
        price_columns: Optional[Tuple[str, ...]],
    ) -> tuple[torch.Tensor, Any | None, tuple[str, ...]]:
        if prices is not None:
            price_tensor, resolved_columns = self._prepare_prices(prices, price_columns)
            if exog is None:
                return price_tensor, None, resolved_columns
            exog_tensor = exog.to(torch.float32)
            if exog_tensor.ndim != 2 or exog_tensor.shape[0] != price_tensor.shape[0]:
                raise ValueError("exog must be a 2-D tensor with the same number of rows as prices")
            exog_columns = tuple(f"exog_{idx}" for idx in range(exog_tensor.shape[1]))
            return torch.cat([price_tensor, exog_tensor], dim=1), None, resolved_columns + exog_columns

        if pd is None:
            synth = self._make_synth_prices(
                int(self.cfg["synth_T"]),
                float(self.cfg["synth_mu"]),
                float(self.cfg["synth_sigma"]),
            )
            return synth, None, tuple(["open", "high", "low", "close"])

        root = Path(str(self.cfg.get("data_root") or "trainingdata")).expanduser().resolve()
        candidate_paths = []
        symbol = self.cfg.get("symbol")
        if symbol:
            upper = str(symbol).upper()
            candidate_paths.extend(sorted(root.glob(f"**/{upper}.csv")))
            candidate_paths.extend(sorted(root.glob(f"**/{upper}_*.csv")))
        if not candidate_paths:
            candidate_paths = sorted(root.glob("**/*.csv"))
        if not candidate_paths:
            synth = self._make_synth_prices(
                int(self.cfg["synth_T"]),
                float(self.cfg["synth_mu"]),
                float(self.cfg["synth_sigma"]),
            )
            return synth, None, tuple(["open", "high", "low", "close"])

        frame = self._read_csv(candidate_paths[0])
        price_cols = self._infer_price_columns(frame)
        price_tensor, resolved_columns = self._prepare_prices(
            torch.from_numpy(frame[list(price_cols)].to_numpy(dtype=np.float32)),
            tuple(price_cols),
        )
        exog_cols = [col for col in frame.columns if col not in price_cols and pd.api.types.is_numeric_dtype(frame[col])]
        if not exog_cols:
            return price_tensor, frame.index.copy(), resolved_columns

        exog_tensor = torch.from_numpy(frame[exog_cols].to_numpy(dtype=np.float32))
        features = torch.cat([price_tensor, exog_tensor], dim=1)
        return features, frame.index.copy(), resolved_columns + tuple(exog_cols)

    def _prepare_prices(
        self,
        prices: torch.Tensor,
        price_columns: Optional[Tuple[str, ...]],
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        tensor = prices.to(torch.float32)
        if tensor.ndim != 2:
            raise ValueError("prices must be a 2-D tensor shaped [T, F]")

        if price_columns is None:
            if tensor.shape[1] < 4:
                raise ValueError("prices must contain at least four columns for open/high/low/close")
            extras = tuple(f"feature_{idx}" for idx in range(4, tensor.shape[1]))
            price_columns = ("open", "high", "low", "close", *extras)

        columns = tuple(str(col) for col in price_columns)
        required = ("open", "high", "low", "close")
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(f"price_columns is missing required OHLC fields: {missing}")

        ordered_indices = [columns.index(name) for name in required]
        ordered_indices.extend(idx for idx, name in enumerate(columns) if name not in required)
        reordered = tensor[:, ordered_indices].contiguous()
        resolved_columns = tuple(columns[idx] for idx in ordered_indices)
        return reordered, resolved_columns

    def _read_csv(self, path: Path):
        frame = pd.read_csv(path)
        frame = frame.rename(columns={col: str(col).lower() for col in frame.columns})
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.sort_values("date").set_index("date")
        elif "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            frame = frame.sort_values("timestamp").set_index("timestamp")
        frame = frame.dropna(axis=0, how="any")

        start = self.cfg.get("start_date")
        end = self.cfg.get("end_date")
        if (start is not None or end is not None) and isinstance(frame.index, pd.DatetimeIndex):
            if start is not None:
                frame = frame[frame.index >= pd.to_datetime(start)]
            if end is not None:
                frame = frame[frame.index <= pd.to_datetime(end)]
        return frame

    def _infer_price_columns(self, frame) -> tuple[str, ...]:
        ordered = ["open", "high", "low", "close"]
        missing = [col for col in ordered if col not in frame.columns]
        if missing:
            raise ValueError(f"CSV is missing required price columns: {missing}")
        extra_numeric = []
        for col in frame.columns:
            if col in ordered:
                continue
            if pd.api.types.is_numeric_dtype(frame[col]):
                extra_numeric.append(col)
        return tuple(ordered + extra_numeric)

    def _make_synth_prices(self, timesteps: int, mu: float, sigma: float) -> torch.Tensor:
        steps = torch.randn((timesteps,), generator=self._g) * float(sigma) + float(mu)
        log_prices = steps.cumsum(dim=0)
        base = torch.exp(log_prices)
        open_ = base
        close = base * torch.exp(torch.randn_like(base) * 0.001)
        high = torch.maximum(open_, close) * (1.0 + torch.rand_like(base) * 0.01)
        low = torch.minimum(open_, close) * (1.0 - torch.rand_like(base) * 0.01)
        return torch.stack([open_, high, low, close], dim=1).to(torch.float32)

    def _init_spaces(self) -> None:
        obs_dim = self.F + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(self.cfg["context_len"]), obs_dim),
            dtype=np.float32,
        )
        action_space = str(self.cfg.get("action_space", "continuous")).lower()
        if action_space == "continuous":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        elif action_space == "discrete":
            self.action_space = spaces.Discrete(3)
        else:
            raise ValueError("action_space must be 'continuous' or 'discrete'")

    def _max_start_index(self, min_episode_steps: int) -> int:
        horizon_guard = max(1, int(self.cfg["horizon"]))
        return self.T - horizon_guard - int(min_episode_steps)

    def _resolve_start_index(self, options: Optional[dict[str, Any]]) -> int:
        explicit_start = None
        if options is not None and "start_index" in options:
            explicit_start = int(options["start_index"])
        elif self.cfg.get("start_index") is not None:
            explicit_start = int(self.cfg["start_index"])
        if explicit_start is not None:
            return explicit_start

        if not bool(self.cfg.get("random_reset", False)):
            return int(self.cfg["context_len"])

        min_steps = int(self.cfg["episode_length"]) if self.cfg.get("episode_length") is not None else 1
        max_start = self._max_start_index(min_steps)
        min_start = int(self.cfg["context_len"])
        if max_start < min_start:
            raise ValueError(
                "Insufficient history to sample randomized episodes with the requested episode_length."
            )
        if max_start == min_start:
            return min_start
        return int(torch.randint(min_start, max_start + 1, size=(1,), generator=self._g).item())

    def _episode_end_for_start(self, start_index: int) -> int:
        max_end = self.T - max(1, int(self.cfg["horizon"]))
        if self.cfg.get("episode_length") is None:
            return max_end

        episode_length = int(self.cfg["episode_length"])
        if episode_length < 1:
            raise ValueError("episode_length must be at least 1 when provided")
        episode_end = start_index + episode_length
        if episode_end > max_end:
            raise ValueError(
                "episode_length exceeds available history for the selected start_index "
                f"(start_index={start_index}, max_episode_end={max_end})"
            )
        return episode_end

    def _reset_state(self, options: Optional[dict[str, Any]] = None) -> None:
        start_index = self._resolve_start_index(options)
        max_start = self._max_start_index(1)
        min_start = int(self.cfg["context_len"])
        if start_index < min_start or start_index > max_start:
            raise ValueError(
                "start_index must leave enough room for context and at least one step "
                f"(received {start_index}, valid range {min_start}..{max_start})"
            )
        self.start_index = int(start_index)
        self.cursor = int(start_index)
        self.episode_end = self._episode_end_for_start(self.cursor)
        self.position = 0.0
        self.equity = 1.0

    def _timestamp(self):
        if self.date_index is None:
            return None
        idx = min(self.cursor, len(self.date_index) - 1)
        return self.date_index[idx]

    def _prepare_action(self, action: Any) -> torch.Tensor:
        action_space = str(self.cfg.get("action_space", "continuous")).lower()
        if action_space == "continuous":
            tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.reshape(-1)
            if tensor.numel() != 1:
                raise ValueError("FastMarketEnv expects scalar continuous actions")
            return tensor

        int_action = int(action)
        if int_action not in {0, 1, 2}:
            raise ValueError("Discrete action out of bounds; expected 0, 1, or 2")
        mapping = {-1: -1.0, 0: 0.0, 1: 1.0}
        signed = mapping[int_action - 1]
        return torch.tensor([signed], dtype=torch.float32, device=self.device)

    def _augment_observation(self, obs: torch.Tensor) -> np.ndarray:
        window = obs.to(torch.float32)
        position_vec = torch.full((window.shape[0], 1), float(self.position), dtype=torch.float32, device=window.device)
        equity_vec = torch.full((window.shape[0], 1), float(self.equity), dtype=torch.float32, device=window.device)
        remaining = torch.full(
            (window.shape[0], 1),
            1.0 - (float(self.cursor) / float(self.T)),
            dtype=torch.float32,
            device=window.device,
        )
        out = torch.cat([window, position_vec, equity_vec, remaining], dim=1)
        return out.detach().cpu().numpy()


__all__ = ["FastMarketEnv"]
