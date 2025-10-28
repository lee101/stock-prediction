"""Torch-first market environment compatible with PufferLib."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

try:  # Optional PufferLib registration utilities.
    import pufferlib.ocean as ocean
except Exception:  # pragma: no cover - pufferlib not installed in some envs
    ocean = None

try:  # Optional CSV ingestion.
    import pandas as pd
except Exception:  # pragma: no cover - pandas optional for synthetic runs
    pd = None


@dataclass
class MarketEnvConfig:
    """Configuration options for :class:`MarketEnv`."""

    # Observation / episode
    context_len: int = 128
    horizon: int = 1
    mode: str = "open_close"  # 'open_close' | 'event' | 'maxdiff'

    # Data
    data_root: Optional[str] = None
    symbol: Optional[str] = None
    price_key: str = "close"
    normalize_returns: bool = True

    # Fees & slippage
    trading_fee: float = 0.0005
    crypto_trading_fee: float = 0.0015
    slip_bps: float = 1.5

    # Leverage & financing (stocks only)
    annual_leverage_rate: float = 0.0675
    intraday_leverage_max: float = 4.0
    overnight_leverage_max: float = 2.0

    # Inventory / regularisation
    inv_penalty: float = 0.0
    action_space: str = "continuous"  # 'continuous' or 'discrete'
    reward_scale: float = 1.0

    # Asset switches
    is_crypto: bool = False

    # RNG / device
    seed: int = 1337
    device: str = "cuda"

    # Synthetic fallback
    synth_T: int = 200_000
    synth_mu: float = 0.0
    synth_sigma: float = 0.02

    # MaxDiff specific tuning
    maxdiff_limit_scale: float = 0.05  # +/-5% limits around open
    maxdiff_deadband: float = 0.05     # no trade if |direction| below threshold


class MarketEnv(gym.Env):
    """High-throughput, differentiable trading environment.

    * Internal state uses :mod:`torch`, enabling differentiable rollouts.
    * Observations are returned as NumPy views detached from torch tensors.
    * Supports multiple execution modes (``open_close``, ``event``, ``maxdiff``).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: Optional[torch.Tensor] = None,   # (T, F) torch float32 [open,high,low,close,volume,...]
        exog: Optional[torch.Tensor] = None,     # (T, E) torch float32 optional exogenous
        price_columns: Optional[Tuple[str, ...]] = None,
        cfg: Optional[MarketEnvConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or MarketEnvConfig()
        device_name = self.cfg.device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self._g = torch.Generator(device="cpu").manual_seed(int(self.cfg.seed))

        prices, exog, date_index, resolved_columns = self._resolve_data(prices, exog, price_columns)
        self.price_columns = resolved_columns
        self.date_index = date_index
        self.prices = prices.to(self.device).contiguous()
        self.exog = exog.to(self.device).contiguous() if exog is not None else None
        self.T, self.F = self.prices.shape
        self.F_total = self.F + (0 if self.exog is None else self.exog.shape[1])
        if self.T <= self.cfg.context_len + max(1, self.cfg.horizon):
            raise ValueError("Insufficient history to satisfy context and horizon requirements.")

        # Pre-compute scalars on device for differentiable math.
        self._fee_rate = torch.tensor(
            self.cfg.crypto_trading_fee if self.cfg.is_crypto else self.cfg.trading_fee,
            dtype=torch.float32,
            device=self.device,
        )
        slip_rate = self.cfg.slip_bps / 10_000.0
        self._slip_rate = torch.tensor(float(slip_rate), dtype=torch.float32, device=self.device)
        self._daily_leverage_rate = torch.tensor(
            0.0 if self.cfg.is_crypto else self.cfg.annual_leverage_rate / 252.0,
            dtype=torch.float32,
            device=self.device,
        )
        intraday_cap = 1.0 if self.cfg.is_crypto else max(1.0, float(self.cfg.intraday_leverage_max))
        overnight_cap = 1.0 if self.cfg.is_crypto else max(1.0, float(self.cfg.overnight_leverage_max))
        overnight_cap = min(overnight_cap, intraday_cap)
        self._intraday_cap = torch.tensor(intraday_cap, dtype=torch.float32, device=self.device)
        self._overnight_cap = torch.tensor(overnight_cap, dtype=torch.float32, device=self.device)
        self._inv_penalty = torch.tensor(float(self.cfg.inv_penalty), dtype=torch.float32, device=self.device)
        self._limit_scale = torch.tensor(float(self.cfg.maxdiff_limit_scale), dtype=torch.float32, device=self.device)
        self._maxdiff_deadband = torch.tensor(float(self.cfg.maxdiff_deadband), dtype=torch.float32, device=self.device)

        self._build_feature_tensor()
        self._init_spaces()
        self._reset_state()

    # ------------------------------------------------------------------ #
    # Environment API                                                    #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        if seed is not None:
            self._g.manual_seed(int(seed))
        self._reset_state()
        return self._get_observation(), {}

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        action_tensor = self._prepare_action(action)
        mode = self.cfg.mode.lower()
        if mode == "maxdiff":
            reward_tensor, info = self._step_maxdiff(action_tensor)
        elif mode == "event":
            reward_tensor, info = self._step_open_close(action_tensor, event_mode=True)
        else:
            reward_tensor, info = self._step_open_close(action_tensor, event_mode=False)

        reward_scaled = reward_tensor * self.cfg.reward_scale
        self.cursor += 1
        terminated = self.cursor >= self.episode_end
        terminated = bool(terminated or (self.equity <= 0.0))
        truncated = False
        observation = self._get_observation()
        reward = float(reward_scaled.detach().cpu().item())
        info["equity"] = float(self.equity.detach().cpu().item())
        info["position"] = float(self.position.detach().cpu().item())
        info["timestamp"] = self._timestamp()
        return observation, reward, terminated, truncated, info

    def render(self) -> None:  # pragma: no cover - rendering not implemented
        return None

    def close(self) -> None:  # pragma: no cover - no external resources
        return None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _resolve_data(
        self,
        prices: Optional[torch.Tensor],
        exog: Optional[torch.Tensor],
        price_columns: Optional[Tuple[str, ...]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any], Tuple[str, ...]]:
        if prices is not None:
            if price_columns is None:
                price_columns = tuple(f"col_{i}" for i in range(prices.size(1)))
            return prices.float(), exog.float() if exog is not None else None, None, tuple(price_columns)
        if pd is None:
            synth = self._make_synth_prices(self.cfg.synth_T, self.cfg.synth_mu, self.cfg.synth_sigma)
            return synth, None, None, ("open", "high", "low", "close")
        root = Path(self.cfg.data_root or "trainingdata").expanduser().resolve()
        candidate_paths = []
        if self.cfg.symbol:
            symbol = str(self.cfg.symbol).upper()
            candidate_paths.extend(sorted(root.glob(f"**/{symbol}.csv")))
            candidate_paths.extend(sorted(root.glob(f"**/{symbol}_*.csv")))
        if not candidate_paths:
            candidate_paths = sorted(root.glob("**/*.csv"))
        if not candidate_paths:
            synth = self._make_synth_prices(self.cfg.synth_T, self.cfg.synth_mu, self.cfg.synth_sigma)
            return synth, None, None, ("open", "high", "low", "close")
        frame = self._read_csv(candidate_paths[0])
        price_cols = self._infer_price_columns(frame)
        price_values = frame[list(price_cols)].to_numpy(dtype="float32")
        prices_tensor = torch.from_numpy(price_values)
        exog_cols = [col for col in frame.columns if col not in price_cols]
        exog_tensor = None
        if exog_cols:
            exog_values = frame[exog_cols].to_numpy(dtype="float32")
            exog_tensor = torch.from_numpy(exog_values)
        return prices_tensor, exog_tensor, frame.index.copy(), tuple(price_cols)

    def _read_csv(self, path: Path) -> "pd.DataFrame":  # type: ignore[override]
        frame = pd.read_csv(path)
        lowered = {c: str(c).lower() for c in frame.columns}
        frame = frame.rename(columns=lowered)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.sort_values("date")
            frame = frame.set_index("date")
        frame = frame.dropna(axis=0, how="any")
        return frame

    def _infer_price_columns(self, frame: "pd.DataFrame") -> Tuple[str, ...]:  # type: ignore[override]
        required = ["open", "high", "low", "close"]
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"CSV is missing required price columns: {missing}")
        ordered = ["open", "high", "low", "close"]
        extra = [col for col in frame.columns if col not in ordered]
        return tuple(list(ordered) + extra[: max(0, len(frame.columns) - len(ordered))])

    def _make_synth_prices(self, T: int, mu: float, sigma: float) -> torch.Tensor:
        steps = torch.randn((T,), generator=self._g) * float(sigma) + float(mu)
        log_prices = steps.cumsum(dim=0)
        base = torch.exp(log_prices)
        open_ = base
        close = base * torch.exp(torch.randn_like(base) * 0.001)
        high = torch.maximum(open_, close) * (1.0 + torch.rand_like(base) * 0.01)
        low = torch.minimum(open_, close) * (1.0 - torch.rand_like(base) * 0.01)
        stacked = torch.stack([open_, high, low, close], dim=1)
        return stacked.to(torch.float32)

    def _build_feature_tensor(self) -> None:
        if self.exog is not None:
            self.features = torch.cat([self.prices, self.exog], dim=1)
        else:
            self.features = self.prices
        self.features = self.features.to(self.device)
        self.price_lookup = {name: idx for idx, name in enumerate(self.price_columns)}
        for key in ("open", "high", "low", "close"):
            if key not in self.price_lookup:
                raise ValueError(f"Missing OHLC column '{key}' in price tensors")

    def _init_spaces(self) -> None:
        mode = self.cfg.mode.lower()
        if mode not in {"open_close", "event", "maxdiff"}:
            raise ValueError(f"Unsupported mode '{self.cfg.mode}'")
        obs_dim = self.F_total + 3
        low = -np.inf
        high = np.inf
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(self.cfg.context_len, obs_dim),
            dtype=np.float32,
        )
        if mode == "maxdiff" and self.cfg.action_space == "discrete":
            raise ValueError("MaxDiff mode requires continuous actions for limit parameterisation")
        if self.cfg.action_space == "continuous":
            action_dim = 2 if mode == "maxdiff" else 1
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        elif self.cfg.action_space == "discrete":
            self.action_space = spaces.Discrete(3)
        else:
            raise ValueError("action_space must be 'continuous' or 'discrete'")

    def _reset_state(self) -> None:
        self.equity = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.position = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.cursor = self.cfg.context_len
        horizon_guard = max(1, self.cfg.horizon)
        self.episode_end = self.T - horizon_guard
        self.last_reward = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.last_trade_cost = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.last_financing_cost = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.last_gross = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def _timestamp(self) -> Optional[Any]:
        if self.date_index is None:
            return None
        idx = min(self.cursor, len(self.date_index) - 1)
        return self.date_index[idx]

    def _prepare_action(self, action: Any) -> torch.Tensor:
        if self.cfg.action_space == "continuous":
            tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            return tensor
        # Discrete case maps {-1, 0, 1} exposures scaled to intraday cap.
        int_action = int(action)
        if int_action not in {0, 1, 2}:
            raise ValueError("Discrete action out of bounds; expected 0,1,2")
        mapping = {-1: -1.0, 0: 0.0, 1: 1.0}
        signed = mapping[int_action - 1]
        return torch.tensor([signed], dtype=torch.float32, device=self.device)

    def _get_observation(self) -> Any:
        start = self.cursor - self.cfg.context_len
        end = self.cursor
        window = self.features[start:end]
        position_vec = torch.full((self.cfg.context_len, 1), self.position, dtype=torch.float32, device=self.device)
        equity_vec = torch.full((self.cfg.context_len, 1), self.equity, dtype=torch.float32, device=self.device)
        remaining = torch.full(
            (self.cfg.context_len, 1),
            torch.tensor(1.0 - (self.cursor / self.T), dtype=torch.float32, device=self.device),
            dtype=torch.float32,
            device=self.device,
        )
        obs = torch.cat([window, position_vec, equity_vec, remaining], dim=1)
        return obs.detach().cpu().numpy()

    def _step_open_close(self, action_tensor: torch.Tensor, *, event_mode: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.cfg.action_space == "continuous":
            raw = torch.tanh(action_tensor[..., 0])
        else:
            raw = torch.clamp(action_tensor[..., 0], -1.0, 1.0)
        target = raw * self._intraday_cap
        gross_intraday = target.abs()
        delta = target - self.position
        equity = self.equity
        fee = (delta.abs() * self._fee_rate * equity)
        slip = (delta.abs() * self._slip_rate * equity)
        financing = torch.clamp(gross_intraday - 1.0, min=0.0) * equity * self._daily_leverage_rate
        open_price = self._price(self.cursor, "open")
        close_idx = self.cursor + max(1, self.cfg.horizon) - 1
        close_price = self._price(close_idx, "close")
        price_return = (close_price - open_price) / torch.clamp(open_price, min=1e-6)
        gross = equity * target * price_return
        overnight = torch.clamp(target, -self._overnight_cap, self._overnight_cap)
        deleverage_delta = target - overnight
        if torch.any(torch.abs(deleverage_delta) > 1e-6):
            fee = fee + deleverage_delta.abs() * self._fee_rate * equity
            slip = slip + deleverage_delta.abs() * self._slip_rate * equity
        penalty = self._inv_penalty * (overnight ** 2)
        net = gross - fee - slip - financing - penalty
        self.equity = equity + net
        self.position = overnight.detach()
        self.last_reward = net
        self.last_trade_cost = fee + slip
        self.last_financing_cost = financing
        self.last_gross = gross
        info = {
            "gross_pnl": float(gross.detach().cpu().item()),
            "trading_cost": float((fee + slip).detach().cpu().item()),
            "financing_cost": float(financing.detach().cpu().item()),
            "intraday_position": float(target.detach().cpu().item()),
            "overnight_position": float(overnight.detach().cpu().item()),
            "price_return": float(price_return.detach().cpu().item()),
            "deleverage_notional": float(deleverage_delta.abs().detach().cpu().item()),
            "mode": "event" if event_mode else "open_close",
        }
        return net, info

    def _step_maxdiff(self, action_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        direction_raw = torch.tanh(action_tensor[..., 0])
        limit_raw = torch.tanh(action_tensor[..., 1])
        magnitude = direction_raw.abs()
        active = magnitude >= self._maxdiff_deadband
        equity = self.equity
        zero = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        open_price = self._price(self.cursor, "open")
        high_price = self._price(self.cursor, "high")
        low_price = self._price(self.cursor, "low")
        close_price = self._price(self.cursor, "close")

        if not bool(active):
            self.position = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            self.last_reward = zero
            self.last_trade_cost = zero
            self.last_financing_cost = zero
            self.last_gross = zero
            return zero, {
                "gross_pnl": 0.0,
                "trading_cost": 0.0,
                "financing_cost": 0.0,
                "maxdiff_filled": False,
                "mode": "maxdiff",
            }

        direction = torch.sign(direction_raw)
        size = torch.clamp(magnitude, 0.0, self._overnight_cap)
        limit_pct = limit_raw.abs() * self._limit_scale
        if bool(direction > 0):
            limit_price = open_price * (1.0 + limit_pct)
            touched = bool(high_price >= limit_price and low_price <= limit_price)
            signed = size
        else:
            limit_price = open_price * (1.0 - limit_pct)
            touched = bool(low_price <= limit_price and high_price >= limit_price)
            signed = -size
        if not touched:
            self.position = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            self.last_reward = zero
            self.last_trade_cost = zero
            self.last_financing_cost = zero
            self.last_gross = zero
            return zero, {
                "gross_pnl": 0.0,
                "trading_cost": 0.0,
                "financing_cost": 0.0,
                "maxdiff_filled": False,
                "mode": "maxdiff",
            }

        # Entry executed at limit, exit at the close (market order)
        price_return = (close_price - limit_price) / torch.clamp(limit_price, min=1e-6)
        gross = equity * signed * price_return
        fee = size * equity * self._fee_rate * 2.0
        slip = size * equity * self._slip_rate * 2.0
        net = gross - fee - slip
        self.equity = equity + net
        self.position = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.last_reward = net
        self.last_trade_cost = fee + slip
        self.last_financing_cost = zero
        self.last_gross = gross
        info = {
            "gross_pnl": float(gross.detach().cpu().item()),
            "trading_cost": float((fee + slip).detach().cpu().item()),
            "financing_cost": 0.0,
            "maxdiff_filled": True,
            "limit_price": float(limit_price.detach().cpu().item()),
            "mode": "maxdiff",
        }
        return net, info

    def _price(self, idx: int, name: str) -> torch.Tensor:
        pos = self.price_lookup[name]
        return self.prices[idx, pos]


if ocean is not None:  # pragma: no cover - optional registration
    try:
        ocean.market_env = MarketEnv  # type: ignore[attr-defined]
    except Exception:
        pass


__all__ = ["MarketEnv", "MarketEnvConfig"]
