from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd


class KronosDMEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-asset continuous-position environment backed by precomputed features."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.Series,
        features: pd.DataFrame,
        returns_window: int = 0,
        transaction_cost_bps: float = 1.0,
        slippage_bps: float = 0.5,
        max_position: float = 1.0,
        hold_penalty: float = 0.0,
        reward: str = "pnl",
    ) -> None:
        super().__init__()
        self.prices = prices.astype(float)
        self.features = features.astype(np.float32)
        self.transaction_cost = transaction_cost_bps / 1e4
        self.slippage = slippage_bps / 1e4
        self.max_position = max_position
        self.hold_penalty = hold_penalty
        if reward not in {"pnl", "log_return"}:
            raise ValueError("reward must be 'pnl' or 'log_return'")
        self.reward_mode = reward
        self.returns = self.prices.pct_change().fillna(0.0).to_numpy()
        self._reset_state()

        obs_shape = (self.features.shape[1],)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _reset_state(self) -> None:
        self._t = 0
        self._pos = 0.0
        self._nav = 1.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        self._reset_state()
        return self.features.iloc[self._t].to_numpy(dtype=np.float32), {}

    def step(self, action: np.ndarray):  # type: ignore[override]
        action = float(np.clip(action[0], -1.0, 1.0)) * self.max_position
        turnover = abs(action - self._pos)
        cost = turnover * (self.transaction_cost + self.slippage)

        if self._t + 1 >= len(self.prices):
            return self.features.iloc[self._t].to_numpy(dtype=np.float32), 0.0, True, False, {
                "nav": self._nav,
                "pos": self._pos,
                "ret": 0.0,
            }

        ret = float(self.returns[self._t + 1])
        pnl = action * ret - cost - self.hold_penalty * (action**2)
        if self.reward_mode == "log_return":
            reward = float(np.log1p(pnl))
        else:
            reward = pnl

        self._pos = action
        self._t += 1
        self._nav *= (1.0 + pnl)

        obs = self.features.iloc[self._t].to_numpy(dtype=np.float32)
        terminated = self._t >= len(self.prices) - 1
        info = {"nav": self._nav, "pos": self._pos, "ret": ret}
        return obs, float(reward), bool(terminated), False, info
