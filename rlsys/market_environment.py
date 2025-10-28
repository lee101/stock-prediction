"""Gymnasium environment for realistic market simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import gymnasium as gym
import numpy as np

from .config import MarketConfig
from .utils import (
    EpisodeMetrics,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
)


@dataclass(slots=True)
class MarketState:
    index: int
    cash: float
    position: float
    portfolio_value: float
    turnover: float


class MarketEnvironment(gym.Env):
    """Event-driven market simulator."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        config: Optional[MarketConfig] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        if prices.ndim != 1:
            raise ValueError("prices must be a 1D array")
        if len(prices) != len(features):
            raise ValueError("prices and features must have matching lengths")
        if len(prices) < 2:
            raise ValueError("At least two price points are required")

        self.config = config or MarketConfig()
        self.prices = prices.astype(np.float64)
        self.features = features.astype(dtype)
        self.dtype = dtype

        feature_dim = self.features.shape[1]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim + 3,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-self.config.max_leverage,
            high=self.config.max_leverage,
            shape=(1,),
            dtype=np.float32,
        )

        self._state: Optional[MarketState] = None
        self._returns: list[float] = []
        self._equity_curve: list[float] = []
        self._last_action: float = 0.0
        self._peak_value = self.config.initial_capital

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        del options
        initial_value = self.config.initial_capital
        self._state = MarketState(
            index=0,
            cash=initial_value,
            position=0.0,
            portfolio_value=initial_value,
            turnover=0.0,
        )
        self._returns = []
        self._equity_curve = [initial_value]
        self._last_action = 0.0
        self._peak_value = initial_value
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping")
        action_value = float(np.clip(action[0], -self.config.max_leverage, self.config.max_leverage))
        state = self._state

        current_price = self.prices[state.index]
        next_index = state.index + 1
        next_price = self.prices[next_index]

        prev_value = state.portfolio_value
        prev_position = state.position
        target_position = action_value
        delta = float(
            np.clip(
                target_position - prev_position,
                -self.config.max_position_change,
                self.config.max_position_change,
            )
        )
        target_position = prev_position + delta

        price_return = (next_price - current_price) / current_price
        trade_penalty = abs(delta) * (
            self.config.transaction_cost
            + self.config.slippage
            + self.config.market_impact * abs(delta)
        )
        reward = target_position * price_return - trade_penalty
        reward -= self.config.risk_aversion * (target_position**2)

        new_portfolio_value = prev_value * (1.0 + reward)
        if not np.isfinite(new_portfolio_value):
            new_portfolio_value = self.config.min_cash
        new_portfolio_value = max(new_portfolio_value, 0.0)
        new_cash = new_portfolio_value * max(0.0, 1.0 - abs(target_position))
        new_turnover = state.turnover + abs(delta)

        state = MarketState(
            index=next_index,
            cash=new_cash,
            position=target_position,
            portfolio_value=new_portfolio_value,
            turnover=new_turnover,
        )
        self._state = state

        done = (
            next_index >= len(self.prices) - 1
            or new_cash < self.config.min_cash
            or new_portfolio_value < self.config.min_cash
        )
        truncated = False

        self._returns.append(reward)
        self._equity_curve.append(new_portfolio_value)
        self._peak_value = max(self._peak_value, new_portfolio_value)
        drawdown = 0.0
        if self._peak_value > 0:
            drawdown = (new_portfolio_value - self._peak_value) / self._peak_value
        drawdown_triggered = (
            self.config.max_drawdown_threshold is not None
            and drawdown <= -self.config.max_drawdown_threshold
        )
        self._last_action = action_value

        obs = self._get_observation()
        info = {
            "portfolio_value": new_portfolio_value,
            "position": state.position,
            "turnover": new_turnover,
            "drawdown": float(drawdown),
            "drawdown_triggered": drawdown_triggered,
        }
        if drawdown_triggered:
            done = True
        if done:
            metrics = self._finalize_metrics()
            info.update(metrics.as_dict())
        return obs, reward, done, truncated, info

    def render(self):
        if self._state is None:
            return {}
        return {
            "index": self._state.index,
            "portfolio_value": self._state.portfolio_value,
            "position": self._state.position,
            "cash": self._state.cash,
            "last_action": self._last_action,
        }

    def _get_observation(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Environment not initialized")
        idx = self._state.index
        obs_features = self.features[idx]
        augmented = np.concatenate(
            [
                obs_features,
                np.asarray(
                    [
                        self._state.position,
                        self._state.cash / self.config.initial_capital,
                        self._state.portfolio_value / self.config.initial_capital,
                    ],
                    dtype=self.dtype,
                ),
            ]
        )
        return augmented.astype(self.dtype, copy=False)

    def _finalize_metrics(self) -> EpisodeMetrics:
        sharpe = compute_sharpe_ratio(self._returns)
        max_dd = compute_max_drawdown(self._equity_curve)
        sortino = compute_sortino_ratio(self._returns)
        return EpisodeMetrics(
            reward=float(sum(self._returns)),
            length=len(self._returns),
            max_drawdown=float(max_dd),
            sharpe_ratio=float(sharpe),
            turnover=float(self._state.turnover if self._state else 0.0),
            sortino_ratio=float(sortino),
        )
