"""
Gymnasium environment for reinforcement-learning-based portfolio allocation.

The environment consumes feature cubes produced by ``gymrl.feature_pipeline`` and
emits observations suitable for Stable-Baselines3 style agents. It tracks trading
costs, turnover penalties, drawdown, and optional risk terms such as predicted CVaR
or forecast uncertainty derived from Toto/Kronos distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - Gymnasium is an explicit dependency
    raise ImportError(
        "gymnasium is required for gymrl.PortfolioEnv. "
        "Install it via `uv pip install gymnasium`."
    ) from exc

try:
    from loss_utils import CRYPTO_TRADING_FEE, TRADING_FEE
except ImportError:
    try:
        from stockagent.constants import CRYPTO_TRADING_FEE, TRADING_FEE
    except ImportError as exc:
        raise ImportError(
            "Trading fee constants not available. Ensure either loss_utils.py or "
            "stockagent.constants is importable before using gymrl.PortfolioEnv."
        ) from exc
from src.fixtures import crypto_symbols
from .config import PortfolioEnvConfig


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for logits -> simplex projection."""
    z = x - np.max(x)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-8)


@dataclass
class EnvStepInfo:
    """Diagnostic information emitted via ``info`` from ``PortfolioEnv.step``."""

    portfolio_value: float
    step_return: float
    net_return: float
    turnover: float
    trading_cost: float
    drawdown: float
    cvar_penalty: float
    uncertainty_penalty: float
    step_return_crypto: float = 0.0
    step_return_non_crypto: float = 0.0
    trading_cost_crypto: float = 0.0
    trading_cost_non_crypto: float = 0.0
    net_return_crypto: float = 0.0
    net_return_non_crypto: float = 0.0
    weight_crypto: float = 0.0
    weight_non_crypto: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "portfolio_value": self.portfolio_value,
            "step_return": self.step_return,
            "net_return": self.net_return,
            "turnover": self.turnover,
            "trading_cost": self.trading_cost,
            "drawdown": self.drawdown,
            "cvar_penalty": self.cvar_penalty,
            "uncertainty_penalty": self.uncertainty_penalty,
            "step_return_crypto": self.step_return_crypto,
            "step_return_non_crypto": self.step_return_non_crypto,
            "trading_cost_crypto": self.trading_cost_crypto,
            "trading_cost_non_crypto": self.trading_cost_non_crypto,
            "net_return_crypto": self.net_return_crypto,
            "net_return_non_crypto": self.net_return_non_crypto,
            "weight_crypto": self.weight_crypto,
            "weight_non_crypto": self.weight_non_crypto,
        }


class PortfolioEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    Portfolio allocation environment that outputs log-return-based rewards.

    The agent observes concatenated per-asset features (plus optional position
    state) and emits allocation logits. For long-only configurations the logits
    are transformed via softmax, enforcing a simplex constraint. If shorting is
    enabled ``allow_short=True`` the environment expects separate long/short
    logits per asset and enforces a configurable leverage cap.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        realized_returns: np.ndarray,
        config: Optional[PortfolioEnvConfig] = None,
        *,
        feature_names: Optional[Sequence[str]] = None,
        symbols: Optional[Sequence[str]] = None,
        timestamps: Optional[Sequence[Any]] = None,
        forecast_cvar: Optional[np.ndarray] = None,
        forecast_uncertainty: Optional[np.ndarray] = None,
        append_portfolio_state: bool = True,
        start_index: int = 0,
        episode_length: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__()
        self.config = config or PortfolioEnvConfig()

        if self.config.allow_short and self.config.include_cash:
            raise ValueError(
                "allow_short=True is currently incompatible with include_cash=True. "
                "Disable cash or extend the projection logic before enabling both."
            )

        self.features = np.asarray(features, dtype=np.float32)
        self.realized_returns = np.asarray(realized_returns, dtype=np.float32)

        if self.features.ndim != 3:
            raise ValueError(f"features must be 3-D (T, N, F); received shape {self.features.shape}")
        if self.realized_returns.shape[:2] != self.features.shape[:2]:
            raise ValueError(
                "realized_returns must align with features on the first two axes "
                f"(expected {self.features.shape[:2]}, got {self.realized_returns.shape[:2]})"
            )

        self.feature_names = list(feature_names) if feature_names is not None else None
        self.symbols = list(symbols) if symbols is not None else None
        self.timestamps = list(timestamps) if timestamps is not None else None
        if self.symbols is None:
            _, num_assets, _ = self.features.shape
            self.symbols = [f"asset_{idx}" for idx in range(num_assets)]

        self.append_portfolio_state = append_portfolio_state
        self.rng = rng or np.random.default_rng()

        self.forecast_cvar = None
        if forecast_cvar is not None:
            forecast_cvar = np.asarray(forecast_cvar, dtype=np.float32)
            if forecast_cvar.shape != self.realized_returns.shape:
                raise ValueError(
                    "forecast_cvar must match realized_returns shape; "
                    f"expected {self.realized_returns.shape}, got {forecast_cvar.shape}"
                )
            self.forecast_cvar = forecast_cvar

        self.forecast_uncertainty = None
        if forecast_uncertainty is not None:
            forecast_uncertainty = np.asarray(forecast_uncertainty, dtype=np.float32)
            if forecast_uncertainty.shape != self.realized_returns.shape:
                raise ValueError(
                    "forecast_uncertainty must match realized_returns shape; "
                    f"expected {self.realized_returns.shape}, got {forecast_uncertainty.shape}"
                )
            self.forecast_uncertainty = forecast_uncertainty

        if self.config.include_cash:
            T, _, F = self.features.shape
            cash_features = np.zeros((T, 1, F), dtype=np.float32)
            self.features = np.concatenate([self.features, cash_features], axis=1)

            cash_returns = np.full((self.realized_returns.shape[0], 1), self.config.cash_return, dtype=np.float32)
            self.realized_returns = np.concatenate([self.realized_returns, cash_returns], axis=1)

            if self.forecast_cvar is not None:
                cash_cvar = np.zeros((self.forecast_cvar.shape[0], 1), dtype=np.float32)
                self.forecast_cvar = np.concatenate([self.forecast_cvar, cash_cvar], axis=1)

            if self.forecast_uncertainty is not None:
                cash_uncertainty = np.zeros((self.forecast_uncertainty.shape[0], 1), dtype=np.float32)
                self.forecast_uncertainty = np.concatenate([self.forecast_uncertainty, cash_uncertainty], axis=1)

            if self.feature_names is not None:
                self.feature_names = list(self.feature_names)

            self.symbols = list(self.symbols) + ["CASH"]

        crypto_set = {symbol.upper() for symbol in crypto_symbols}
        self.crypto_mask = np.array([symbol.upper() in crypto_set for symbol in self.symbols], dtype=bool)

        self.T, self.N, self.F = self.features.shape

        base_costs = np.full(self.N, TRADING_FEE, dtype=np.float32)
        for idx, symbol in enumerate(self.symbols):
            if symbol.upper() in crypto_set:
                base_costs[idx] = CRYPTO_TRADING_FEE
        if self.config.include_cash:
            base_costs[-1] = 0.0

        if self.config.costs_bps:
            bps_cost = self.config.costs_bps / 1e4
            if self.config.include_cash:
                base_costs[:-1] = base_costs[:-1] + bps_cost
            else:
                base_costs = base_costs + bps_cost

        if self.config.per_asset_costs_bps is not None:
            per_asset = np.asarray(self.config.per_asset_costs_bps, dtype=np.float32) / 1e4
            if per_asset.shape[0] != self.N:
                raise ValueError(
                    f"per_asset_costs_bps expected length {self.N}, received {per_asset.shape[0]}"
                )
            base_costs = per_asset

        self.costs_vector = base_costs
        self.start_index = start_index
        self.episode_length = episode_length or (self.T - start_index - 1)
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive given provided data slice.")
        self._max_index = min(self.T - 1, self.start_index + self.episode_length)

        # Allocate buffers
        self._weights = np.zeros(self.N, dtype=np.float32)
        self._last_weights = np.zeros(self.N, dtype=np.float32)
        self._portfolio_value = 1.0
        self._peak_value = 1.0
        self._step_count = 0
        self._index = self.start_index

        # Observation space: flatten per-asset features and optionally append position state.
        obs_dim = self.N * self.F
        if self.append_portfolio_state:
            obs_dim += self.N + 1  # current weights + portfolio multiplier
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        if self.config.allow_short:
            action_dim = self.N * 2
            self.action_space = spaces.Box(low=-8.0, high=8.0, shape=(action_dim,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-8.0, high=8.0, shape=(self.N,), dtype=np.float32)

    # Gymnasium API -----------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}

        # Optionally randomise starting index within permissible range.
        random_start = options.get("random_start", False)
        if random_start:
            max_start = self.T - self.episode_length - 1
            self.start_index = int(self.rng.integers(0, max(1, max_start + 1)))
            self._max_index = min(self.T - 1, self.start_index + self.episode_length)
        else:
            self.start_index = int(options.get("start_index", self.start_index))
            self._max_index = min(self.T - 1, self.start_index + self.episode_length)

        self._weights.fill(0.0)
        self._last_weights.fill(0.0)
        self._portfolio_value = 1.0
        self._peak_value = 1.0
        self._step_count = 0
        self._index = self.start_index

        observation = self._get_observation()
        return observation, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        action = np.asarray(action, dtype=np.float32)
        new_weights = self._project_weights(action)
        return self._transition(new_weights)

    def step_with_weights(self, weights: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        """
        Step the environment using explicit portfolio weights instead of logits.

        This helper is primarily intended for offline dataset creation where an
        existing policy (e.g., heuristic) already outputs simplex allocations.
        """

        if self.config.allow_short:
            raise NotImplementedError("step_with_weights is only supported for long-only configurations.")

        new_weights = self._normalise_long_only(weights)
        return self._transition(new_weights)

    # Internal helpers --------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        obs_components = [self.features[self._index].reshape(-1)]
        if self.append_portfolio_state:
            obs_components.append(self._weights.astype(np.float32))
            obs_components.append(np.array([self._portfolio_value], dtype=np.float32))
        return np.concatenate(obs_components, axis=0).astype(np.float32, copy=False)

    def _project_weights(self, action: np.ndarray) -> np.ndarray:
        if self.config.allow_short:
            if action.shape[0] != 2 * self.N:
                raise ValueError(
                    f"Short-enabled action must have length {2 * self.N}; received {action.shape[0]}"
                )
            half = self.N
            long_logits = action[:half]
            short_logits = action[half:]
            long_weights = _softmax(long_logits)
            short_weights = _softmax(short_logits)
            weights = self.config.leverage_cap * (long_weights - short_weights)
            gross = np.sum(np.abs(weights))
            if gross > self.config.leverage_cap:
                weights *= self.config.leverage_cap / max(gross, 1e-6)
            return weights.astype(np.float32)

        weights = _softmax(action)
        if self.config.weight_cap is not None:
            capped = np.minimum(weights, self.config.weight_cap)
            capped_sum = capped.sum()
            if capped_sum <= 1e-6:
                weights = np.full_like(weights, 1.0 / weights.size)
            else:
                weights = capped / capped_sum
        return weights.astype(np.float32)

    def _normalise_long_only(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != (self.N,):
            if self.config.include_cash and weights.shape == (self.N - 1,):
                cash_buffer = max(0.0, 1.0 - float(np.sum(np.maximum(weights, 0.0))))
                weights = np.concatenate([weights, np.array([cash_buffer], dtype=np.float32)], axis=0)
            else:
                raise ValueError(f"Explicit weights must have shape {(self.N,)}, received {weights.shape}")
        weights = np.maximum(weights, 0.0)
        total = weights.sum()
        if total <= 1e-6:
            weights = np.full(self.N, 1.0 / self.N, dtype=np.float32)
        else:
            weights = weights / total
        if self.config.weight_cap is not None:
            weights = np.minimum(weights, self.config.weight_cap)
            normaliser = weights.sum()
            if normaliser <= 1e-6:
                weights = np.full(self.N, 1.0 / self.N, dtype=np.float32)
            else:
                weights = weights / normaliser
        return weights

    def _transition(self, new_weights: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        deltas = np.abs(new_weights - self._weights)
        turnover = float(deltas.sum())
        trading_cost = float(np.dot(deltas, self.costs_vector))

        crypto_cost = float(np.dot(deltas[self.crypto_mask], self.costs_vector[self.crypto_mask])) if np.any(self.crypto_mask) else 0.0
        non_crypto_cost = trading_cost - crypto_cost

        realized_vector = self.realized_returns[self._index]
        asset_returns = new_weights * realized_vector
        crypto_return = float(asset_returns[self.crypto_mask].sum()) if np.any(self.crypto_mask) else 0.0
        step_return = float(asset_returns.sum())
        non_crypto_return = step_return - crypto_return
        net_return = step_return - trading_cost
        net_crypto_return = crypto_return - crypto_cost
        net_non_crypto_return = net_return - net_crypto_return
        weight_crypto = float(new_weights[self.crypto_mask].sum()) if np.any(self.crypto_mask) else 0.0
        weight_non_crypto = float(new_weights[~self.crypto_mask].sum()) if np.any(~self.crypto_mask) else 0.0
        net_multiplier = max(1e-8, 1.0 + net_return)

        self._portfolio_value *= net_multiplier
        self._peak_value = max(self._peak_value, self._portfolio_value)
        drawdown = 0.0
        if self._peak_value > 1e-8:
            drawdown = (self._peak_value - self._portfolio_value) / self._peak_value

        cvar_penalty = 0.0
        if self.forecast_cvar is not None and self.config.cvar_penalty:
            cvar_contrib = np.abs(new_weights) * np.maximum(self.forecast_cvar[self._index], 0.0)
            cvar_penalty = self.config.cvar_penalty * float(cvar_contrib.sum())

        uncertainty_penalty = 0.0
        if self.forecast_uncertainty is not None and self.config.uncertainty_penalty:
            uncertainty_contrib = np.abs(new_weights) * np.maximum(self.forecast_uncertainty[self._index], 0.0)
            uncertainty_penalty = self.config.uncertainty_penalty * float(uncertainty_contrib.sum())

        reward = np.log(net_multiplier) - self.config.turnover_penalty * turnover
        reward -= self.config.drawdown_penalty * drawdown
        reward -= cvar_penalty
        reward -= uncertainty_penalty

        self._last_weights = self._weights.copy()
        self._weights = new_weights.astype(np.float32)
        self._index += 1
        self._step_count += 1

        terminated = self._index >= self._max_index
        truncated = False

        info = EnvStepInfo(
            portfolio_value=self._portfolio_value,
            step_return=step_return,
            net_return=net_return,
            turnover=turnover,
            trading_cost=trading_cost,
            drawdown=drawdown,
            cvar_penalty=cvar_penalty,
            uncertainty_penalty=uncertainty_penalty,
            step_return_crypto=crypto_return,
            step_return_non_crypto=non_crypto_return,
            trading_cost_crypto=crypto_cost,
            trading_cost_non_crypto=non_crypto_cost,
            net_return_crypto=net_crypto_return,
            net_return_non_crypto=net_non_crypto_return,
            weight_crypto=weight_crypto,
            weight_non_crypto=weight_non_crypto,
        ).to_dict()

        observation = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return observation, float(reward), bool(terminated), bool(truncated), info

    # Convenience accessors ---------------------------------------------------------
    @property
    def portfolio_value(self) -> float:
        return float(self._portfolio_value)

    @property
    def current_weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def last_weights(self) -> np.ndarray:
        return self._last_weights.copy()
