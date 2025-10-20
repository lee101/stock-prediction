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
from src.alpaca_utils import (
    BASE_GROSS_EXPOSURE,
    INTRADAY_GROSS_EXPOSURE,
    MAX_GROSS_EXPOSURE,
    clamp_end_of_day_weights,
    leverage_penalty,
)
from src.leverage_settings import get_leverage_settings
from .config import PortfolioEnvConfig


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for logits -> simplex projection."""
    z = x - np.max(x)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-8)


def _sigmoid(value: float) -> float:
    """Logistic squashing."""
    return float(1.0 / (1.0 + np.exp(-float(value))))


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
    loss_shutdown_penalty: float = 0.0
    loss_shutdown_active_long: float = 0.0
    loss_shutdown_active_short: float = 0.0
    loss_shutdown_clipped: float = 0.0
    gross_exposure: float = 0.0
    post_close_gross_exposure: float = 0.0
    leverage_penalty: float = 0.0
    deleverage_turnover: float = 0.0

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
            "loss_shutdown_penalty": self.loss_shutdown_penalty,
            "loss_shutdown_active_long": self.loss_shutdown_active_long,
            "loss_shutdown_active_short": self.loss_shutdown_active_short,
            "loss_shutdown_clipped": self.loss_shutdown_clipped,
            "gross_exposure": self.gross_exposure,
            "post_close_gross_exposure": self.post_close_gross_exposure,
            "leverage_penalty": self.leverage_penalty,
            "deleverage_turnover": self.deleverage_turnover,
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

        leverage_settings = get_leverage_settings()
        self._use_leverage_head = bool(getattr(self.config, "leverage_head", False))
        base_gross = getattr(self.config, "base_gross_exposure", BASE_GROSS_EXPOSURE)
        max_gross_default = getattr(leverage_settings, "max_gross_leverage", MAX_GROSS_EXPOSURE)
        max_gross = getattr(self.config, "max_gross_leverage", max_gross_default)
        intraday_cap = getattr(self.config, "intraday_leverage_cap", INTRADAY_GROSS_EXPOSURE)

        self._base_gross = max(1.0, float(base_gross))
        self._max_gross = max(self._base_gross, float(max_gross))
        self._intraday_cap = max(self._max_gross, float(intraday_cap))
        self._enforce_eod_cap = bool(getattr(self.config, "enforce_end_of_day_cap", True))

        daily_rate = getattr(self.config, "daily_leverage_rate", None)
        if daily_rate is None:
            annual_rate = getattr(
                self.config,
                "leverage_penalty_annual_rate",
                getattr(leverage_settings, "annual_cost", None),
            )
            if annual_rate is None:
                annual_rate = leverage_settings.annual_cost if leverage_settings else 0.0675
            trading_days = getattr(
                self.config,
                "leverage_penalty_trading_days",
                getattr(leverage_settings, "trading_days_per_year", 252),
            )
            trading_days = max(1, int(trading_days))
            daily_rate = annual_rate / float(trading_days)
        self._daily_leverage_rate = float(daily_rate)

        if self.config.allow_short:
            action_dim = self.N * 2
            self.action_space = spaces.Box(low=-8.0, high=8.0, shape=(action_dim,), dtype=np.float32)
        else:
            action_len = self.N + (1 if self._use_leverage_head else 0)
            self.action_space = spaces.Box(low=-8.0, high=8.0, shape=(action_len,), dtype=np.float32)

        # Loss shutdown state ----------------------------------------------------
        self._loss_shutdown_penalty_term: float = 0.0
        self._loss_shutdown_clipped_amount: float = 0.0
        self._loss_shutdown_active_long: float = 0.0
        self._loss_shutdown_active_short: float = 0.0
        self._loss_shutdown_counters_long: Optional[np.ndarray] = None
        self._loss_shutdown_counters_short: Optional[np.ndarray] = None
        if self.config.loss_shutdown_enabled:
            self._ensure_loss_shutdown_buffers()

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
        self._loss_shutdown_penalty_term = 0.0
        self._loss_shutdown_clipped_amount = 0.0
        self._loss_shutdown_active_long = 0.0
        self._loss_shutdown_active_short = 0.0
        if self.config.loss_shutdown_enabled:
            self._ensure_loss_shutdown_buffers()
            self._loss_shutdown_counters_long.fill(0)
            if self._loss_shutdown_counters_short is not None:
                self._loss_shutdown_counters_short.fill(0)

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
            weights = self._apply_loss_shutdown(weights)
            return weights.astype(np.float32, copy=False)

        if self._use_leverage_head:
            if action.shape[0] != self.N + 1:
                raise ValueError(
                    f"Expected action of length {self.N + 1} when leverage head is enabled; received {action.shape[0]}"
                )
            logits = action[:-1]
            leverage_logit = float(action[-1])
        else:
            if action.shape[0] != self.N:
                raise ValueError(f"Expected action of length {self.N}; received {action.shape[0]}")
            logits = action
            leverage_logit = None

        weights = _softmax(logits)
        gross_target = self._base_gross
        if leverage_logit is not None:
            leverage_scale = _sigmoid(leverage_logit)
            gross_target = self._base_gross + leverage_scale * (self._intraday_cap - self._base_gross)
        gross_target = float(np.clip(gross_target, self._base_gross, self._intraday_cap))
        weights = weights * gross_target

        cap_setting = getattr(self.config, "weight_cap", None)
        if cap_setting is not None:
            cap_setting = float(cap_setting)
            per_asset_cap = cap_setting if cap_setting > 1.0 else cap_setting * gross_target
            weights = np.minimum(weights, per_asset_cap)
            total = weights.sum()
            if total <= 1e-6:
                weights = np.full(self.N, gross_target / self.N, dtype=np.float32)
            else:
                weights = weights * (gross_target / total)
            weights = np.minimum(weights, per_asset_cap)

        weights = self._apply_loss_shutdown(weights)
        return weights.astype(np.float32, copy=False)

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
        weights = self._apply_loss_shutdown(weights)
        return weights

    def _ensure_loss_shutdown_buffers(self) -> None:
        if not self.config.loss_shutdown_enabled:
            return
        if self._loss_shutdown_counters_long is None or self._loss_shutdown_counters_long.shape[0] != self.N:
            self._loss_shutdown_counters_long = np.zeros(self.N, dtype=np.int32)
        if self.config.allow_short:
            if self._loss_shutdown_counters_short is None or self._loss_shutdown_counters_short.shape[0] != self.N:
                self._loss_shutdown_counters_short = np.zeros(self.N, dtype=np.int32)
        else:
            self._loss_shutdown_counters_short = None

    def _apply_loss_shutdown(self, proposed: np.ndarray) -> np.ndarray:
        proposed = np.asarray(proposed, dtype=np.float32)
        if not self.config.loss_shutdown_enabled:
            self._loss_shutdown_penalty_term = 0.0
            self._loss_shutdown_clipped_amount = 0.0
            return proposed

        self._ensure_loss_shutdown_buffers()
        weights = proposed.copy()
        min_pos = float(self.config.loss_shutdown_min_position)
        probe = float(np.clip(self.config.loss_shutdown_probe_weight, 0.0, 1.0))

        active_long = self._loss_shutdown_counters_long > 0
        active_short = None
        if self.config.allow_short and self._loss_shutdown_counters_short is not None:
            active_short = self._loss_shutdown_counters_short > 0

        if np.any(active_long):
            positive_mask = weights > min_pos
            exceed_long = active_long & positive_mask & (weights > probe)
            if np.any(exceed_long):
                weights[exceed_long] = probe

        if active_short is not None and np.any(active_short):
            negative_mask = weights < -min_pos
            exceed_short = active_short & negative_mask & (np.abs(weights) > probe)
            if np.any(exceed_short):
                weights[exceed_short] = -probe if probe > 0.0 else 0.0

        diff = np.maximum(0.0, np.abs(proposed) - np.abs(weights))
        self._loss_shutdown_clipped_amount = float(diff.sum())

        penalty_scale = float(self.config.loss_shutdown_penalty)
        if penalty_scale > 0.0:
            penalty_contrib = 0.0
            if np.any(active_long):
                penalty_contrib += float(np.sum(np.abs(weights[active_long & (weights > min_pos)])))
            if active_short is not None and np.any(active_short):
                penalty_contrib += float(np.sum(np.abs(weights[active_short & (weights < -min_pos)])))
            self._loss_shutdown_penalty_term = penalty_scale * penalty_contrib
        else:
            self._loss_shutdown_penalty_term = 0.0

        return weights.astype(np.float32, copy=False)

    def _update_loss_shutdown_counters(self, executed_weights: np.ndarray, net_asset_returns: np.ndarray) -> None:
        if not self.config.loss_shutdown_enabled:
            self._loss_shutdown_active_long = 0.0
            self._loss_shutdown_active_short = 0.0
            return

        self._ensure_loss_shutdown_buffers()

        cooldown = int(max(1, self.config.loss_shutdown_cooldown))
        min_pos = float(self.config.loss_shutdown_min_position)
        tolerance = float(self.config.loss_shutdown_return_tolerance)

        long_counts = self._loss_shutdown_counters_long
        long_counts[:] = np.maximum(long_counts - 1, 0)
        positive_mask = executed_weights > min_pos
        loss_long = positive_mask & (net_asset_returns < -tolerance)
        profit_long = positive_mask & (net_asset_returns > tolerance)
        if np.any(loss_long):
            long_counts[loss_long] = cooldown
        if np.any(profit_long):
            long_counts[profit_long] = 0

        if self.config.allow_short and self._loss_shutdown_counters_short is not None:
            short_counts = self._loss_shutdown_counters_short
            short_counts[:] = np.maximum(short_counts - 1, 0)
            negative_mask = executed_weights < -min_pos
            loss_short = negative_mask & (net_asset_returns < -tolerance)
            profit_short = negative_mask & (net_asset_returns > tolerance)
            if np.any(loss_short):
                short_counts[loss_short] = cooldown
            if np.any(profit_short):
                short_counts[profit_short] = 0
            self._loss_shutdown_active_short = float(np.count_nonzero(short_counts))
        else:
            self._loss_shutdown_active_short = 0.0

        self._loss_shutdown_active_long = float(np.count_nonzero(long_counts))

    def _transition(self, new_weights: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        previous_weights = self._weights.copy()
        executed_weights = new_weights.astype(np.float32, copy=False)
        deltas = np.abs(executed_weights - previous_weights)
        turnover = float(deltas.sum())
        trading_cost_components = deltas * self.costs_vector

        realized_vector = self.realized_returns[self._index]
        asset_returns = executed_weights * realized_vector

        gross_exposure = float(np.sum(np.abs(executed_weights)))
        leverage_cost = leverage_penalty(
            gross_exposure,
            base_exposure=self._base_gross,
            daily_rate=self._daily_leverage_rate,
        )

        per_asset_cost = trading_cost_components.copy()
        trading_cost = float(per_asset_cost.sum())

        crypto_cost = float(per_asset_cost[self.crypto_mask].sum()) if np.any(self.crypto_mask) else 0.0
        non_crypto_cost = trading_cost - crypto_cost

        end_of_day_weights = executed_weights
        deleverage_turnover = 0.0
        extra_crypto_cost = 0.0
        if self._enforce_eod_cap:
            clamped, forced_turnover = clamp_end_of_day_weights(executed_weights, max_gross=self._max_gross)
            if forced_turnover > 1e-9:
                forced_deltas = np.abs(executed_weights - clamped)
                forced_cost = forced_deltas * self.costs_vector
                per_asset_cost += forced_cost
                turnover += float(forced_turnover)
                trading_cost += float(forced_cost.sum())
                extra_crypto_cost = float(forced_cost[self.crypto_mask].sum()) if np.any(self.crypto_mask) else 0.0
                end_of_day_weights = clamped
                deleverage_turnover = float(forced_turnover)
            else:
                end_of_day_weights = clamped

        crypto_cost += extra_crypto_cost
        non_crypto_cost = trading_cost - crypto_cost

        net_asset_returns_vector = asset_returns - per_asset_cost
        crypto_return = float(asset_returns[self.crypto_mask].sum()) if np.any(self.crypto_mask) else 0.0
        step_return = float(asset_returns.sum())
        non_crypto_return = step_return - crypto_return
        net_return = step_return - trading_cost - leverage_cost
        net_crypto_return = crypto_return - crypto_cost
        net_non_crypto_return = net_return - net_crypto_return
        weight_crypto = float(end_of_day_weights[self.crypto_mask].sum()) if np.any(self.crypto_mask) else 0.0
        weight_non_crypto = float(end_of_day_weights[~self.crypto_mask].sum()) if np.any(~self.crypto_mask) else 0.0
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
        reward -= self._loss_shutdown_penalty_term
        loss_shutdown_penalty = self._loss_shutdown_penalty_term

        self._last_weights = previous_weights
        self._weights = end_of_day_weights.astype(np.float32)
        self._index += 1
        self._step_count += 1

        self._update_loss_shutdown_counters(executed_weights, net_asset_returns_vector)
        loss_shutdown_active_long = self._loss_shutdown_active_long
        loss_shutdown_active_short = self._loss_shutdown_active_short
        loss_shutdown_clipped = self._loss_shutdown_clipped_amount

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
            loss_shutdown_penalty=loss_shutdown_penalty,
            loss_shutdown_active_long=loss_shutdown_active_long,
            loss_shutdown_active_short=loss_shutdown_active_short,
            loss_shutdown_clipped=loss_shutdown_clipped,
            gross_exposure=gross_exposure,
            post_close_gross_exposure=float(np.sum(np.abs(self._weights))),
            leverage_penalty=leverage_cost,
            deleverage_turnover=deleverage_turnover,
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
