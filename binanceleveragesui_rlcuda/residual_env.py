from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

HOURLY_PERIODS_PER_YEAR = 8760.0


@dataclass(frozen=True)
class ResidualLeverageEnvConfig:
    window_size: int = 32
    max_leverage: float = 5.0
    maker_fee: float = 0.001
    margin_hourly_rate: float = 0.0000025457
    initial_cash: float = 10_000.0
    min_equity_frac: float = 0.01
    allow_short: bool = True
    scale_limit: float = 1.0
    residual_penalty: float = 0.00005
    downside_penalty: float = 0.0
    pnl_smoothness_penalty: float = 0.0
    leverage_cap_smoothness_penalty: float = 0.0
    cap_floor_ratio: float = 0.0
    max_cap_change_per_step: Optional[float] = None
    enforce_hard_cap: bool = True
    random_start: bool = False
    episode_length: Optional[int] = None


class ResidualLeverageEnv(gym.Env[np.ndarray, np.ndarray]):
    """PPO residual-controller environment over baseline neural actions.

    The baseline policy provides per-bar prices/amounts. The RL policy learns:
    - multiplicative residual scaling of buy/sell amounts
    - a per-step leverage cap ratio in [0, 1] relative to max_leverage
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        features: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        base_buy_prices: np.ndarray,
        base_sell_prices: np.ndarray,
        base_buy_amounts: np.ndarray,
        base_sell_amounts: np.ndarray,
        timestamps: Optional[Sequence[Any]],
        config: ResidualLeverageEnvConfig,
        start_index: int,
        end_index: int,
    ) -> None:
        super().__init__()
        arrays = {
            "features": np.asarray(features, dtype=np.float32),
            "highs": np.asarray(highs, dtype=np.float32),
            "lows": np.asarray(lows, dtype=np.float32),
            "closes": np.asarray(closes, dtype=np.float32),
            "base_buy_prices": np.asarray(base_buy_prices, dtype=np.float32),
            "base_sell_prices": np.asarray(base_sell_prices, dtype=np.float32),
            "base_buy_amounts": np.asarray(base_buy_amounts, dtype=np.float32),
            "base_sell_amounts": np.asarray(base_sell_amounts, dtype=np.float32),
        }
        n = len(arrays["closes"])
        if arrays["features"].ndim != 2:
            raise ValueError(f"features must be 2D [T,F], got {arrays['features'].shape}")
        for name, arr in arrays.items():
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} does not match closes length {n}")

        self.features = arrays["features"]
        self.highs = arrays["highs"]
        self.lows = arrays["lows"]
        self.closes = arrays["closes"]
        self.base_buy_prices = arrays["base_buy_prices"]
        self.base_sell_prices = arrays["base_sell_prices"]
        self.base_buy_amounts = arrays["base_buy_amounts"]
        self.base_sell_amounts = arrays["base_sell_amounts"]
        self.timestamps = list(timestamps) if timestamps is not None else None
        self.cfg = config

        self._window = max(2, int(config.window_size))
        self._start_min = max(self._window - 1, int(start_index))
        self._end_index = int(end_index)
        if self._end_index <= self._start_min + 1:
            raise ValueError(
                f"Invalid range start={self._start_min} end={self._end_index}; need at least two tradable steps."
            )

        max_full_episode = self._end_index - self._start_min
        if config.episode_length is None:
            self._episode_length = max_full_episode
        else:
            self._episode_length = max(1, min(int(config.episode_length), max_full_episode))

        feat_dim = self.features.shape[1]
        obs_dim = self._window * feat_dim + 7
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._idx = self._start_min
        self._episode_end = self._start_min + self._episode_length
        self._cash = float(config.initial_cash)
        self._inventory = 0.0
        self._equity = float(config.initial_cash)
        self._peak_equity = float(config.initial_cash)
        self._num_trades = 0
        self._margin_cost_total = 0.0
        self._step_returns: List[float] = []
        self._equity_curve: List[float] = [float(config.initial_cash)]
        self._active_cap_ratio = 1.0
        self._prev_step_return = 0.0
        self._cap_ratio_trace: List[float] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        random_start = bool(options.get("random_start", self.cfg.random_start))
        if random_start:
            max_start = self._end_index - self._episode_length
            if max_start <= self._start_min:
                self._idx = self._start_min
            else:
                self._idx = int(self.np_random.integers(self._start_min, max_start + 1))
        else:
            self._idx = self._start_min

        self._episode_end = min(self._end_index, self._idx + self._episode_length)
        self._cash = float(self.cfg.initial_cash)
        self._inventory = 0.0
        self._equity = float(self.cfg.initial_cash)
        self._peak_equity = float(self.cfg.initial_cash)
        self._num_trades = 0
        self._margin_cost_total = 0.0
        self._step_returns = []
        self._equity_curve = [float(self.cfg.initial_cash)]
        self._active_cap_ratio = 1.0
        self._prev_step_return = 0.0
        self._cap_ratio_trace = []
        return self._get_obs(), self._info(
            0.0,
            0.0,
            1.0,
            1.0,
            cap_ratio=self._active_cap_ratio,
            dynamic_max_leverage=self.cfg.max_leverage * self._active_cap_ratio,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._idx >= self._episode_end:
            return self._get_obs(), 0.0, True, False, self._info(
                0.0,
                0.0,
                1.0,
                1.0,
                cap_ratio=self._active_cap_ratio,
                dynamic_max_leverage=self.cfg.max_leverage * self._active_cap_ratio,
            )

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 3:
            raise ValueError(f"Expected action shape (3,), got {action.shape}")

        buy_factor = self._decode_factor(action[0])
        sell_factor = self._decode_factor(action[1])
        target_cap_ratio = self._decode_cap_ratio(action[2])
        cap_ratio_prev = float(self._active_cap_ratio)
        cap_ratio = target_cap_ratio
        if self.cfg.max_cap_change_per_step is not None:
            cap_step = max(0.0, float(self.cfg.max_cap_change_per_step))
            cap_ratio = float(np.clip(cap_ratio, cap_ratio_prev - cap_step, cap_ratio_prev + cap_step))
        dynamic_max_leverage = float(self.cfg.max_leverage * cap_ratio)

        high = float(self.highs[self._idx])
        low = float(self.lows[self._idx])
        close = float(self.closes[self._idx])
        equity_before = float(self._cash + self._inventory * close)

        # Margin interest accrual (same semantics as margin simulator).
        if self._cash < 0.0:
            interest = abs(self._cash) * float(self.cfg.margin_hourly_rate)
            self._cash -= interest
            self._margin_cost_total += interest
        if self._inventory < 0.0:
            borrowed_value = abs(self._inventory) * close
            interest = borrowed_value * float(self.cfg.margin_hourly_rate)
            self._cash -= interest
            self._margin_cost_total += interest

        buy_price = float(self.base_buy_prices[self._idx])
        sell_price = float(self.base_sell_prices[self._idx])
        base_buy = float(self.base_buy_amounts[self._idx])
        base_sell = float(self.base_sell_amounts[self._idx])

        buy_amount = np.clip(base_buy * buy_factor, 0.0, 100.0) / 100.0
        sell_amount = np.clip(base_sell * sell_factor, 0.0, 100.0) / 100.0

        trades_this_step = 0
        if buy_amount > 0.0 and buy_price > 0.0 and low <= buy_price:
            max_buy_value = dynamic_max_leverage * max(equity_before, 0.0) - self._inventory * buy_price
            if max_buy_value > 0.0:
                buy_qty = buy_amount * max_buy_value / (buy_price * (1.0 + float(self.cfg.maker_fee)))
                if buy_qty > 0.0:
                    cost = buy_qty * buy_price * (1.0 + float(self.cfg.maker_fee))
                    self._cash -= cost
                    self._inventory += buy_qty
                    trades_this_step += 1

        if sell_amount > 0.0 and sell_price > 0.0 and high >= sell_price:
            if self._inventory > 0.0:
                sell_qty = min(sell_amount * self._inventory, self._inventory)
            elif self.cfg.allow_short:
                max_short_value = dynamic_max_leverage * max(equity_before, 0.0)
                sell_qty = min(
                    sell_amount * max_short_value / (sell_price * (1.0 + float(self.cfg.maker_fee))),
                    max_short_value / (sell_price * (1.0 + float(self.cfg.maker_fee))),
                )
            else:
                sell_qty = 0.0
            sell_qty = max(0.0, float(sell_qty))
            if sell_qty > 0.0:
                proceeds = sell_qty * sell_price * (1.0 - float(self.cfg.maker_fee))
                self._cash += proceeds
                self._inventory -= sell_qty
                trades_this_step += 1

        # Enforce the policy-selected leverage cap on end-of-step holdings.
        forced_deleverage_qty = 0.0
        if self.cfg.enforce_hard_cap and close > 0.0 and dynamic_max_leverage >= 0.0:
            equity_mid = float(self._cash + self._inventory * close)
            allowed_notional = dynamic_max_leverage * max(equity_mid, 0.0)
            inventory_notional = abs(self._inventory) * close
            if inventory_notional > allowed_notional + 1e-10:
                excess_notional = inventory_notional - allowed_notional
                forced_deleverage_qty = max(0.0, excess_notional / close)
                forced_deleverage_qty = min(forced_deleverage_qty, abs(self._inventory))
                if forced_deleverage_qty > 0.0:
                    if self._inventory > 0.0:
                        proceeds = forced_deleverage_qty * close * (1.0 - float(self.cfg.maker_fee))
                        self._cash += proceeds
                        self._inventory -= forced_deleverage_qty
                    else:
                        cost = forced_deleverage_qty * close * (1.0 + float(self.cfg.maker_fee))
                        self._cash -= cost
                        self._inventory += forced_deleverage_qty
                    trades_this_step += 1

        equity_after = float(self._cash + self._inventory * close)
        floor_equity = float(self.cfg.initial_cash * self.cfg.min_equity_frac)
        if not np.isfinite(equity_after):
            equity_after = floor_equity
        equity_after = max(equity_after, floor_equity)
        self._equity = equity_after

        step_return = (equity_after - equity_before) / (abs(equity_before) + 1e-12)
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = 1.0 - (self._equity / (self._peak_equity + 1e-12))

        residual_distance = abs(buy_factor - 1.0) + abs(sell_factor - 1.0)
        downside_term = max(-step_return, 0.0)
        step_smoothness_term = abs(step_return - self._prev_step_return)
        cap_change_term = abs(cap_ratio - cap_ratio_prev)
        reward = (
            step_return
            - float(self.cfg.residual_penalty) * residual_distance
            - float(self.cfg.downside_penalty) * downside_term
            - float(self.cfg.pnl_smoothness_penalty) * step_smoothness_term
            - float(self.cfg.leverage_cap_smoothness_penalty) * cap_change_term
        )

        self._idx += 1
        self._num_trades += trades_this_step
        self._step_returns.append(float(step_return))
        self._equity_curve.append(float(self._equity))
        self._active_cap_ratio = cap_ratio
        self._prev_step_return = float(step_return)
        self._cap_ratio_trace.append(float(cap_ratio))

        terminated = self._idx >= self._episode_end
        if terminated and self._inventory != 0.0:
            final_close = close
            if self._inventory > 0.0:
                self._cash += self._inventory * final_close * (1.0 - float(self.cfg.maker_fee))
            else:
                self._cash -= abs(self._inventory) * final_close * (1.0 + float(self.cfg.maker_fee))
            self._inventory = 0.0
            final_equity = float(self._cash)
            final_return = (final_equity - self._equity) / (abs(self._equity) + 1e-12)
            self._equity = max(final_equity, floor_equity)
            reward += final_return
            self._step_returns[-1] += final_return
            self._equity_curve[-1] = self._equity

        info = self._info(
            step_return,
            drawdown,
            buy_factor,
            sell_factor,
            buy_price,
            sell_price,
            float(np.clip(base_buy * buy_factor, 0.0, 100.0)),
            float(np.clip(base_sell * sell_factor, 0.0, 100.0)),
            base_buy,
            base_sell,
            high,
            low,
            close,
            cap_ratio,
            dynamic_max_leverage,
            forced_deleverage_qty,
            downside_term,
            step_smoothness_term,
            cap_change_term,
        )
        return self._get_obs(), float(reward), terminated, False, info

    def _decode_factor(self, raw: float) -> float:
        raw = float(np.clip(raw, -1.0, 1.0))
        limit = float(max(0.0, self.cfg.scale_limit))
        factor = 1.0 + raw * limit
        return float(np.clip(factor, 0.0, 1.0 + limit))

    def _decode_cap_ratio(self, raw: float) -> float:
        clipped = float(np.clip(raw, -1.0, 1.0))
        unit = 0.5 * (clipped + 1.0)
        floor = float(np.clip(self.cfg.cap_floor_ratio, 0.0, 1.0))
        return float(np.clip(floor + (1.0 - floor) * unit, floor, 1.0))

    def _get_obs(self) -> np.ndarray:
        start = self._idx - self._window + 1
        window = self.features[start : self._idx + 1]
        flat = window.reshape(-1)

        close = float(self.closes[min(self._idx, self._end_index - 1)])
        equity = float(self._cash + self._inventory * close)
        equity = max(equity, 1e-8)
        inv_value = self._inventory * close

        base_buy = float(self.base_buy_amounts[min(self._idx, self._end_index - 1)]) / 100.0
        base_sell = float(self.base_sell_amounts[min(self._idx, self._end_index - 1)]) / 100.0
        extras = np.asarray(
            [
                float(np.clip(inv_value / equity, -float(self.cfg.max_leverage), float(self.cfg.max_leverage))),
                float(np.clip(self._cash / equity, -float(self.cfg.max_leverage), float(self.cfg.max_leverage))),
                float(base_buy),
                float(base_sell),
                float(np.log(max(equity / float(self.cfg.initial_cash), 1e-8))),
                float(np.clip(self._active_cap_ratio, 0.0, 1.0)),
                float(np.clip(self._prev_step_return, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return np.concatenate([flat.astype(np.float32), extras], dtype=np.float32)

    def _info(
        self,
        step_return: float,
        drawdown: float,
        buy_factor: float,
        sell_factor: float,
        buy_price: float = 0.0,
        sell_price: float = 0.0,
        buy_amount_scaled: float = 0.0,
        sell_amount_scaled: float = 0.0,
        base_buy_amount: float = 0.0,
        base_sell_amount: float = 0.0,
        high: float = 0.0,
        low: float = 0.0,
        close: float = 0.0,
        cap_ratio: float = 1.0,
        dynamic_max_leverage: float = 0.0,
        forced_deleverage_qty: float = 0.0,
        downside_term: float = 0.0,
        step_smoothness_term: float = 0.0,
        cap_change_term: float = 0.0,
    ) -> Dict[str, Any]:
        timestamp = None
        if self.timestamps is not None and 0 <= self._idx < len(self.timestamps):
            timestamp = self.timestamps[self._idx]
        return {
            "timestamp": timestamp,
            "equity": float(self._equity),
            "cash": float(self._cash),
            "inventory": float(self._inventory),
            "step_return": float(step_return),
            "drawdown": float(drawdown),
            "buy_factor": float(buy_factor),
            "sell_factor": float(sell_factor),
            "buy_price": float(buy_price),
            "sell_price": float(sell_price),
            "buy_amount_scaled": float(buy_amount_scaled),
            "sell_amount_scaled": float(sell_amount_scaled),
            "base_buy_amount": float(base_buy_amount),
            "base_sell_amount": float(base_sell_amount),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "cap_ratio": float(cap_ratio),
            "dynamic_max_leverage": float(dynamic_max_leverage),
            "forced_deleverage_qty": float(forced_deleverage_qty),
            "downside_term": float(downside_term),
            "step_smoothness_term": float(step_smoothness_term),
            "cap_change_term": float(cap_change_term),
            "num_trades": int(self._num_trades),
            "margin_cost_total": float(self._margin_cost_total),
        }

    @property
    def metrics(self) -> Dict[str, float]:
        returns = np.asarray(self._step_returns, dtype=np.float64)
        equity = np.asarray(self._equity_curve, dtype=np.float64)
        if len(returns) == 0:
            return {
                "total_return": 0.0,
                "sortino": 0.0,
                "max_drawdown": 0.0,
                "final_equity": float(self._equity),
                "num_trades": float(self._num_trades),
                "margin_cost_total": float(self._margin_cost_total),
                "margin_cost_pct": float(self._margin_cost_total / max(self.cfg.initial_cash, 1e-9) * 100.0),
            }

        downside = returns[returns < 0.0]
        downside_std = float(np.std(downside)) if len(downside) > 0 else 0.0
        sortino = 0.0
        if downside_std > 0.0:
            sortino = float(np.mean(returns) / (downside_std + 1e-12) * np.sqrt(HOURLY_PERIODS_PER_YEAR))

        running_max = np.maximum.accumulate(equity)
        max_dd = float(np.min((equity - running_max) / (running_max + 1e-12)))
        cap_ratios = np.asarray(self._cap_ratio_trace, dtype=np.float64) if self._cap_ratio_trace else np.zeros(0, dtype=np.float64)
        step_deltas = np.diff(returns) if len(returns) > 1 else np.zeros(0, dtype=np.float64)
        return {
            "total_return": float(equity[-1] / equity[0] - 1.0),
            "sortino": sortino,
            "max_drawdown": max_dd,
            "final_equity": float(equity[-1]),
            "num_trades": float(self._num_trades),
            "margin_cost_total": float(self._margin_cost_total),
            "margin_cost_pct": float(self._margin_cost_total / max(self.cfg.initial_cash, 1e-9) * 100.0),
            "avg_cap_ratio": float(np.mean(cap_ratios)) if len(cap_ratios) > 0 else 0.0,
            "cap_ratio_std": float(np.std(cap_ratios)) if len(cap_ratios) > 0 else 0.0,
            "step_return_std": float(np.std(returns)),
            "step_return_delta_std": float(np.std(step_deltas)) if len(step_deltas) > 0 else 0.0,
        }


class _FixedResidualActionModel:
    def __init__(self, action: np.ndarray) -> None:
        self._action = np.asarray(action, dtype=np.float32)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return self._action, None


def evaluate_deterministic_episode(model: Any, env: ResidualLeverageEnv) -> Dict[str, Any]:
    obs, _ = env.reset(options={"random_start": False})
    done = False
    steps: List[Dict[str, Any]] = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        steps.append(info)
        done = bool(terminated or truncated)
    return {"metrics": env.metrics, "steps": steps}


def evaluate_baseline_episode(env: ResidualLeverageEnv) -> Dict[str, Any]:
    # Neutral residuals + full dynamic leverage cap.
    return evaluate_deterministic_episode(_FixedResidualActionModel(np.array([0.0, 0.0, 1.0], dtype=np.float32)), env)


__all__ = [
    "HOURLY_PERIODS_PER_YEAR",
    "ResidualLeverageEnv",
    "ResidualLeverageEnvConfig",
    "evaluate_baseline_episode",
    "evaluate_deterministic_episode",
]
