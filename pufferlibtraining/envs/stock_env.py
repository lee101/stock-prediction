import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional


class StockTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        max_position_size: float = 1.0,
        features: Optional[List[str]] = None,
        spread_pct: float = 0.0001,
        slippage_pct: float = 0.0001,
        min_commission: float = 1.0,
    ):
        super().__init__()

        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct
        self.min_commission = min_commission

        self.features = features or ["Open", "High", "Low", "Close", "Volume"]

        self.prices = self.df[["Open", "Close"]].values
        self.feature_data = self.df[self.features].values

        self.n_days = max(1, len(self.df) - self.window_size - 1)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, len(self.features) + 3),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.position = 0.0
        self.entry_price = 0.0
        self.trades: List[Dict[str, Any]] = []
        self.returns: List[float] = []
        self.positions_history: List[float] = []
        self.balance_history: List[float] = [self.initial_balance]
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        start_idx = self.current_step
        end_idx = start_idx + self.window_size
        window_data = self.feature_data[start_idx:end_idx]

        normalized_data = (window_data - np.mean(window_data, axis=0)) / (np.std(window_data, axis=0) + 1e-8)

        position_info = np.full((self.window_size, 1), self.position, dtype=np.float32)

        balance_ratio = self.balance / self.initial_balance
        balance_info = np.full((self.window_size, 1), balance_ratio, dtype=np.float32)

        if self.position != 0 and self.entry_price > 0:
            current_price = self.prices[end_idx - 1, 1]
            pnl = (current_price - self.entry_price) / max(1e-8, self.entry_price) * self.position
        else:
            pnl = 0.0
        pnl_info = np.full((self.window_size, 1), pnl, dtype=np.float32)

        observation = np.concatenate([normalized_data, position_info, balance_info, pnl_info], axis=1)
        return observation.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = float(np.clip(action[0], -1.0, 1.0))

        current_idx = self.current_step + self.window_size
        current_open = self.prices[current_idx, 0]
        current_close = self.prices[current_idx, 1]

        old_position = self.position
        new_position = action * self.max_position_size

        reward = 0.0

        if old_position != 0:
            position_return = (current_close - current_open) / max(1e-8, current_open)
            profit = position_return * abs(old_position) * (1 if old_position > 0 else -1)
            reward += profit * self.balance
            self.balance *= (1 + profit)

        if old_position != new_position:
            position_change = abs(new_position - old_position)
            trade_value = position_change * self.balance
            commission = max(self.transaction_cost * trade_value, self.min_commission)
            spread_cost = self.spread_pct * trade_value
            slippage_cost = self.slippage_pct * trade_value
            total_cost = commission + spread_cost + slippage_cost
            self.balance -= total_cost
            reward -= total_cost / self.initial_balance
            self.entry_price = current_close if new_position != 0 else 0.0
            self.trades.append(
                {
                    "step": int(self.current_step),
                    "action": float(action),
                    "old_position": float(old_position),
                    "new_position": float(new_position),
                    "price": float(current_close),
                    "balance": float(self.balance),
                }
            )

        self.position = new_position
        self.positions_history.append(self.position)
        self.balance_history.append(self.balance)

        reward = float(reward / self.initial_balance)

        self.current_step += 1
        terminated = self.current_step >= self.n_days
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        daily_return = (
            (self.balance - self.balance_history[-2]) / max(1e-8, self.balance_history[-2])
            if len(self.balance_history) > 1
            else 0.0
        )
        self.returns.append(float(daily_return))

        info = {
            "balance": float(self.balance),
            "position": float(self.position),
            "trades": len(self.trades),
            "current_price": float(current_close),
            "daily_return": float(daily_return),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, Position: {self.position:.3f}")

    def get_metrics(self) -> Dict[str, float]:
        if len(self.returns) == 0:
            return {}
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        returns_array = np.array(self.returns)
        sharpe = (
            (returns_array.mean() / (returns_array.std() + 1e-8)) * np.sqrt(252)
            if len(returns_array) > 0
            else 0.0
        )
        cumulative = np.cumprod(1 + returns_array) if len(returns_array) else np.array([1.0])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / np.maximum(running_max, 1e-8)
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0
        winning_trades = sum(1 for t in self.trades if t.get("profit", 0) > 0)
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "num_trades": int(total_trades),
            "win_rate": float(win_rate),
            "final_balance": float(self.balance),
        }

