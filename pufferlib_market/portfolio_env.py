"""
Multi-position Portfolio RL Environment.
Learns to allocate across multiple symbols simultaneously.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import List, Optional, Tuple
import struct


class PortfolioEnv(gym.Env):
    """
    Portfolio RL environment where agent controls allocation across all symbols.

    Observation: [per_symbol_features, current_allocations, portfolio_state]
    Action: target_allocations[num_symbols] - continuous [0,1] per symbol
    Reward: hourly return * sortino_adjustment
    """

    def __init__(
        self,
        data_path: str,
        max_steps: int = 720,
        fee_rate: float = 0.001,
        initial_cash: float = 10000.0,
        features_per_symbol: int = 16,
        discrete_bins: int = 0,  # 0 = continuous, >0 = discretized
        sortino_reward: bool = True,
        downside_penalty: float = 2.0,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.fee_rate = fee_rate
        self.initial_cash = initial_cash
        self.features_per_symbol = features_per_symbol
        self.discrete_bins = discrete_bins
        self.sortino_reward = sortino_reward
        self.downside_penalty = downside_penalty

        # Load data
        self._load_data(data_path)

        # Observation: features + allocations + portfolio state
        # per_symbol: 16 features + 1 current_allocation = 17 per symbol
        # portfolio: cash_ratio, total_value_ratio, step_ratio, drawdown, recent_return
        obs_dim = self.num_symbols * (features_per_symbol + 1) + 5
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action: allocation per symbol
        if discrete_bins > 0:
            # Discrete: each symbol gets one of N allocation levels
            self.action_space = spaces.MultiDiscrete([discrete_bins] * self.num_symbols)
        else:
            # Continuous: allocation % per symbol [0, 1]
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.num_symbols,), dtype=np.float32
            )

        self.reset()

    def _load_data(self, data_path: str):
        """Load MKTD binary format."""
        with open(data_path, 'rb') as f:
            magic = f.read(4)
            assert magic == b'MKTD', f"Invalid magic: {magic}"

            header = struct.unpack('<IIIII40s', f.read(60))
            version, num_symbols, num_timesteps, features_per_sym, price_features, _ = header

            self.num_symbols = num_symbols
            self.num_timesteps = num_timesteps

            # Read symbol names
            self.symbols = []
            for _ in range(num_symbols):
                name = f.read(16).rstrip(b'\x00').decode('ascii')
                self.symbols.append(name)

            # Read features [T, S, 16]
            feat_size = num_timesteps * num_symbols * features_per_sym
            feat_data = np.frombuffer(f.read(feat_size * 4), dtype=np.float32)
            self.features = feat_data.reshape(num_timesteps, num_symbols, features_per_sym)

            # Read prices [T, S, 5] (open, high, low, close, volume)
            price_size = num_timesteps * num_symbols * price_features
            price_data = np.frombuffer(f.read(price_size * 4), dtype=np.float32)
            self.prices = price_data.reshape(num_timesteps, num_symbols, price_features)

            # Close prices for trading
            self.close_prices = self.prices[:, :, 3]  # [T, S]

        print(f"Loaded {num_symbols} symbols, {num_timesteps} timesteps")

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Random start point (leave room for max_steps)
        max_start = self.num_timesteps - self.max_steps - 1
        self.start_idx = self.np_random.integers(0, max(1, max_start))
        self.step_idx = 0

        # Portfolio state
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.num_symbols, dtype=np.float32)  # shares held
        self.allocations = np.zeros(self.num_symbols, dtype=np.float32)  # target allocations

        # Tracking
        self.portfolio_values = [self.initial_cash]
        self.returns = []
        self.peak_value = self.initial_cash

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        t = self.start_idx + self.step_idx

        # Per-symbol features + current allocation
        obs_parts = []
        for s in range(self.num_symbols):
            feat = self.features[t, s]  # [16]
            alloc = self.allocations[s:s+1]  # [1]
            obs_parts.append(np.concatenate([feat, alloc]))

        # Portfolio state
        total_value = self._portfolio_value()
        cash_ratio = self.cash / max(total_value, 1e-8)
        value_ratio = total_value / self.initial_cash
        step_ratio = self.step_idx / self.max_steps
        drawdown = (self.peak_value - total_value) / max(self.peak_value, 1e-8)
        recent_return = self.returns[-1] if self.returns else 0.0

        portfolio_state = np.array([
            cash_ratio, value_ratio, step_ratio, drawdown, recent_return
        ], dtype=np.float32)

        obs = np.concatenate(obs_parts + [portfolio_state])
        return obs.astype(np.float32)

    def _portfolio_value(self) -> float:
        t = self.start_idx + self.step_idx
        prices = self.close_prices[t]
        holdings_value = np.sum(self.holdings * prices)
        return self.cash + holdings_value

    def _rebalance(self, target_allocations: np.ndarray):
        """Rebalance portfolio to target allocations."""
        t = self.start_idx + self.step_idx
        prices = self.close_prices[t]

        # Normalize allocations to sum <= 1
        total_alloc = np.sum(target_allocations)
        if total_alloc > 1.0:
            target_allocations = target_allocations / total_alloc

        total_value = self._portfolio_value()
        target_values = target_allocations * total_value
        current_values = self.holdings * prices

        # Calculate trades needed
        value_changes = target_values - current_values

        total_fees = 0.0
        for s in range(self.num_symbols):
            if abs(value_changes[s]) > 1.0:  # Min trade size
                trade_value = abs(value_changes[s])
                fee = trade_value * self.fee_rate
                total_fees += fee

                if value_changes[s] > 0:  # Buy
                    shares_to_buy = (value_changes[s] - fee) / prices[s]
                    cost = shares_to_buy * prices[s] + fee
                    if cost <= self.cash:
                        self.holdings[s] += shares_to_buy
                        self.cash -= cost
                else:  # Sell
                    shares_to_sell = min(-value_changes[s] / prices[s], self.holdings[s])
                    proceeds = shares_to_sell * prices[s] - fee
                    self.holdings[s] -= shares_to_sell
                    self.cash += proceeds

        # Update allocations
        self.allocations = target_allocations.copy()
        return total_fees

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Convert action to allocations
        if self.discrete_bins > 0:
            # Discrete bins to continuous
            target_allocations = action.astype(np.float32) / (self.discrete_bins - 1)
        else:
            target_allocations = np.clip(action, 0, 1).astype(np.float32)

        # Get value before rebalance
        value_before = self._portfolio_value()

        # Rebalance
        fees = self._rebalance(target_allocations)

        # Advance time
        self.step_idx += 1
        t = self.start_idx + self.step_idx

        # Get value after price change
        value_after = self._portfolio_value()

        # Calculate return
        ret = (value_after - value_before) / max(value_before, 1e-8)
        self.returns.append(ret)
        self.portfolio_values.append(value_after)
        self.peak_value = max(self.peak_value, value_after)

        # Reward
        if self.sortino_reward:
            # Penalize downside more
            if ret < 0:
                reward = ret * self.downside_penalty
            else:
                reward = ret
        else:
            reward = ret

        # Scale reward
        reward = reward * 100  # Scale to reasonable range

        # Done conditions
        terminated = self.step_idx >= self.max_steps
        truncated = t >= self.num_timesteps - 1

        # Info
        info = {
            'portfolio_value': value_after,
            'return': ret,
            'fees': fees,
            'allocations': self.allocations.copy(),
            'step': self.step_idx,
        }

        if terminated or truncated:
            # Episode stats
            total_return = (value_after - self.initial_cash) / self.initial_cash
            returns_arr = np.array(self.returns)
            neg_returns = returns_arr[returns_arr < 0]
            downside_std = np.std(neg_returns) if len(neg_returns) > 0 else 1e-8
            mean_return = np.mean(returns_arr)
            sortino = mean_return / max(downside_std, 1e-8) * np.sqrt(252 * 24)

            info['total_return'] = total_return
            info['sortino'] = sortino
            info['num_trades'] = len([r for r in self.returns if abs(r) > 0.0001])

        return self._get_obs(), reward, terminated, truncated, info


def make_portfolio_env(data_path: str, **kwargs):
    """Factory for creating portfolio environments."""
    return PortfolioEnv(data_path, **kwargs)


if __name__ == "__main__":
    # Test
    env = PortfolioEnv("pufferlib_market/data/stocks10_data.bin", discrete_bins=5)
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Symbols: {env.symbols}")

    # Random episode
    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc:
            break

    print(f"Total reward: {total_reward:.2f}")
    print(f"Final info: {info}")
