"""
PufferLib-compatible Market Trading Environment

Fast C-based market simulation for RL training with CUDA acceleration via libtorch.
"""

import numpy as np
from typing import Dict, Tuple, Any

try:
    from pufferlib import namespace
    from pufferlib.emulation import GymnasiumPufferEnv
    PUFFERLIB_AVAILABLE = True
except ImportError:
    PUFFERLIB_AVAILABLE = False
    print("Warning: pufferlib not available. Install with: pip install pufferlib")


class MarketSimConfig:
    """Configuration for market simulation environment"""

    def __init__(
        self,
        num_assets: int = 10,
        num_agents: int = 1,
        max_steps: int = 1000,
        initial_cash: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position_size: float = 0.25,
        leverage: float = 1.0,
    ):
        self.num_assets = num_assets
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.leverage = leverage


if PUFFERLIB_AVAILABLE:
    class MarketSimEnv(GymnasiumPufferEnv):
        """
        High-performance market trading environment using C backend

        Action Space:
        - 0: Hold
        - 1 to N: Buy asset i (where i = action - 1)
        - N+1 to 2N: Sell asset i (where i = action - N - 1)

        Observation Space:
        - Price history: OHLCV data for lookback window
        - Portfolio state: current positions, cash, total value
        - Step progress: normalized timestep
        """

        def __init__(self, config: MarketSimConfig = None, render_mode: str = None):
            if config is None:
                config = MarketSimConfig()

            self.config = config

            # Observation size calculation
            # OHLCV (5) * MAX_HISTORY (256) + positions (num_assets) + cash + value + progress
            obs_size = 5 * 256 + config.num_assets + 3

            # Action space: hold + buy each asset + sell each asset
            num_actions = 1 + 2 * config.num_assets

            # Initialize base environment
            super().__init__(
                driver_env=None,  # Will load C module
                c_envs=None,
                num_envs=config.num_agents,
                env_module="market_sim_c.binding",  # C module to load
                observation_size=obs_size,
                action_size=num_actions,
                # Pass config to C environment
                num_assets=config.num_assets,
                num_agents=config.num_agents,
                max_steps=config.max_steps,
            )

        def reset(
            self, seed: int = None, options: Dict = None
        ) -> Tuple[np.ndarray, Dict]:
            """Reset environment to initial state"""
            obs = super().reset(seed=seed)
            info = {}
            return obs, info

        def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
            """
            Execute actions in the environment

            Args:
                actions: Array of actions [num_agents]

            Returns:
                observations, rewards, dones, truncated, info
            """
            return super().step(actions)


# Pure Python fallback (slower, for testing)
class MarketSimPython:
    """Python-only implementation for testing without C compilation"""

    def __init__(self, config: MarketSimConfig = None):
        if config is None:
            config = MarketSimConfig()
        self.config = config
        self.reset()

    def reset(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.cash = np.full(self.config.num_agents, self.config.initial_cash)
        self.positions = np.zeros((self.config.num_agents, self.config.num_assets))
        self.prices = np.random.uniform(50, 150, self.config.num_assets)
        self.price_history = [self.prices.copy()]

        obs = self._get_observation()
        return obs, {}

    def step(self, actions):
        # Update prices (simple random walk)
        self.prices *= (1 + np.random.normal(0, 0.02, self.config.num_assets))
        self.prices = np.maximum(self.prices, 0.01)
        self.price_history.append(self.prices.copy())

        rewards = np.zeros(self.config.num_agents)
        dones = np.zeros(self.config.num_agents, dtype=bool)

        for i in range(self.config.num_agents):
            old_value = self._portfolio_value(i)

            # Execute action
            action = actions[i] if isinstance(actions, np.ndarray) else actions
            self._execute_trade(i, action)

            new_value = self._portfolio_value(i)
            rewards[i] = (new_value - old_value) / old_value * 100

        self.current_step += 1
        dones = self.current_step >= self.config.max_steps

        obs = self._get_observation()
        truncated = np.zeros_like(dones)
        info = {}

        return obs, rewards, dones, truncated, info

    def _portfolio_value(self, agent_idx):
        return self.cash[agent_idx] + np.sum(
            self.positions[agent_idx] * self.prices
        )

    def _execute_trade(self, agent_idx, action):
        if action == 0:  # Hold
            return

        asset_idx = (action - 1) % self.config.num_assets
        is_buy = (action - 1) < self.config.num_assets

        if is_buy:
            # Buy
            max_value = self._portfolio_value(agent_idx) * self.config.max_position_size
            shares = min(
                self.cash[agent_idx] / self.prices[asset_idx],
                max_value / self.prices[asset_idx],
            )
            cost = shares * self.prices[asset_idx] * (1 + self.config.transaction_cost)

            if cost <= self.cash[agent_idx]:
                self.cash[agent_idx] -= cost
                self.positions[agent_idx, asset_idx] += shares
        else:
            # Sell
            if self.positions[agent_idx, asset_idx] > 0:
                proceeds = (
                    self.positions[agent_idx, asset_idx]
                    * self.prices[asset_idx]
                    * (1 - self.config.transaction_cost)
                )
                self.cash[agent_idx] += proceeds
                self.positions[agent_idx, asset_idx] = 0

    def _get_observation(self):
        # Simplified observation
        obs = np.concatenate(
            [
                self.prices / 100.0,  # Normalized prices
                self.positions.flatten(),
                self.cash / self.config.initial_cash,
                [self.current_step / self.config.max_steps],
            ]
        )
        return obs


def make_env(config: MarketSimConfig = None, use_python: bool = False):
    """
    Factory function to create market simulation environment

    Args:
        config: Environment configuration
        use_python: If True, use pure Python implementation (slower)

    Returns:
        Environment instance
    """
    if use_python or not PUFFERLIB_AVAILABLE:
        return MarketSimPython(config)
    return MarketSimEnv(config)
