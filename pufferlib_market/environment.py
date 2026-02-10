"""
PufferLib-compatible Python wrapper for the C trading environment.
"""

import importlib
import numpy as np
import gymnasium
from pathlib import Path

from pufferlib.emulation import GymnasiumPufferEnv


class TradingEnvConfig:
    """Configuration for the C trading environment."""

    def __init__(
        self,
        data_path: str = "pufferlib_market/data/market_data.bin",
        max_steps: int = 720,       # 30 days
        fee_rate: float = 0.001,
        max_leverage: float = 1.0,
        periods_per_year: float = 8760.0,
        num_symbols: int = 14,      # read from binary header, but needed for space defs
        reward_scale: float = 10.0,
        reward_clip: float = 5.0,
        cash_penalty: float = 0.01,
        drawdown_penalty: float = 0.0,
        downside_penalty: float = 0.0,
        trade_penalty: float = 0.0,
    ):
        self.data_path = str(Path(data_path).resolve())
        self.max_steps = max_steps
        self.fee_rate = fee_rate
        self.max_leverage = max_leverage
        self.periods_per_year = float(periods_per_year)
        self.num_symbols = num_symbols
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.cash_penalty = cash_penalty
        self.drawdown_penalty = drawdown_penalty
        self.downside_penalty = downside_penalty
        self.trade_penalty = trade_penalty


def _load_binding():
    """Load the compiled C binding module."""
    return importlib.import_module("pufferlib_market.binding")


class TradingEnv(GymnasiumPufferEnv):
    """
    PufferLib C trading environment.

    Action space (Discrete):
      0         = go flat (sell position)
      1..S      = go long symbol i
      S+1..2S   = go short symbol i

    Observation space (Box):
      [S * 16 forecast features] + [5 portfolio state] + [S one-hot position]
    """

    def __init__(self, config: TradingEnvConfig = None, buf=None):
        if config is None:
            config = TradingEnvConfig()
        self.config = config
        S = config.num_symbols

        obs_size = S * 16 + 5 + S   # features + portfolio + position encoding
        num_actions = 1 + 2 * S      # flat + long each + short each

        super().__init__(
            env_creator=None,
            env_module="pufferlib_market.binding",
            observation_size=obs_size,
            action_size=num_actions,
            buf=buf,
            max_steps=config.max_steps,
            fee_rate=config.fee_rate,
            max_leverage=config.max_leverage,
            periods_per_year=config.periods_per_year,
            reward_scale=config.reward_scale,
            reward_clip=config.reward_clip,
            cash_penalty=config.cash_penalty,
            drawdown_penalty=config.drawdown_penalty,
            downside_penalty=config.downside_penalty,
            trade_penalty=config.trade_penalty,
        )

    @staticmethod
    def load_shared_data(data_path: str):
        """Load market data once per process (call before creating envs)."""
        binding = _load_binding()
        binding.shared(data_path=str(Path(data_path).resolve()))


def make_env(config: TradingEnvConfig = None, buf=None):
    """Factory function for PufferLib env creation."""
    return TradingEnv(config=config, buf=buf)
