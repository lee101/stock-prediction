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
        action_allocation_bins: int = 1,
        action_level_bins: int = 1,
        action_max_offset_bps: float = 0.0,
        cash_penalty: float = 0.01,
        drawdown_penalty: float = 0.0,
        downside_penalty: float = 0.0,
        smooth_downside_penalty: float = 0.0,
        smooth_downside_temperature: float = 0.02,
        trade_penalty: float = 0.0,
        smoothness_penalty: float = 0.0,
        long_only: bool = False,
    ):
        self.data_path = str(Path(data_path).resolve())
        self.max_steps = max_steps
        self.fee_rate = fee_rate
        self.max_leverage = max_leverage
        self.periods_per_year = float(periods_per_year)
        self.num_symbols = num_symbols
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.action_allocation_bins = max(1, int(action_allocation_bins))
        self.action_level_bins = max(1, int(action_level_bins))
        self.action_max_offset_bps = max(0.0, float(action_max_offset_bps))
        self.cash_penalty = cash_penalty
        self.drawdown_penalty = drawdown_penalty
        self.downside_penalty = downside_penalty
        self.smooth_downside_penalty = smooth_downside_penalty
        self.smooth_downside_temperature = smooth_downside_temperature
        self.trade_penalty = trade_penalty
        self.smoothness_penalty = smoothness_penalty
        self.long_only = long_only


def _load_binding():
    """Load the compiled C binding module."""
    return importlib.import_module("pufferlib_market.binding")


class TradingEnv(GymnasiumPufferEnv):
    """
    PufferLib C trading environment.

    Action space (Discrete):
      0         = go flat (sell position)
      1..K      = go long(symbol, alloc_bin, level_bin)
      K+1..2K   = go short(symbol, alloc_bin, level_bin)
      where K = S * allocation_bins * level_bins

    Observation space (Box):
      [S * 16 forecast features] + [5 portfolio state] + [S one-hot position]
    """

    def __init__(self, config: TradingEnvConfig = None, buf=None):
        if config is None:
            config = TradingEnvConfig()
        self.config = config
        S = config.num_symbols
        per_symbol_actions = config.action_allocation_bins * config.action_level_bins

        obs_size = S * 16 + 5 + S   # features + portfolio + position encoding
        if config.long_only:
            num_actions = 1 + S * per_symbol_actions  # flat + longs only
        else:
            num_actions = 1 + 2 * S * per_symbol_actions  # flat + longs + shorts

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
            action_allocation_bins=config.action_allocation_bins,
            action_level_bins=config.action_level_bins,
            action_max_offset_bps=config.action_max_offset_bps,
            cash_penalty=config.cash_penalty,
            drawdown_penalty=config.drawdown_penalty,
            downside_penalty=config.downside_penalty,
            smooth_downside_penalty=config.smooth_downside_penalty,
            smooth_downside_temperature=config.smooth_downside_temperature,
            trade_penalty=config.trade_penalty,
            smoothness_penalty=config.smoothness_penalty,
        )

    @staticmethod
    def load_shared_data(data_path: str):
        """Load market data once per process (call before creating envs)."""
        binding = _load_binding()
        binding.shared(data_path=str(Path(data_path).resolve()))


def make_env(config: TradingEnvConfig = None, buf=None):
    """Factory function for PufferLib env creation."""
    return TradingEnv(config=config, buf=buf)
