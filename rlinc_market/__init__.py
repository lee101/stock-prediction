"""rlinc_market: C-backed market simulator with Gym wrapper.

Usage:
    from rlinc_market.env import RlincMarketEnv
    env = RlincMarketEnv(n_assets=8, window=32)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from .env import RlincMarketEnv

__all__ = ["RlincMarketEnv"]

