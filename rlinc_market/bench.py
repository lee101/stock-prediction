from __future__ import annotations

import os
import time
import numpy as np

from rlinc_market import RlincMarketEnv


def run_solo(n_steps: int = 200_000, n_assets: int = 8, window: int = 32) -> float:
    env = RlincMarketEnv(n_assets=n_assets, window=window, episode_len=n_steps + 1)
    obs, _ = env.reset()
    a = np.zeros((n_assets,), dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        obs, r, term, trunc, info = env.step(a)
    t1 = time.perf_counter()
    sps = n_steps / (t1 - t0)
    return sps


if __name__ == "__main__":
    sps = run_solo()
    print(f"Solo env speed: {sps:,.0f} steps/s")

