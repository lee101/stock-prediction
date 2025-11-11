import os
import time
import numpy as np
import pytest

from rlinc_market import RlincMarketEnv


def test_env_basic_shapes():
    env = RlincMarketEnv(n_assets=4, window=8, episode_len=16, leverage_limit=1.0)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (4 * (8 + 1),)
    a = env.action_space.sample()
    obs2, r, term, trunc, info = env.step(a)
    assert obs2.shape == obs.shape
    assert isinstance(r, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)


def test_rollout_episode_ends():
    env = RlincMarketEnv(n_assets=2, window=4, episode_len=5)
    env.reset()
    done = False
    steps = 0
    while not done:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        steps += 1
        done = term or trunc
    assert steps == 5


@pytest.mark.parametrize("num_envs", [1, 4])
def test_pufferlib_vectorization(num_envs):
    pytest.importorskip("pufferlib")
    import pufferlib.emulation
    import pufferlib.vector as pv

    if not os.access("/dev/shm", os.W_OK):
        pytest.skip("Shared memory unavailable in sandbox; skipping pufferlib vector test")

    envf = lambda: pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=lambda: RlincMarketEnv(n_assets=3, window=6, episode_len=16)
    )
    try:
        vec = pv.make(
            [envf] * num_envs,
            env_args=[[] for _ in range(num_envs)],
            env_kwargs=[{} for _ in range(num_envs)],
            backend=pv.Multiprocessing,
            num_envs=num_envs,
            num_workers=1,
        )
    except PermissionError:
        pytest.skip("/dev/shm permissions blocked; skipping pufferlib vector test")
    obs = vec.reset()
    for _ in range(8):
        actions = np.stack([np.random.uniform(-1, 1, size=(3,)).astype(np.float32) for _ in range(num_envs)], axis=0)
        obs, rew, term, trunc, info = vec.step(actions)
    vec.close()


def test_leverage_policy_and_financing():
    # steps_per_day=2 so every second step is a close; finance charged at next open
    env = RlincMarketEnv(
        n_assets=2,
        window=2,
        episode_len=6,
        steps_per_day=2,
        intraday_leverage_max=4.0,
        overnight_leverage_max=2.0,
        trading_fee_bps=0.0,
        return_sigma=0.0,
    )

    obs, _ = env.reset()
    # Step 0 (open day): push action with huge leverage; intraday should clamp to 4x, not close yet
    a = np.array([3.0, 3.0], dtype=np.float32)  # L1=6 -> clamp to 4
    obs, r, term, trunc, info = env.step(a)
    st = env._cenv.state()
    assert pytest.approx(st["l1"], rel=0, abs=1e-5) == 4.0

    # Step 1 (close): keep high leverage; auto-deleverage to 2x at close
    obs, r, term, trunc, info = env.step(a)
    st = env._cenv.state()
    assert st["l1"] <= 2.00001

    # Step 2 (open): financing applies on overnight leverage above 1x (we held 2x)
    # With sigma=0 and fees=0, reward should be exactly -daily_rate*(2-1)
    rate_daily = 0.065 / 252.0
    obs, r, term, trunc, info = env.step(np.zeros((2,), dtype=np.float32))
    assert pytest.approx(r, rel=0, abs=1e-7) == -rate_daily
