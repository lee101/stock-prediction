import numpy as np
import pandas as pd
import torch

from pufferlibtraining.market_env import MarketEnv


def _write_dummy_data(tmp_path, symbol="TEST", rows=400):
    idx = np.arange(rows)
    data = pd.DataFrame(
        {
            "timestamps": idx,
            "open": np.linspace(100, 110, rows) + np.random.randn(rows) * 0.5,
            "high": np.linspace(101, 111, rows) + np.random.randn(rows) * 0.5,
            "low": np.linspace(99, 109, rows) + np.random.randn(rows) * 0.5,
            "close": np.linspace(100, 112, rows) + np.random.randn(rows) * 0.5,
            "volume": np.random.lognormal(mean=12, sigma=0.1, size=rows),
        }
    )
    path = tmp_path / f"{symbol}.csv"
    data.to_csv(path, index=False)
    return path.parent


def test_market_env_step_shapes(tmp_path):
    data_dir = _write_dummy_data(tmp_path)
    env = MarketEnv(
        data_dir=str(data_dir),
        tickers=["TEST"],
        context_len=16,
        episode_len=32,
        seed=42,
        device="cpu",
        precision="fp32",
    )

    obs, info = env.reset()
    assert obs.shape == (16, env.observation_space.shape[-1])

    next_obs, reward, terminated, truncated, info = env.step(np.zeros((1,), dtype=np.float32))
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_tensor" in info
    assert isinstance(info["reward_tensor"], torch.Tensor)


def test_market_env_random_episode(tmp_path):
    data_dir = _write_dummy_data(tmp_path)
    env = MarketEnv(
        data_dir=str(data_dir),
        tickers=["TEST"],
        context_len=8,
        episode_len=10,
        seed=7,
        device="cpu",
        precision="fp32",
    )
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
    assert np.isfinite(total_reward)
    assert abs(total_reward) < 10.0  # sanity: reward should be bounded for synthetic data

