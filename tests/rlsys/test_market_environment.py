import numpy as np

from rlsys.config import MarketConfig
from rlsys.market_environment import MarketEnvironment


def test_market_environment_step_and_metrics():
    prices = np.linspace(100.0, 110.0, num=120, dtype=np.float64)
    feature_dim = 5
    features = np.stack(
        [
            np.linspace(0.1, 1.0, num=120, dtype=np.float32) + i
            for i in range(feature_dim)
        ],
        axis=1,
    )
    config = MarketConfig(
        initial_capital=100_000.0,
        max_leverage=2.0,
        transaction_cost=0.0001,
        slippage=0.0001,
        market_impact=0.0,
        risk_aversion=0.0,
        max_position_change=0.5,
    )
    env = MarketEnvironment(prices=prices, features=features, config=config)

    observation, info = env.reset()
    assert observation.shape[0] == feature_dim + 3
    assert info == {}

    total_reward = 0.0
    for _ in range(50):
        action = np.array([0.4], dtype=np.float32)
        observation, reward, done, truncated, info = env.step(action)
        assert np.isfinite(reward)
        total_reward += reward
        render = env.render()
        assert abs(render["position"]) <= config.max_leverage + 1e-6
        if done or truncated:
            assert "episode_reward" in info
            assert np.isfinite(info["episode_sharpe"])
            assert "episode_sortino" in info
            assert np.isfinite(info["episode_sortino"])
            break
    assert np.isfinite(total_reward)


def test_market_environment_drawdown_threshold_triggers_done():
    prices = np.array([100.0, 99.0, 97.0, 95.0, 93.0], dtype=np.float64)
    features = np.ones((prices.shape[0], 3), dtype=np.float32)
    config = MarketConfig(
        initial_capital=10_000.0,
        max_leverage=1.0,
        transaction_cost=0.0,
        slippage=0.0,
        risk_aversion=0.0,
        max_position_change=1.0,
        min_cash=0.0,
        max_drawdown_threshold=0.02,
    )
    env = MarketEnvironment(prices=prices, features=features, config=config)

    env.reset()
    done = False
    while not done:
        _, _, done, _, info = env.step(np.array([1.0], dtype=np.float32))
        if done:
            assert info["drawdown_triggered"]
            assert info["drawdown"] <= -config.max_drawdown_threshold
            break
