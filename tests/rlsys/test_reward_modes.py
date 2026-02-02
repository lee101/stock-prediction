import numpy as np

from rlsys.config import MarketConfig
from rlsys.market_environment import MarketEnvironment


def _make_env(config: MarketConfig) -> MarketEnvironment:
    prices = np.array([100.0, 90.0, 95.0], dtype=np.float64)
    features = np.ones((prices.shape[0], 3), dtype=np.float32)
    return MarketEnvironment(prices=prices, features=features, config=config)


def test_risk_adjusted_reward_penalizes_drawdown():
    raw_config = MarketConfig(
        initial_capital=10_000.0,
        max_leverage=1.0,
        transaction_cost=0.0,
        slippage=0.0,
        market_impact=0.0,
        risk_aversion=0.0,
        reward_mode="raw",
    )
    adj_config = MarketConfig(
        initial_capital=10_000.0,
        max_leverage=1.0,
        transaction_cost=0.0,
        slippage=0.0,
        market_impact=0.0,
        risk_aversion=0.0,
        reward_mode="risk_adjusted",
        drawdown_penalty=1.0,
    )
    raw_env = _make_env(raw_config)
    adj_env = _make_env(adj_config)

    raw_env.reset()
    adj_env.reset()
    action = np.array([1.0], dtype=np.float32)
    raw_env.step(action)
    adj_env.step(action)
    _, raw_reward, _, _, _ = raw_env.step(action)
    _, adj_reward, _, _, _ = adj_env.step(action)

    assert adj_reward < raw_reward


def test_sharpe_like_reward_clipped():
    config = MarketConfig(
        initial_capital=10_000.0,
        max_leverage=1.0,
        transaction_cost=0.0,
        slippage=0.0,
        market_impact=0.0,
        risk_aversion=0.0,
        reward_mode="sharpe_like",
        sharpe_clip=0.05,
    )
    env = _make_env(config)
    env.reset()
    action = np.array([1.0], dtype=np.float32)
    _, reward, _, _, _ = env.step(action)
    assert reward <= 0.05
