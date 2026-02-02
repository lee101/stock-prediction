import numpy as np

from src.tradinglib.benchmarks import buy_and_hold_returns, oracle_long_flat_returns, oracle_long_short_returns


def test_buy_and_hold_returns():
    prices = [100.0, 110.0, 99.0]
    returns = buy_and_hold_returns(prices)
    assert np.allclose(returns, [0.1, -0.1])


def test_oracle_long_flat_returns():
    returns = [0.1, -0.05, 0.02]
    oracle = oracle_long_flat_returns(returns)
    assert np.allclose(oracle, [0.1, 0.0, 0.02])


def test_oracle_long_flat_with_costs():
    returns = [0.02, -0.01, 0.03]
    oracle = oracle_long_flat_returns(returns, trade_cost_bps=10)
    assert oracle[0] < 0.02  # cost applied on entry


def test_oracle_long_short_returns():
    returns = [0.1, -0.05, 0.0]
    oracle = oracle_long_short_returns(returns)
    assert np.allclose(oracle, [0.1, 0.05, 0.0])
