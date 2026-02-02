import numpy as np

from src.metrics_utils import annualized_sharpe, annualized_sortino, compute_step_returns


def test_compute_step_returns_basic():
    series = [100.0, 110.0, 99.0]
    returns = compute_step_returns(series)
    expected = np.array([0.10, (99.0 - 110.0) / 110.0])
    assert np.allclose(returns, expected)


def test_annualized_sharpe_zero_variance():
    returns = [0.0, 0.0, 0.0]
    assert annualized_sharpe(returns) == 0.0


def test_annualized_sortino_all_positive_matches_sharpe():
    returns = [0.01, 0.02, 0.03, 0.015]
    sharpe = annualized_sharpe(returns)
    sortino = annualized_sortino(returns)
    assert np.isclose(sortino, sharpe)


def test_annualized_sortino_with_downside():
    returns = np.array([0.01, -0.02, 0.03, -0.01])
    mean = returns.mean()
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1)
    expected = mean / downside_std * np.sqrt(252.0)
    assert np.isclose(annualized_sortino(returns), expected)
