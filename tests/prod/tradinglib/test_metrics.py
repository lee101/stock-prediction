import numpy as np

from src.tradinglib.metrics import (
    annualized_return_from_returns,
    drawdown_curve,
    max_drawdown,
    pnl_metrics,
    profit_factor,
    total_return,
    trade_stats,
)


def test_total_return_handles_short_series():
    assert total_return([100.0]) == 0.0
    assert total_return([]) == 0.0


def test_drawdown_curve_and_max_drawdown():
    equity = [100.0, 110.0, 105.0, 120.0, 90.0]
    dd = drawdown_curve(equity)
    assert dd.shape[0] == len(equity)
    assert np.isclose(max_drawdown(equity), dd.min())
    assert max_drawdown([100.0, 101.0, 102.0]) == 0.0


def test_profit_factor_edge_cases():
    assert profit_factor([]) == 0.0
    assert profit_factor([0.1, 0.2]) == float("inf")
    assert profit_factor([-0.1, -0.2]) == 0.0


def test_trade_stats_basic():
    stats = trade_stats([0.1, -0.05, 0.2])
    assert stats.num_trades == 3
    assert np.isclose(stats.win_rate, 2 / 3)
    assert stats.avg_win > 0
    assert stats.avg_loss < 0
    assert stats.expectancy > 0


def test_pnl_metrics_summary():
    equity = [100.0, 101.0, 99.0, 103.0]
    metrics = pnl_metrics(equity_curve=equity, periods_per_year=252)
    assert metrics.total_return != 0.0
    assert metrics.volatility >= 0.0
    assert metrics.max_drawdown <= 0.0
    assert np.isfinite(metrics.sharpe)
    assert np.isfinite(metrics.sortino)


def test_annualized_return_from_returns_compound():
    returns = [0.01] * 10
    ann = annualized_return_from_returns(returns, periods_per_year=10)
    assert ann > 0
