from src.tradinglib.metrics import PnlMetrics
from src.tradinglib.targets import PnlTargets, evaluate_targets


def test_evaluate_targets_scores():
    metrics = PnlMetrics(
        total_return=0.2,
        annualized_return=0.3,
        sharpe=1.4,
        sortino=1.8,
        max_drawdown=-0.1,
        calmar=2.0,
        profit_factor=1.5,
        avg_return=0.001,
        volatility=0.02,
    )
    targets = PnlTargets(
        sharpe=1.0,
        sortino=1.5,
        max_drawdown=-0.2,
        profit_factor=1.2,
        annualized_return=0.15,
    )
    result = evaluate_targets(metrics, targets)
    assert result.score == 1.0
    assert all(result.passed.values())


def test_evaluate_targets_partial():
    metrics = PnlMetrics(
        total_return=0.0,
        annualized_return=0.0,
        sharpe=0.5,
        sortino=0.4,
        max_drawdown=-0.5,
        calmar=0.0,
        profit_factor=0.9,
        avg_return=0.0,
        volatility=0.0,
    )
    targets = PnlTargets()
    result = evaluate_targets(metrics, targets)
    assert 0.0 <= result.score < 1.0
    assert result.passed["max_drawdown"] is False
