from __future__ import annotations

from src.market_sim_early_exit import evaluate_drawdown_vs_profit_early_exit


def test_early_exit_waits_until_halfway() -> None:
    decision = evaluate_drawdown_vs_profit_early_exit(
        [100.0, 110.0, 105.0],
        total_steps=10,
        label="unit",
    )
    assert not decision.should_stop


def test_early_exit_ignores_tiny_windows() -> None:
    decision = evaluate_drawdown_vs_profit_early_exit(
        [100.0, 80.0, 70.0],
        total_steps=3,
        label="unit",
    )
    assert not decision.should_stop


def test_early_exit_triggers_when_drawdown_exceeds_profit_after_halfway() -> None:
    decision = evaluate_drawdown_vs_profit_early_exit(
        [100.0, 140.0, 120.0, 105.0],
        total_steps=4,
        min_total_steps=4,
        label="unit",
    )
    assert decision.should_stop
    assert "max drawdown" in decision.reason
    assert decision.max_drawdown > decision.total_return
