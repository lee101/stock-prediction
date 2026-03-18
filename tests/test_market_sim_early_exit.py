from __future__ import annotations

from src.market_sim_early_exit import (
    evaluate_baseline_comparability_early_exit,
    evaluate_drawdown_vs_profit_early_exit,
)


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


def test_baseline_comparability_early_exit_triggers_for_clearly_weaker_path() -> None:
    decision = evaluate_baseline_comparability_early_exit(
        [100.0, 92.0, 88.0, 84.0],
        total_steps=4,
        min_total_steps=4,
        label="unit",
        periods_per_year=24.0 * 365.0,
        baseline_total_return=0.05,
        baseline_sortino=1.0,
        baseline_max_drawdown=0.02,
        stage1_progress=0.25,
        stage2_progress=0.50,
        stage3_progress=0.75,
        return_tolerance=0.01,
        sortino_tolerance=0.25,
        max_drawdown_tolerance=0.01,
    )
    assert decision.should_stop
    assert "baseline gate" in decision.reason
    assert decision.max_drawdown > 0.03
