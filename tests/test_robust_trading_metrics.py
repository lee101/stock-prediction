from __future__ import annotations

import pytest

from src.robust_trading_metrics import (
    compute_max_drawdown,
    compute_pnl_smoothness,
    compute_pnl_smoothness_from_equity,
    compute_pnl_smoothness_score,
    compute_return_series,
    compute_market_sim_goodness_score,
    summarize_lag_results,
    summarize_scenario_results,
)


def test_compute_return_series_handles_zero_prev() -> None:
    returns = compute_return_series([100.0, 0.0, 50.0, 100.0])
    assert returns.tolist() == pytest.approx([-1.0, 0.0, 1.0])


def test_compute_max_drawdown_basic() -> None:
    assert compute_max_drawdown([100.0, 110.0, 120.0]) == pytest.approx(0.0)
    assert compute_max_drawdown([100.0, 120.0, 90.0, 95.0]) == pytest.approx(0.25)


def test_compute_pnl_smoothness_zero_for_constant_returns() -> None:
    assert compute_pnl_smoothness([0.01, 0.01, 0.01, 0.01]) == pytest.approx(0.0)
    assert compute_pnl_smoothness_from_equity([100.0, 101.0, 102.01, 103.0301]) == pytest.approx(0.0, abs=1e-9)
    assert compute_pnl_smoothness_score(0.0) == pytest.approx(1.0)
    assert compute_pnl_smoothness_score(0.01) < 1.0


def test_compute_market_sim_goodness_score_accepts_trade_count() -> None:
    trade_count_score = compute_market_sim_goodness_score(
        total_return=0.05,
        sortino=1.2,
        max_drawdown=0.03,
        pnl_smoothness=0.002,
        trade_count=12,
        period_count=48,
    )
    trade_rate_score = compute_market_sim_goodness_score(
        total_return=0.05,
        sortino=1.2,
        max_drawdown=0.03,
        pnl_smoothness=0.002,
        trade_rate=12.0 / 48.0,
        period_count=48,
    )

    assert trade_count_score == pytest.approx(trade_rate_score)


def test_summarize_lag_results_outputs_expected_fields() -> None:
    lag_results = [
        {"sortino": 2.0, "return_pct": 5.0, "max_drawdown_pct": 2.0, "pnl_smoothness": 0.002},
        {"sortino": 1.0, "return_pct": 3.0, "max_drawdown_pct": 3.0, "pnl_smoothness": 0.004},
        {"sortino": 0.0, "return_pct": 1.0, "max_drawdown_pct": 5.0, "pnl_smoothness": 0.006},
    ]
    summary = summarize_lag_results(lag_results)

    assert summary["lag_count"] == 3.0
    assert summary["sortino_mean"] == pytest.approx(1.0)
    assert summary["sortino_std"] == pytest.approx(0.81649658, rel=1e-6)
    assert summary["sortino_p10"] == pytest.approx(0.2)
    assert summary["return_mean_pct"] == pytest.approx(3.0)
    assert summary["max_drawdown_mean_pct"] == pytest.approx(10.0 / 3.0)
    assert summary["pnl_smoothness_mean"] == pytest.approx(0.004)
    assert "robust_score" in summary


def test_summarize_lag_results_requires_non_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        summarize_lag_results([])


def test_summarize_lag_results_clips_sortino_outliers() -> None:
    lag_results = [
        {"sortino": 0.0, "return_pct": 0.0, "max_drawdown_pct": 1.0, "pnl_smoothness": 0.001},
        {"sortino": 0.0, "return_pct": 0.0, "max_drawdown_pct": 1.0, "pnl_smoothness": 0.001},
        {"sortino": 100.0, "return_pct": 0.0, "max_drawdown_pct": 1.0, "pnl_smoothness": 0.001},
    ]

    clipped = summarize_lag_results(lag_results, sortino_clip=10.0)
    unclipped = summarize_lag_results(lag_results, sortino_clip=0.0)

    assert clipped["sortino_mean"] == pytest.approx(10.0 / 3.0)
    assert clipped["sortino_std"] < unclipped["sortino_std"]
    assert clipped["sortino_std_raw"] == pytest.approx(unclipped["sortino_std"])


def test_summarize_scenario_results_outputs_expected_fields() -> None:
    scenario_results = [
        {
            "sortino": 2.5,
            "return_pct": 6.0,
            "annualized_return_pct": 24.0,
            "max_drawdown_pct": 2.0,
            "pnl_smoothness": 0.001,
            "trade_count": 12,
        },
        {
            "sortino": 1.0,
            "return_pct": 2.0,
            "annualized_return_pct": 8.0,
            "max_drawdown_pct": 4.0,
            "pnl_smoothness": 0.003,
            "trade_count": 8,
        },
        {
            "sortino": 0.5,
            "return_pct": 1.0,
            "annualized_return_pct": 4.0,
            "max_drawdown_pct": 5.0,
            "pnl_smoothness": 0.004,
            "trade_count": 6,
        },
        {
            "sortino": 0.2,
            "return_pct": 0.5,
            "annualized_return_pct": 2.0,
            "max_drawdown_pct": 6.0,
            "pnl_smoothness": 0.006,
            "trade_count": 4,
        },
    ]

    summary = summarize_scenario_results(scenario_results)

    assert summary["scenario_count"] == 4.0
    assert summary["return_mean_pct"] == pytest.approx(2.375)
    assert summary["return_worst_pct"] == pytest.approx(0.5)
    assert summary["annualized_return_mean_pct"] == pytest.approx(9.5)
    assert summary["annualized_return_worst_pct"] == pytest.approx(2.0)
    assert summary["negative_return_rate"] == pytest.approx(0.0)
    assert summary["max_drawdown_worst_pct"] == pytest.approx(6.0)
    assert summary["trade_count_mean"] == pytest.approx(7.5)
    assert "robust_score" in summary


def test_summarize_scenario_results_penalizes_negative_windows() -> None:
    robust = summarize_scenario_results(
        [
            {"sortino": 2.0, "return_pct": 4.0, "max_drawdown_pct": 2.0, "pnl_smoothness": 0.001, "trade_count": 10},
            {"sortino": 1.5, "return_pct": 3.0, "max_drawdown_pct": 2.5, "pnl_smoothness": 0.0015, "trade_count": 9},
            {"sortino": 1.0, "return_pct": 2.5, "max_drawdown_pct": 3.0, "pnl_smoothness": 0.002, "trade_count": 8},
        ]
    )
    fragile = summarize_scenario_results(
        [
            {"sortino": 2.0, "return_pct": 4.0, "max_drawdown_pct": 2.0, "pnl_smoothness": 0.001, "trade_count": 10},
            {"sortino": 1.5, "return_pct": -1.0, "max_drawdown_pct": 7.0, "pnl_smoothness": 0.004, "trade_count": 9},
            {"sortino": 1.0, "return_pct": 2.5, "max_drawdown_pct": 3.0, "pnl_smoothness": 0.002, "trade_count": 8},
        ]
    )

    assert fragile["negative_return_rate"] == pytest.approx(1.0 / 3.0)
    assert fragile["robust_score"] < robust["robust_score"]
