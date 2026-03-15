from __future__ import annotations

import pytest

from pufferlib_market.autoresearch_rl import (
    select_rank_score,
    summarize_holdout_payload,
    summarize_market_validation_payload,
)
from src.robust_trading_metrics import summarize_scenario_results


def test_summarize_holdout_payload_computes_robust_metrics() -> None:
    payload = {
        "summary": {
            "median_total_return": 0.04,
            "p10_total_return": -0.03,
            "median_sortino": 1.1,
            "p90_max_drawdown": 0.09,
        },
        "windows": [
            {
                "total_return": 0.10,
                "annualized_return": 0.80,
                "sortino": 1.8,
                "max_drawdown": 0.04,
                "num_trades": 4,
            },
            {
                "total_return": 0.03,
                "annualized_return": 0.20,
                "sortino": 0.9,
                "max_drawdown": 0.02,
                "num_trades": 3,
            },
            {
                "total_return": -0.06,
                "annualized_return": -0.50,
                "sortino": -0.4,
                "max_drawdown": 0.12,
                "num_trades": 2,
            },
        ],
    }

    summary = summarize_holdout_payload(payload)
    expected = summarize_scenario_results(
        [
            {
                "return_pct": 10.0,
                "annualized_return_pct": 80.0,
                "sortino": 1.8,
                "max_drawdown_pct": 4.0,
                "pnl_smoothness": 0.0,
                "trade_count": 4.0,
            },
            {
                "return_pct": 3.0,
                "annualized_return_pct": 20.0,
                "sortino": 0.9,
                "max_drawdown_pct": 2.0,
                "pnl_smoothness": 0.0,
                "trade_count": 3.0,
            },
            {
                "return_pct": -6.0,
                "annualized_return_pct": -50.0,
                "sortino": -0.4,
                "max_drawdown_pct": 12.0,
                "pnl_smoothness": 0.0,
                "trade_count": 2.0,
            },
        ]
    )

    assert summary["holdout_robust_score"] == pytest.approx(expected["robust_score"])
    assert summary["holdout_return_p25_pct"] == pytest.approx(expected["return_p25_pct"])
    assert summary["holdout_return_worst_pct"] == pytest.approx(expected["return_worst_pct"])
    assert summary["holdout_max_drawdown_worst_pct"] == pytest.approx(expected["max_drawdown_worst_pct"])
    assert summary["holdout_negative_return_rate"] == pytest.approx(expected["negative_return_rate"])
    assert summary["holdout_median_return_pct"] == pytest.approx(4.0)
    assert summary["holdout_p10_return_pct"] == pytest.approx(-3.0)
    assert summary["holdout_median_sortino"] == pytest.approx(1.1)
    assert summary["holdout_p90_max_drawdown_pct"] == pytest.approx(9.0)


def test_summarize_market_validation_payload_extracts_first_result() -> None:
    payload = [
        {
            "return_pct": 1.25,
            "sortino": 0.8,
            "max_drawdown_pct": 2.5,
            "trade_count": 7,
            "goodness_score": 3.6,
        }
    ]

    summary = summarize_market_validation_payload(payload)

    assert summary == {
        "market_return_pct": 1.25,
        "market_sortino": 0.8,
        "market_max_drawdown_pct": 2.5,
        "market_trade_count": 7.0,
        "market_goodness_score": 3.6,
    }


def test_select_rank_score_uses_expected_fallback_order() -> None:
    metrics = {
        "val_return": 0.04,
        "holdout_robust_score": 1.5,
        "market_goodness_score": 2.75,
    }

    assert select_rank_score(metrics, rank_metric="auto") == ("market_goodness_score", 2.75)
    assert select_rank_score(metrics, rank_metric="holdout_robust_score") == ("holdout_robust_score", 1.5)
    assert select_rank_score(metrics, rank_metric="val_return") == ("val_return", 0.04)
    assert select_rank_score({"val_return": 0.01}, rank_metric="auto") == ("val_return", 0.01)
    assert select_rank_score({}, rank_metric="auto") == ("none", None)
