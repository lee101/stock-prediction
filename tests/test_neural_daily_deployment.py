from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.neural_daily_deployment import (
    annualized_return_pct,
    build_recent_window_start_dates,
    build_scenario_row,
    load_deployment_config,
    parse_optional_float_grid,
    resolve_deployment_settings,
    selection_metric_sort_key,
    selection_metric_value,
    should_promote_candidate,
    summarize_threshold_scenarios,
    write_deployment_config,
)


def test_parse_optional_float_grid_keeps_none_and_dedupes() -> None:
    values = parse_optional_float_grid("none,0.25,0.5,0.25,off", allow_none=True)
    assert values == (None, 0.25, 0.5)


def test_build_recent_window_start_dates_uses_latest_windows() -> None:
    dates = list(pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"))
    starts = build_recent_window_start_dates(
        dates,
        window_days=20,
        max_windows=3,
        stride_days=10,
    )
    assert starts == ["2024-03-01", "2024-03-11", "2024-03-21"]


def test_build_scenario_row_and_annualized_return_pct() -> None:
    row = build_scenario_row(
        start_date="2026-01-13",
        days=60,
        summary={
            "total_return": 0.20,
            "sortino": 1.5,
            "max_drawdown": 0.03,
            "pnl_smoothness": 0.01,
            "trade_count": 12,
            "goodness_score": 7.5,
            "final_equity": 1.2,
            "pnl": 0.2,
        },
        account_fraction=0.15,
        min_trade_amount=0.05,
        risk_threshold=0.5,
        confidence_threshold=0.35,
    )
    assert row["return_pct"] == 20.0
    assert row["max_drawdown_pct"] == 3.0
    assert row["account_fraction"] == 0.15
    assert row["min_trade_amount"] == 0.05
    assert row["risk_threshold"] == 0.5
    assert row["confidence_threshold"] == 0.35
    assert math.isclose(row["annualized_return_pct"], annualized_return_pct(0.20, 60), rel_tol=0, abs_tol=1e-9)


def test_summarize_threshold_scenarios_groups_rows() -> None:
    rows = [
        {
            "start_date": "2026-01-01",
            "account_fraction": 0.15,
            "min_trade_amount": 0.05,
            "risk_threshold": 0.5,
            "confidence_threshold": 0.2,
            "return_pct": 5.0,
            "annualized_return_pct": 35.0,
            "sortino": 1.0,
            "max_drawdown_pct": 2.0,
            "pnl_smoothness": 0.01,
            "trade_count": 8,
            "goodness_score": 3.0,
        },
        {
            "start_date": "2026-01-21",
            "account_fraction": 0.15,
            "min_trade_amount": 0.05,
            "risk_threshold": 0.5,
            "confidence_threshold": 0.2,
            "return_pct": 4.0,
            "annualized_return_pct": 28.0,
            "sortino": 0.8,
            "max_drawdown_pct": 1.5,
            "pnl_smoothness": 0.008,
            "trade_count": 7,
            "goodness_score": 2.8,
        },
        {
            "start_date": "2026-01-01",
            "account_fraction": 0.25,
            "min_trade_amount": 0.0,
            "risk_threshold": 1.0,
            "confidence_threshold": None,
            "return_pct": -1.0,
            "annualized_return_pct": -6.0,
            "sortino": -0.2,
            "max_drawdown_pct": 4.0,
            "pnl_smoothness": 0.02,
            "trade_count": 15,
            "goodness_score": -1.0,
        },
    ]

    summaries = summarize_threshold_scenarios(rows)

    assert len(summaries) == 2
    assert summaries[0]["account_fraction"] == 0.15
    assert summaries[0]["min_trade_amount"] == 0.05
    assert summaries[0]["risk_threshold"] == 0.5
    assert summaries[0]["confidence_threshold"] == 0.2
    assert summaries[0]["scenario_count"] == 2
    assert summaries[0]["goodness_score_mean"] > summaries[1]["goodness_score_mean"]


def test_deployment_config_round_trip_and_resolution(tmp_path: Path) -> None:
    target = tmp_path / "active_latest.json"
    write_deployment_config(
        target,
        checkpoint=tmp_path / "model.pt",
        account_fraction=0.15,
        min_trade_amount=0.05,
        risk_threshold=0.75,
        confidence_threshold=0.3,
        symbols=("spy", "qqq"),
        selection_metric="robust_score",
        selection_value=12.5,
        summary={"robust_score": 12.5},
        metadata={"days": 60},
    )

    payload = load_deployment_config(target)
    resolved = resolve_deployment_settings(
        deployment_payload=payload,
        checkpoint=None,
        symbols=None,
        account_fraction=None,
        min_trade_amount=None,
        risk_threshold=None,
        confidence_threshold=None,
    )

    assert Path(resolved["checkpoint"]).name == "model.pt"
    assert resolved["symbols"] == ("SPY", "QQQ")
    assert resolved["account_fraction"] == 0.15
    assert resolved["min_trade_amount"] == 0.05
    assert resolved["risk_threshold"] == 0.75
    assert resolved["confidence_threshold"] == 0.3


def test_resolve_deployment_settings_prefers_explicit_overrides() -> None:
    resolved = resolve_deployment_settings(
        deployment_payload={
            "checkpoint": "/tmp/base.pt",
            "symbols": ["spy"],
            "account_fraction": 0.2,
            "min_trade_amount": 0.04,
            "risk_threshold": 0.25,
            "confidence_threshold": 0.15,
        },
        checkpoint="/tmp/override.pt",
        symbols=("qqq", "iwm"),
        account_fraction=0.1,
        min_trade_amount=0.06,
        risk_threshold=0.5,
        confidence_threshold=0.4,
    )

    assert resolved["checkpoint"] == "/tmp/override.pt"
    assert resolved["symbols"] == ("QQQ", "IWM")
    assert resolved["account_fraction"] == 0.1
    assert resolved["min_trade_amount"] == 0.06
    assert resolved["risk_threshold"] == 0.5
    assert resolved["confidence_threshold"] == 0.4


def test_should_promote_candidate_applies_gates() -> None:
    candidate = {"robust_score": 10.0, "return_p25_pct": 2.0, "sortino_p25": 0.5}
    baseline = {"robust_score": 9.5, "return_p25_pct": 1.0, "sortino_p25": 0.2}

    assert should_promote_candidate(
        candidate,
        baseline,
        min_robust_improvement=0.25,
        min_return_p25_pct=1.5,
        min_sortino_p25=0.4,
    )
    assert not should_promote_candidate(
        candidate,
        baseline,
        min_robust_improvement=1.0,
        min_return_p25_pct=1.5,
        min_sortino_p25=0.4,
    )


def test_selection_metric_helpers_handle_lower_is_better_metrics() -> None:
    summary_a = {"sortino_p25": 1.0, "robust_score": 10.0, "pnl_smoothness_mean": 0.02}
    summary_b = {"sortino_p25": 0.8, "robust_score": 9.0, "pnl_smoothness_mean": 0.01}

    assert selection_metric_value(summary_a, "sortino_p25") == 1.0
    assert selection_metric_value(summary_b, "pnl_smoothness_mean") == -0.01
    assert selection_metric_sort_key(summary_a, "sortino_p25") > selection_metric_sort_key(summary_b, "sortino_p25")
    assert selection_metric_sort_key(summary_b, "pnl_smoothness_mean") > selection_metric_sort_key(
        summary_a,
        "pnl_smoothness_mean",
    )
