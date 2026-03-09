from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from binanceexp1.sweep_multiasset_selector_robust import (
    SweepConfig,
    build_best_command,
    build_scenario_row,
    build_config_label,
    build_start_state_kwargs,
    compute_selection_score,
)


def test_build_start_state_kwargs_for_flat_start() -> None:
    merged = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True),
            "symbol": ["BTCUSD"],
            "close": [100.0],
        }
    )

    kwargs = build_start_state_kwargs(merged, initial_cash=10_000.0, start_symbol=None)

    assert kwargs["initial_cash"] == pytest.approx(10_000.0)
    assert kwargs["initial_inventory"] == pytest.approx(0.0)
    assert kwargs["initial_symbol"] is None


def test_build_start_state_kwargs_seeds_full_notional_position() -> None:
    merged = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T00:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["BTCUSD", "BTCUSD", "ETHUSD"],
            "close": [200.0, 210.0, 50.0],
        }
    )

    kwargs = build_start_state_kwargs(
        merged,
        initial_cash=10_000.0,
        start_symbol="BTCUSD",
        position_fraction=1.0,
    )

    assert kwargs["initial_cash"] == pytest.approx(0.0)
    assert kwargs["initial_symbol"] == "BTCUSD"
    assert kwargs["initial_open_price"] == pytest.approx(200.0)
    assert kwargs["initial_inventory"] == pytest.approx(50.0)


def test_compute_selection_score_penalizes_negative_worst_case_and_low_activity() -> None:
    summary = {
        "robust_score": 12.0,
        "trade_count_mean": 3.0,
        "return_worst_pct": -0.5,
    }

    score = compute_selection_score(summary, min_trade_count_mean=6.0, require_all_positive=True)

    assert score == pytest.approx(12.0 - (0.75 * 3.0) - 100.0 - 5.0)


def test_build_config_label_includes_fill_model_knobs() -> None:
    cfg = SweepConfig(
        default_intensity=6.0,
        default_offset=0.0,
        min_edge=0.0015,
        risk_weight=0.25,
        edge_mode="high_low",
        max_hold_hours=6,
        decision_lag_bars=2,
        fill_buffer_bps=20.0,
        max_volume_fraction=0.1,
        limit_fill_model="penetration",
        touch_fill_fraction=0.05,
        max_concurrent_positions=1,
    )

    label = build_config_label(cfg)

    assert "fillpenetration" in label
    assert "touch0.05" in label


def test_build_scenario_row_includes_annualized_return_and_trade_fallback() -> None:
    result = SimpleNamespace(
        trades=[{"qty": 1.0}, {"qty": 2.0}],
        open_symbol="ETHUSD",
        final_cash=1234.5,
        final_inventory=6.7,
    )
    metrics = {
        "total_return": 0.0125,
        "annualized_return": 0.45,
        "sortino": 1.2,
        "calmar": 0.8,
        "max_drawdown": -0.031,
        "pnl_smoothness": 0.002,
        "work_steal_count": 3,
    }

    row = build_scenario_row(
        config_name="cfg",
        period="14d",
        start_state="ETHUSD",
        metrics=metrics,
        result=result,
    )

    assert row["config_name"] == "cfg"
    assert row["period"] == "14d"
    assert row["start_state"] == "ETHUSD"
    assert row["return_pct"] == pytest.approx(1.25)
    assert row["annualized_return_pct"] == pytest.approx(45.0)
    assert row["trade_count"] == 2
    assert row["work_steal_count"] == 3
    assert row["open_symbol"] == "ETHUSD"
    assert row["final_cash"] == pytest.approx(1234.5)
    assert row["final_inventory"] == pytest.approx(6.7)


def test_build_best_command_carries_action_cache_root(tmp_path) -> None:
    cfg = SweepConfig(
        default_intensity=6.0,
        default_offset=0.0,
        min_edge=0.0015,
        risk_weight=0.25,
        edge_mode="high_low",
        max_hold_hours=6,
        decision_lag_bars=2,
        fill_buffer_bps=20.0,
        max_volume_fraction=0.1,
        limit_fill_model="binary",
        touch_fill_fraction=0.0,
        max_concurrent_positions=1,
    )
    args = SimpleNamespace(
        symbols=["BTCUSD", "ETHUSD", "SOLUSD"],
        checkpoints="BTCUSD=a,ETHUSD=b,SOLUSD=c",
        horizon=1,
        sequence_length=96,
        forecast_horizons="1,24",
        data_root="trainingdatahourly/crypto",
        forecast_cache_root="binanceneural/forecast_cache",
        action_cache_root="experiments/binance_action_cache",
        validation_days=14.0,
        cache_only=True,
        intensity_map=None,
        offset_map=None,
        max_volume_fraction=None,
        realistic_selection=True,
        work_steal=False,
        work_steal_min_profit_pct=0.0,
        work_steal_min_edge=0.0,
        work_steal_edge_margin=0.0,
    )

    command = build_best_command(best_cfg=cfg, args=args, output_dir=tmp_path / "best_run")

    assert "--action-cache-root" in command
    assert "experiments/binance_action_cache" in command
