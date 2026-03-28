from __future__ import annotations

import pandas as pd
import pytest

from unified_hourly_experiment.xgb_chronos_baseline import (
    SearchConfig,
    blend_forecast_price,
    build_action_row,
    build_labeled_rows,
)


def _cfg(**overrides) -> SearchConfig:
    payload = {
        "label_horizon_hours": 4,
        "label_basis": "reference_close",
        "residual_scale": 1.0,
        "risk_penalty": 1.0,
        "entry_alpha": 0.5,
        "exit_alpha": 0.75,
        "edge_threshold": 0.002,
        "edge_to_full_size": 0.02,
        "max_positions": 5,
        "max_hold_hours": 4,
        "close_at_eod": False,
        "market_order_entry": False,
        "entry_selection_mode": "edge_rank",
        "entry_allocator_mode": "legacy",
        "entry_allocator_edge_power": 2.0,
    }
    payload.update(overrides)
    return SearchConfig(**payload)


def test_build_action_row_prefers_long_when_long_edge_wins() -> None:
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-03-28T14:00:00Z"),
            "symbol": "NET",
            "reference_close": 100.0,
            "pred_high_ret_xgb": 0.03,
            "pred_low_ret_xgb": -0.005,
            "pred_close_ret_xgb": 0.015,
        }
    )

    action = build_action_row(row, _cfg())

    assert action["buy_amount"] > 0.0
    assert action["sell_amount"] == 0.0
    assert action["sell_price"] > action["buy_price"]
    assert action["xgb_long_edge"] > action["xgb_short_edge"]


def test_build_action_row_prefers_short_for_short_only_symbol() -> None:
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-03-28T14:00:00Z"),
            "symbol": "DBX",
            "reference_close": 50.0,
            "pred_high_ret_xgb": 0.01,
            "pred_low_ret_xgb": -0.04,
            "pred_close_ret_xgb": -0.02,
        }
    )

    action = build_action_row(row, _cfg())

    assert action["sell_amount"] > 0.0
    assert action["buy_amount"] == 0.0
    assert action["sell_price"] > action["buy_price"]
    assert action["xgb_short_edge"] > action["xgb_long_edge"]


def test_build_action_row_skips_when_edge_below_threshold() -> None:
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2026-03-28T14:00:00Z"),
            "symbol": "NET",
            "reference_close": 100.0,
            "pred_high_ret_xgb": 0.004,
            "pred_low_ret_xgb": -0.003,
            "pred_close_ret_xgb": 0.001,
        }
    )

    action = build_action_row(row, _cfg(edge_threshold=0.01))

    assert action["buy_amount"] == 0.0
    assert action["sell_amount"] == 0.0
    assert action["trade_amount"] == 0.0


def test_blend_forecast_price_interpolates_between_horizons() -> None:
    row = pd.Series(
        {
            "reference_close": 100.0,
            "predicted_close_p50_h1": 101.0,
            "predicted_close_p50_h24": 124.0,
        }
    )

    blended = blend_forecast_price(
        row,
        kind="close",
        target_horizon_hours=4,
        forecast_horizons=(1, 24),
    )

    assert 101.0 < blended < 124.0


def test_build_labeled_rows_uses_reference_close_basis() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-28T10:00:00Z",
                    "2026-03-28T11:00:00Z",
                    "2026-03-28T12:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["NET", "NET", "NET"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 102.0, 103.0],
            "reference_close": [100.0, 102.0, 103.0],
            "predicted_high_p50_h1": [102.0, 104.0, 105.0],
            "predicted_low_p50_h1": [99.0, 101.0, 102.0],
            "predicted_close_p50_h1": [101.0, 103.0, 104.0],
            "predicted_high_p50_h24": [110.0, 111.0, 112.0],
            "predicted_low_p50_h24": [95.0, 96.0, 97.0],
            "predicted_close_p50_h24": [108.0, 109.0, 110.0],
        }
    )

    labeled = build_labeled_rows(
        frame,
        label_horizon_hours=1,
        forecast_horizons=(1, 24),
        label_basis="reference_close",
    )

    assert labeled.loc[0, "target_close_ret"] == pytest.approx(0.02)
    assert labeled.loc[0, "prior_close_ret"] == pytest.approx(0.01)


def test_build_labeled_rows_uses_next_open_basis() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-28T10:00:00Z",
                    "2026-03-28T11:00:00Z",
                    "2026-03-28T12:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["NET", "NET", "NET"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 102.0, 103.0],
            "reference_close": [100.0, 102.0, 103.0],
            "predicted_high_p50_h1": [102.0, 104.0, 105.0],
            "predicted_low_p50_h1": [99.0, 101.0, 102.0],
            "predicted_close_p50_h1": [101.0, 103.0, 104.0],
            "predicted_high_p50_h24": [110.0, 111.0, 112.0],
            "predicted_low_p50_h24": [95.0, 96.0, 97.0],
            "predicted_close_p50_h24": [108.0, 109.0, 110.0],
        }
    )

    labeled = build_labeled_rows(
        frame,
        label_horizon_hours=1,
        forecast_horizons=(1, 24),
        label_basis="next_open",
    )

    assert labeled.loc[0, "target_close_ret"] == pytest.approx((102.0 / 101.0) - 1.0)
    assert labeled.loc[0, "prior_close_ret"] == pytest.approx(0.0)
