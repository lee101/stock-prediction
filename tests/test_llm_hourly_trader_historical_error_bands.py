from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_hourly_trader.gemini_wrapper import build_prompt
from llm_hourly_trader.historical_error_bands import HistoricalForecastErrorEstimator


def test_historical_error_estimator_uses_only_resolved_targets() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T02:00:00Z",
                    "2026-01-01T03:00:00Z",
                ],
                utc=True,
            ),
            "close": [100.0, 100.0, 100.0],
        }
    )
    forecasts = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T02:00:00Z",
                    "2026-01-01T03:00:00Z",
                ],
                utc=True,
            ),
            "target_timestamp": pd.to_datetime(
                [
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T02:00:00Z",
                    "2026-01-01T03:00:00Z",
                ],
                utc=True,
            ),
            "predicted_close_p50": [101.0, 98.0, 104.0],
        }
    )

    estimator = HistoricalForecastErrorEstimator.from_frames(
        bars=bars,
        forecasts=forecasts,
        horizon_hours=1,
    )
    band = estimator.band_at(pd.Timestamp("2026-01-01T02:30:00Z"), lookback_days=30, min_samples=1)

    assert band is not None
    assert band.samples == 2
    assert band.mae_pct == pytest.approx(1.5, abs=1e-9)


def test_historical_error_estimator_derives_missing_target_timestamp() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-02T00:00:00Z"], utc=True),
            "close": [104.0],
        }
    )
    forecasts = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T01:00:00Z"], utc=True),
            "predicted_close_p50": [100.0],
        }
    )

    estimator = HistoricalForecastErrorEstimator.from_frames(
        bars=bars,
        forecasts=forecasts,
        horizon_hours=24,
    )
    band = estimator.band_at(pd.Timestamp("2026-01-02T00:00:00Z"), lookback_days=30, min_samples=1)

    assert band is not None
    assert band.samples == 1
    assert band.mae_pct == pytest.approx((4.0 / 104.0) * 100.0, rel=1e-9)


def test_build_prompt_freeform_includes_open_ended_language() -> None:
    prompt = build_prompt(
        symbol="BTCUSD",
        history_rows=[
            {"timestamp": "2026-03-10T00:00", "open": 69000, "high": 69500, "low": 68800, "close": 69200, "volume": 100},
            {"timestamp": "2026-03-10T01:00", "open": 69200, "high": 69800, "low": 69100, "close": 69700, "volume": 120},
            {"timestamp": "2026-03-10T02:00", "open": 69700, "high": 70100, "low": 69600, "close": 70000, "volume": 110},
        ],
        forecast_1h={
            "predicted_close_p50": 70100.0,
            "predicted_close_p10": 69750.0,
            "predicted_close_p90": 70400.0,
            "predicted_high_p50": 70300.0,
            "predicted_low_p50": 69900.0,
        },
        forecast_24h={
            "predicted_close_p50": 71000.0,
            "predicted_close_p10": 69800.0,
            "predicted_close_p90": 72100.0,
            "predicted_high_p50": 71600.0,
            "predicted_low_p50": 70400.0,
        },
        current_position="flat",
        cash=2000.0,
        equity=2000.0,
        allowed_directions=["long"],
        asset_class="crypto",
        maker_fee=0.001,
        variant="freeform",
    )

    assert "Use the data however you think is best." in prompt
    assert "STRICT RISK RULES" not in prompt


def test_build_prompt_mae_bands_uses_historical_band_wording() -> None:
    prompt = build_prompt(
        symbol="BTCUSD",
        history_rows=[
            {"timestamp": "2026-03-10T00:00", "open": 69000, "high": 69500, "low": 68800, "close": 69200, "volume": 100},
            {"timestamp": "2026-03-10T01:00", "open": 69200, "high": 69800, "low": 69100, "close": 69700, "volume": 120},
            {"timestamp": "2026-03-10T02:00", "open": 69700, "high": 70100, "low": 69600, "close": 70000, "volume": 110},
        ],
        forecast_1h={
            "predicted_close_p50": 70100.0,
            "predicted_close_p10": 69750.0,
            "predicted_close_p90": 70400.0,
            "predicted_high_p50": 70300.0,
            "predicted_low_p50": 69900.0,
        },
        forecast_24h={
            "predicted_close_p50": 71000.0,
            "predicted_close_p10": 69800.0,
            "predicted_close_p90": 72100.0,
            "predicted_high_p50": 71600.0,
            "predicted_low_p50": 70400.0,
        },
        current_position="flat",
        cash=2000.0,
        equity=2000.0,
        allowed_directions=["long"],
        asset_class="crypto",
        maker_fee=0.001,
        variant="mae_bands",
        forecast_error_1h={"mae_pct": 0.5, "samples": 72},
        forecast_error_24h={"mae_pct": 1.75, "samples": 240},
    )

    assert "Chronos2 Forecasts With Historical Error Bands" in prompt
    assert "historical MAE band" in prompt
    assert "MAE=0.50%" in prompt
    assert "n=72" in prompt
