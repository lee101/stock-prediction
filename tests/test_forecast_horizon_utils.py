from __future__ import annotations

from src.forecast_horizon_utils import infer_feature_horizons, resolve_required_forecast_horizons


def test_infer_feature_horizons_reads_forecast_and_prediction_columns() -> None:
    feature_columns = [
        "return_1h",
        "return_24h",
        "chronos_close_delta_h1",
        "forecast_delta_spread_h1_h24",
        "chronos_high_delta_h6",
        "chronos_low_delta_h6",
        "chronos_close_delta_h12",
        "predicted_close_p50_h48",
    ]

    assert infer_feature_horizons(feature_columns) == (1, 6, 12, 24, 48)


def test_resolve_required_forecast_horizons_unions_requested_and_inferred() -> None:
    resolved = resolve_required_forecast_horizons(
        (1,),
        feature_columns=["chronos_close_delta_h1", "chronos_high_delta_h12"],
    )

    assert resolved == (1, 12)


def test_resolve_required_forecast_horizons_uses_fallback_when_no_feature_horizons_present() -> None:
    resolved = resolve_required_forecast_horizons(
        (),
        feature_columns=["return_1h", "return_24h"],
        fallback_horizons=(6,),
    )

    assert resolved == (6,)
