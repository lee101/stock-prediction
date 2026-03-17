from __future__ import annotations

from src.models.chronos2_postprocessing import repair_forecast_ohlc


def test_repair_forecast_ohlc_orders_quantiles_and_wraps_high_low() -> None:
    repaired = repair_forecast_ohlc(
        last_close=100.0,
        close_p50=101.0,
        close_p10=103.0,
        close_p90=99.0,
        high_p50=98.0,
        low_p50=104.0,
    )

    assert repaired.close_p10 == 99.0
    assert repaired.close_p50 == 101.0
    assert repaired.close_p90 == 103.0
    assert repaired.low_p50 <= repaired.close_p50 <= repaired.high_p50
    assert repaired.forecast_move_pct == 0.01
    assert repaired.forecast_volatility_pct == 0.04
