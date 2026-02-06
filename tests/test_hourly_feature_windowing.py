from __future__ import annotations

import pandas as pd

from binanceexp1.config import DatasetConfig
from binanceexp1.data import build_feature_frame
from src.hourly_feature_windowing import (
    apply_feature_max_window_hours,
    filter_feature_columns_max_window,
)


def test_filter_feature_columns_max_window_removes_long_hour_features() -> None:
    cols = [
        "volatility_24h",
        "volatility_168h",
        "return_4h",
        "return_240h",
        "chronos_close_delta_h4",
        "chronos_close_delta_h240",
        "vol_regime_24_72",
        "vol_regime_24_168",
        "hour_sin",
    ]
    kept = filter_feature_columns_max_window(cols, max_window_hours=72)
    assert "volatility_24h" in kept
    assert "return_4h" in kept
    assert "chronos_close_delta_h4" in kept
    assert "vol_regime_24_72" in kept
    assert "hour_sin" in kept
    assert "volatility_168h" not in kept
    assert "return_240h" not in kept
    assert "chronos_close_delta_h240" not in kept
    assert "vol_regime_24_168" not in kept


def test_apply_feature_max_window_hours_caps_windows_and_rebuilds_feature_columns() -> None:
    cfg = DatasetConfig(
        moving_average_windows=(24, 72, 168),
        ema_windows=(24, 72, 168),
        atr_windows=(24, 72, 168),
        trend_windows=(24, 72, 168),
        drawdown_windows=(24, 72, 168),
        vol_regime_short=24,
        vol_regime_long=168,
        volume_z_window=168,
        volume_shock_window=24,
        rsi_window=14,
        forecast_horizons=(1, 4, 24),
    )
    capped = apply_feature_max_window_hours(cfg, max_window_hours=72)
    assert capped.moving_average_windows == (24, 72)
    assert capped.ema_windows == (24, 72)
    assert capped.atr_windows == (24, 72)
    assert capped.trend_windows == (24, 72)
    assert capped.drawdown_windows == (24, 72)
    assert capped.vol_regime_short == 24
    assert capped.vol_regime_long == 72
    assert capped.volume_z_window == 72
    assert capped.volume_shock_window == 24
    assert capped.rsi_window == 14

    assert capped.feature_columns is not None
    cols = list(capped.feature_columns)
    assert "volatility_168h" not in cols
    assert "vol_regime_24_168" not in cols
    assert "vol_regime_24_72" in cols
    assert "ma_delta_72h" in cols
    assert "ema_delta_72h" in cols
    assert "atr_pct_72h" in cols


def test_build_feature_frame_handles_zero_volume_without_infinite_or_nan_gaps() -> None:
    ts = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "TEST",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 0.0,
        }
    )
    cfg = DatasetConfig(
        symbol="TEST",
        forecast_horizons=(1,),
        moving_average_windows=(4,),
        ema_windows=(4,),
        atr_windows=(4,),
        trend_windows=(4,),
        drawdown_windows=(4,),
        vol_regime_short=2,
        vol_regime_long=4,
        volume_z_window=4,
        volume_shock_window=4,
        rsi_window=4,
    )
    enriched = build_feature_frame(frame, cfg)

    assert enriched["volume_change_1h"].isna().sum() == 0
    assert enriched["volume_change_1h"].abs().max() == 0.0

    # Rolling windows produce NaNs for the warmup period, but once available they should be finite (0.0 here).
    assert float(enriched["volume_z"].iloc[-1]) == 0.0
    assert float(enriched["volume_shock_4h"].iloc[-1]) == 0.0

