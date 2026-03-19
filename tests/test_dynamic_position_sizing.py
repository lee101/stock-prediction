from __future__ import annotations
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pandas as pd
from dynamic_position_sizing import compute_position_scale, compute_atr_pct, NO_FORECAST_FALLBACK_SCALE


def test_perfect_forecast_high_scale():
    scale = compute_position_scale(
        forecast_p10=99.5, forecast_p50=102.0, forecast_p90=100.5,
        current_price=100.0, atr_pct=0.01, base_mae_pct=0.0055,
    )
    assert scale > 0.8


def test_uncertain_forecast_low_scale():
    scale = compute_position_scale(
        forecast_p10=90.0, forecast_p50=100.1, forecast_p90=110.0,
        current_price=100.0, atr_pct=0.01, base_mae_pct=0.0055,
    )
    assert scale < 0.15


def test_high_atr_reduces_scale():
    normal = compute_position_scale(
        forecast_p10=99.0, forecast_p50=101.0, forecast_p90=101.5,
        current_price=100.0, atr_pct=0.005, base_mae_pct=0.0055,
    )
    high_atr = compute_position_scale(
        forecast_p10=99.0, forecast_p50=101.0, forecast_p90=101.5,
        current_price=100.0, atr_pct=0.012, base_mae_pct=0.0055,
    )
    assert high_atr == pytest.approx(normal * 0.5, abs=1e-9)


def test_missing_forecast_returns_zero():
    scale = compute_position_scale(
        forecast_p10=0.0, forecast_p50=0.0, forecast_p90=0.0,
        current_price=100.0, atr_pct=0.02, base_mae_pct=0.0055,
    )
    assert scale == 0.0


def test_zero_atr():
    scale = compute_position_scale(
        forecast_p10=99.0, forecast_p50=101.0, forecast_p90=101.5,
        current_price=100.0, atr_pct=0.0, base_mae_pct=0.0055,
    )
    assert 0.0 <= scale <= 1.0


def test_negative_price_returns_zero():
    scale = compute_position_scale(
        forecast_p10=99.0, forecast_p50=101.0, forecast_p90=101.5,
        current_price=-1.0, atr_pct=0.02, base_mae_pct=0.0055,
    )
    assert scale == 0.0


def test_zero_base_mae_returns_zero():
    scale = compute_position_scale(
        forecast_p10=99.0, forecast_p50=101.0, forecast_p90=101.5,
        current_price=100.0, atr_pct=0.02, base_mae_pct=0.0,
    )
    assert scale == 0.0


def test_scale_clipped_to_one():
    scale = compute_position_scale(
        forecast_p10=99.99, forecast_p50=110.0, forecast_p90=100.01,
        current_price=100.0, atr_pct=0.001, base_mae_pct=0.0055,
    )
    assert scale <= 1.0


def test_scale_never_negative():
    scale = compute_position_scale(
        forecast_p10=80.0, forecast_p50=100.0, forecast_p90=120.0,
        current_price=100.0, atr_pct=0.05, base_mae_pct=0.0055,
    )
    assert scale >= 0.0


def test_moderate_confidence():
    scale = compute_position_scale(
        forecast_p10=99.5, forecast_p50=101.0, forecast_p90=101.5,
        current_price=100.0, atr_pct=0.005, base_mae_pct=0.0055,
    )
    assert 0.1 < scale < 0.9


def test_fallback_constant():
    assert NO_FORECAST_FALLBACK_SCALE == 0.5


def test_compute_atr_pct_basic():
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
        "high": [105, 110, 108],
        "low": [95, 100, 102],
        "close": [100, 105, 106],
    })
    atr = compute_atr_pct(bars, pd.Timestamp("2026-01-03", tz="UTC"))
    assert 0.0 < atr < 0.2


def test_compute_atr_pct_insufficient_data():
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01"], utc=True),
        "high": [105], "low": [95], "close": [100],
    })
    assert compute_atr_pct(bars, pd.Timestamp("2026-01-01", tz="UTC")) == 0.0


def test_compute_atr_pct_empty():
    bars = pd.DataFrame(columns=["timestamp", "high", "low", "close"])
    assert compute_atr_pct(bars, pd.Timestamp("2026-01-01", tz="UTC")) == 0.0
