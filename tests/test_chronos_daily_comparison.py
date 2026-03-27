from __future__ import annotations

import pandas as pd
import pytest

from src.chronos_daily_comparison import (
    DailyOHLC,
    aggregate_hourly_prediction_frame_to_daily,
    aggregate_intraday_ohlc,
    intersect_symbol_days,
    ohlc_error_percent_by_column,
    ohlc_mape_percent,
)


def _hourly_frame(start: str, closes: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
        }
    )


def test_aggregate_intraday_ohlc_uses_first_open_last_close_and_extrema() -> None:
    frame = _hourly_frame("2026-01-01 00:00:00+00:00", [100.0, 102.0, 101.0, 105.0])
    out = aggregate_intraday_ohlc(frame)
    assert out.timestamp == pd.Timestamp("2026-01-01 00:00:00+00:00")
    assert out.open == pytest.approx(100.0)
    assert out.high == pytest.approx(106.0)
    assert out.low == pytest.approx(99.0)
    assert out.close == pytest.approx(105.0)


def test_aggregate_hourly_prediction_frame_to_daily_accepts_indexed_frame() -> None:
    frame = _hourly_frame("2026-01-01 00:00:00+00:00", [10.0, 12.0, 14.0]).set_index("timestamp")
    out = aggregate_hourly_prediction_frame_to_daily(frame)
    assert out.timestamp == pd.Timestamp("2026-01-01 00:00:00+00:00")
    assert out.open == pytest.approx(10.0)
    assert out.high == pytest.approx(15.0)
    assert out.low == pytest.approx(9.0)
    assert out.close == pytest.approx(14.0)


def test_ohlc_mape_percent_reports_average_percentage_error() -> None:
    predicted = DailyOHLC(
        timestamp=pd.Timestamp("2026-01-01 00:00:00+00:00"),
        open=110.0,
        high=120.0,
        low=90.0,
        close=105.0,
    )
    actual = DailyOHLC(
        timestamp=pd.Timestamp("2026-01-01 00:00:00+00:00"),
        open=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
    )
    assert ohlc_mape_percent(predicted, actual) == pytest.approx(11.25)
    by_column = ohlc_error_percent_by_column(predicted, actual)
    assert by_column == pytest.approx(
        {
            "open": 10.0,
            "high": 20.0,
            "low": 10.0,
            "close": 5.0,
        }
    )


def test_intersect_symbol_days_requires_common_daily_and_complete_hourly_days() -> None:
    daily_a = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 00:00:00+00:00", "2026-01-02 00:00:00+00:00", "2026-01-03 00:00:00+00:00"],
                utc=True,
            ),
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
        }
    )
    daily_b = daily_a.copy()
    hourly_a = pd.concat(
        [
            _hourly_frame("2026-01-01 00:00:00+00:00", [1.0] * 24),
            _hourly_frame("2026-01-02 00:00:00+00:00", [1.0] * 23),
            _hourly_frame("2026-01-03 00:00:00+00:00", [1.0] * 24),
        ],
        ignore_index=True,
    )
    hourly_b = pd.concat(
        [
            _hourly_frame("2026-01-01 00:00:00+00:00", [1.0] * 24),
            _hourly_frame("2026-01-02 00:00:00+00:00", [1.0] * 24),
        ],
        ignore_index=True,
    )

    out = intersect_symbol_days(
        daily_frames={"AAA": daily_a, "BBB": daily_b},
        hourly_frames={"AAA": hourly_a, "BBB": hourly_b},
        symbols=["AAA", "BBB"],
    )
    assert out == [
        pd.Timestamp("2026-01-01 00:00:00+00:00"),
    ]
