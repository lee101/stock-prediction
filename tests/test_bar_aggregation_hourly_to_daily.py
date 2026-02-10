from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.bar_aggregation import hourly_to_daily_ohlcv


def _hourly_rows(start: datetime, count: int) -> list[datetime]:
    return [start.replace(tzinfo=timezone.utc) + pd.Timedelta(hours=i) for i in range(count)]


def test_hourly_to_daily_ohlcv_basic_aggregation():
    # Use 4 "hourly" bars per day to keep the fixture small.
    start = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    ts = _hourly_rows(start, 4) + _hourly_rows(start + pd.Timedelta(days=1), 4)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [10, 11, 12, 13, 20, 21, 22, 23],
            "high": [11, 12, 13, 14, 21, 22, 23, 24],
            "low": [9, 8, 7, 6, 19, 18, 17, 16],
            "close": [10.5, 11.5, 12.5, 13.5, 20.5, 21.5, 22.5, 23.5],
            "volume": [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 4.0],
            "trade_count": [1, 2, 3, 4, 5, 6, 7, 8],
            # Pick non-trivial vwap values so weighting matters.
            "vwap": [10.0, 10.0, 20.0, 30.0, 20.0, 999.0, 22.0, 24.0],
            "symbol": ["SOLFDUSD"] * 8,
        }
    )

    daily, stats = hourly_to_daily_ohlcv(
        df,
        expected_bars_per_day=4,
        drop_incomplete_last_day=False,
        output_symbol="SOLUSD",
    )

    assert stats.dropped_incomplete_last_day is False
    assert stats.dropped_incomplete_days == 0
    assert len(daily) == 2
    assert list(daily.columns)[:5] == ["timestamp", "open", "high", "low", "close"]

    # Day 1 (first 4 rows)
    d1 = daily.iloc[0]
    assert pd.Timestamp(d1["timestamp"]) == pd.Timestamp("2026-02-01T00:00:00Z")
    assert d1["open"] == 10
    assert d1["high"] == 14
    assert d1["low"] == 6
    assert d1["close"] == 13.5
    assert d1["volume"] == 10.0
    assert d1["trade_count"] == 10
    # Weighted vwap: sum(vol*vwap)/sum(vol) = (1*10 + 2*10 + 3*20 + 4*30)/10 = 21
    assert np.isclose(float(d1["vwap"]), 21.0)
    assert d1["symbol"] == "SOLUSD"
    assert d1["bar_count"] == 4

    # Day 2 (next 4 rows)
    d2 = daily.iloc[1]
    assert pd.Timestamp(d2["timestamp"]) == pd.Timestamp("2026-02-02T00:00:00Z")
    assert d2["open"] == 20
    assert d2["high"] == 24
    assert d2["low"] == 16
    assert d2["close"] == 23.5
    assert d2["volume"] == 10.0
    assert d2["trade_count"] == 26
    # Weighted vwap: volume=0 row should not contribute.
    # (5*20 + 0*999 + 1*22 + 4*24)/10 = (100 + 0 + 22 + 96) / 10 = 21.8
    assert np.isclose(float(d2["vwap"]), 21.8)
    assert d2["symbol"] == "SOLUSD"
    assert d2["bar_count"] == 4


def test_hourly_to_daily_ohlcv_drops_incomplete_last_day_only():
    start = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    ts = _hourly_rows(start, 4) + _hourly_rows(start + pd.Timedelta(days=1), 3)  # day1=4, day2=3 (incomplete)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": np.arange(7, dtype=float),
            "high": np.arange(7, dtype=float) + 1.0,
            "low": np.arange(7, dtype=float) - 1.0,
            "close": np.arange(7, dtype=float) + 0.5,
            "symbol": ["SOLFDUSD"] * 7,
        }
    )

    daily, stats = hourly_to_daily_ohlcv(
        df,
        expected_bars_per_day=4,
        drop_incomplete_last_day=True,
        drop_incomplete_days=False,
    )
    assert len(daily) == 1
    assert stats.dropped_incomplete_last_day is True
    assert stats.dropped_incomplete_days == 0


def test_hourly_to_daily_ohlcv_drops_all_incomplete_days():
    start = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    ts = (
        _hourly_rows(start, 4)
        + _hourly_rows(start + pd.Timedelta(days=1), 4)
        + _hourly_rows(start + pd.Timedelta(days=2), 3)
    )  # day1=4, day2=4, day3=3 incomplete
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": np.arange(11, dtype=float),
            "high": np.arange(11, dtype=float) + 1.0,
            "low": np.arange(11, dtype=float) - 1.0,
            "close": np.arange(11, dtype=float) + 0.5,
            "symbol": ["SOLFDUSD"] * 11,
        }
    )

    daily, stats = hourly_to_daily_ohlcv(
        df,
        expected_bars_per_day=4,
        drop_incomplete_last_day=True,
        drop_incomplete_days=True,
    )
    assert len(daily) == 2
    assert stats.dropped_incomplete_last_day is False  # drop_incomplete_days takes precedence
    assert stats.dropped_incomplete_days == 1


def test_hourly_to_daily_ohlcv_rejects_mixed_symbols():
    start = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    ts = _hourly_rows(start, 4)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [1, 1, 1, 1],
            "high": [1, 1, 1, 1],
            "low": [1, 1, 1, 1],
            "close": [1, 1, 1, 1],
            "symbol": ["SOLFDUSD", "SOLFDUSD", "BTCFDUSD", "SOLFDUSD"],
        }
    )
    with pytest.raises(ValueError, match="single symbol"):
        hourly_to_daily_ohlcv(df, expected_bars_per_day=4)
