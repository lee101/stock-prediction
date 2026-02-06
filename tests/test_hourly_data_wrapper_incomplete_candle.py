from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from binance_data_wrapper import _drop_incomplete_current_hour


def test_drop_incomplete_current_hour_datetime_index() -> None:
    now = datetime(2026, 2, 6, 11, 37, tzinfo=timezone.utc)
    current_hour = pd.Timestamp("2026-02-06T11:00:00Z")
    prior_hour = pd.Timestamp("2026-02-06T10:00:00Z")

    frame = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.DatetimeIndex([prior_hour, current_hour]))
    cleaned = _drop_incomplete_current_hour(frame, now=now)
    assert cleaned is not None
    assert len(cleaned) == 1
    assert cleaned.index[0] == prior_hour


def test_drop_incomplete_current_hour_timestamp_column() -> None:
    now = datetime(2026, 2, 6, 11, 1, tzinfo=timezone.utc)
    current_hour = pd.Timestamp("2026-02-06T11:00:00Z")
    prior_hour = pd.Timestamp("2026-02-06T10:00:00Z")

    frame = pd.DataFrame(
        {
            "timestamp": [prior_hour.isoformat(), current_hour.isoformat()],
            "close": [1.0, 2.0],
        }
    )
    cleaned = _drop_incomplete_current_hour(frame, now=now)
    assert cleaned is not None
    assert len(cleaned) == 1
    assert pd.to_datetime(cleaned["timestamp"], utc=True).iloc[0] == prior_hour

