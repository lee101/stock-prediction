from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.hourly_gap_filler import find_large_gaps, fill_hourly_gaps_for_symbol


def test_find_large_gaps_detects_multi_day_gap():
    base = pd.Timestamp(datetime(2024, 1, 1, tzinfo=timezone.utc))
    timestamps = [
        base,
        base + pd.Timedelta(hours=1),
        base + pd.Timedelta(hours=2),
        base + pd.Timedelta(days=10),
        base + pd.Timedelta(days=10, hours=1),
    ]
    gaps = find_large_gaps(timestamps, min_gap=pd.Timedelta(days=5))
    assert len(gaps) == 1
    assert gaps[0].start == timestamps[2]
    assert gaps[0].end == timestamps[3]


def test_fill_hourly_gaps_for_symbol_merges_downloaded_rows(tmp_path: Path):
    # Minimal hourly CSV with a large gap.
    stock_dir = tmp_path / "stocks"
    stock_dir.mkdir(parents=True, exist_ok=True)
    path = stock_dir / "TEST.csv"

    base = pd.Timestamp(datetime(2024, 1, 1, tzinfo=timezone.utc))
    existing = pd.DataFrame(
        [
            {
                "timestamp": base,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
                "trade_count": 0.0,
                "vwap": 1.0,
                "symbol": "TEST",
            },
            {
                "timestamp": base + pd.Timedelta(hours=1),
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
                "trade_count": 0.0,
                "vwap": 1.0,
                "symbol": "TEST",
            },
            {
                "timestamp": base + pd.Timedelta(days=10),
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
                "trade_count": 0.0,
                "vwap": 1.0,
                "symbol": "TEST",
            },
        ]
    )
    existing.to_csv(path, index=False)

    def _dummy_stock_fetch(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        assert symbol == "TEST"
        # Return one bar inside the gap.
        idx = pd.DatetimeIndex([base + pd.Timedelta(days=5)], tz=timezone.utc)
        return pd.DataFrame(
            {
                "open": [2.0],
                "high": [2.0],
                "low": [2.0],
                "close": [2.0],
                "volume": [2.0],
                "trade_count": [0.0],
                "vwap": [2.0],
                "symbol": ["TEST"],
            },
            index=idx,
        )

    result = fill_hourly_gaps_for_symbol(
        "TEST",
        data_root=tmp_path,
        min_gap=pd.Timedelta(days=5),
        scan_start=base - pd.Timedelta(days=1),
        scan_end=base + pd.Timedelta(days=11),
        overlap_hours=0,
        stock_fetcher=_dummy_stock_fetch,
        sleep_seconds=0.0,
    )
    assert result["status"] == "ok"
    assert result["gaps_filled"] == 1
    assert result["added_rows"] == 1

    repaired = pd.read_csv(path)
    repaired["timestamp"] = pd.to_datetime(repaired["timestamp"], utc=True)
    assert (repaired["timestamp"] == base + pd.Timedelta(days=5)).any()

