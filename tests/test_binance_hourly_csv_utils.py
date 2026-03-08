from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.binance_hourly_csv_utils import append_hourly_binance_bars


def test_append_hourly_binance_bars_uses_overlap_to_fill_gap(tmp_path: Path) -> None:
    csv_path = tmp_path / "BTCUSD.csv"
    existing = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-01T00:00:00Z",
                    "2026-03-01T01:00:00Z",
                    "2026-03-01T03:00:00Z",
                ],
                utc=True,
            ),
            "open": [10.0, 11.0, 13.0],
            "high": [10.5, 11.5, 13.5],
            "low": [9.5, 10.5, 12.5],
            "close": [10.2, 11.2, 13.2],
            "volume": [100, 110, 130],
            "symbol": ["BTCUSD", "BTCUSD", "BTCUSD"],
        }
    )
    existing.to_csv(csv_path, index=False)

    def _fake_fetch(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        assert symbol == "BTCUSDT"
        assert start == datetime(2026, 3, 1, 1, 0, tzinfo=timezone.utc)
        index = pd.to_datetime(
            [
                "2026-03-01T01:00:00Z",
                "2026-03-01T02:00:00Z",
                "2026-03-01T03:00:00Z",
                "2026-03-01T04:00:00Z",
            ],
            utc=True,
        )
        frame = pd.DataFrame(
            {
                "open": [21.0, 22.0, 23.0, 24.0],
                "high": [21.5, 22.5, 23.5, 24.5],
                "low": [20.5, 21.5, 22.5, 23.5],
                "close": [21.2, 22.2, 23.2, 24.2],
                "volume": [210, 220, 230, 240],
            },
            index=index,
        )
        frame.index.name = "timestamp"
        return frame

    result = append_hourly_binance_bars(
        csv_path,
        fetch_symbol="BTCUSDT",
        csv_symbol="BTCUSD",
        fetcher=_fake_fetch,
        overlap_hours=2,
        end_time=datetime(2026, 3, 1, 5, 0, tzinfo=timezone.utc),
    )

    assert result["status"] == "updated"
    assert result["rows_added"] == 2
    merged = pd.read_csv(csv_path, parse_dates=["timestamp"])
    assert list(merged["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")) == [
        "2026-03-01T00:00:00Z",
        "2026-03-01T01:00:00Z",
        "2026-03-01T02:00:00Z",
        "2026-03-01T03:00:00Z",
        "2026-03-01T04:00:00Z",
    ]
    assert merged.loc[merged["timestamp"] == pd.Timestamp("2026-03-01T03:00:00Z"), "open"].item() == 23.0


def test_append_hourly_binance_bars_missing_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "MISSING.csv"
    result = append_hourly_binance_bars(
        csv_path,
        fetch_symbol="BTCUSDT",
        csv_symbol="BTCUSD",
        fetcher=lambda symbol, start, end: pd.DataFrame(),
    )
    assert result["status"] == "missing_csv"
