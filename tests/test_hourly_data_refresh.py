from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.hourly_data_refresh import HourlyDataRefresher, fetch_binance_bars
from src.hourly_data_utils import HourlyDataValidator


def _build_frame(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    index = pd.date_range(start=start, end=end, freq="h", tz=timezone.utc)
    if index.empty:
        index = pd.date_range(end=end, periods=2, freq="h", tz=timezone.utc)
    data = {
        "open": [1.0 + idx for idx in range(len(index))],
        "high": [1.1 + idx for idx in range(len(index))],
        "low": [0.9 + idx for idx in range(len(index))],
        "close": [1.05 + idx for idx in range(len(index))],
        "volume": [10 + idx for idx in range(len(index))],
        "trade_count": [5 + idx for idx in range(len(index))],
        "vwap": [1.02 + idx for idx in range(len(index))],
        "symbol": symbol,
    }
    frame = pd.DataFrame(data, index=index)
    frame.index.name = "timestamp"
    return frame


def test_hourly_refresher_populates_missing_stock(tmp_path: Path) -> None:
    validator = HourlyDataValidator(tmp_path, max_staleness_hours=6)
    refresher = HourlyDataRefresher(
        data_root=tmp_path,
        validator=validator,
        stock_fetcher=_build_frame,
        crypto_fetcher=_build_frame,
        crypto_max_staleness_hours=1.5,
    )
    statuses, issues = refresher.refresh(["AAPL"])
    assert not issues
    assert statuses and statuses[0].symbol == "AAPL"
    path = tmp_path / "stocks" / "AAPL.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert not df.empty


def test_hourly_refresher_appends_overlaps(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    past_index = pd.date_range(end=now - timedelta(hours=8), periods=2, freq="h", tz=timezone.utc)
    existing = _build_frame("BTCUSD", past_index[0], past_index[-1])
    target = tmp_path / "crypto" / "BTCUSD.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    existing.to_csv(target)

    validator = HourlyDataValidator(tmp_path, max_staleness_hours=2)

    captured: dict[str, datetime] = {}

    def _crypto_fetcher(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        captured["start"] = start
        captured["end"] = end
        return _build_frame(symbol, end - timedelta(hours=1), end)

    refresher = HourlyDataRefresher(
        data_root=tmp_path,
        validator=validator,
        stock_fetcher=_build_frame,
        crypto_fetcher=_crypto_fetcher,
        crypto_max_staleness_hours=1.5,
        overlap_hours=1,
    )
    statuses, issues = refresher.refresh(["BTCUSD"])
    assert not issues
    assert statuses and statuses[0].symbol == "BTCUSD"
    assert "start" in captured and "end" in captured
    updated = pd.read_csv(target)
    assert len(updated) >= len(existing)
    latest = pd.to_datetime(updated["timestamp"], utc=True).max()
    assert latest >= captured["end"] - timedelta(hours=1)


def test_hourly_refresher_chunks_large_crypto_gaps(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    # Force a large gap so the refresher needs multiple requests.
    past_end = now - timedelta(hours=2500)
    existing = _build_frame("ALGOUSD", past_end - timedelta(hours=1), past_end)
    target = tmp_path / "crypto" / "ALGOUSD.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    existing.to_csv(target, index=True)

    validator = HourlyDataValidator(tmp_path, max_staleness_hours=1)

    calls: list[tuple[datetime, datetime]] = []

    def _crypto_fetcher(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        calls.append((start, end))
        # Emit one bar per hour for the requested window.
        index = pd.date_range(start=start, end=end, freq="h", tz=timezone.utc)
        if index.empty:
            index = pd.date_range(end=end, periods=1, freq="h", tz=timezone.utc)
        frame = pd.DataFrame(
            {
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 0.0,
                "trade_count": 0.0,
                "vwap": 1.0,
                "symbol": symbol,
            },
            index=index,
        )
        frame.index.name = "timestamp"
        return frame

    refresher = HourlyDataRefresher(
        data_root=tmp_path,
        validator=validator,
        stock_fetcher=_build_frame,
        crypto_fetcher=_crypto_fetcher,
        max_request_hours_crypto=100,
        crypto_max_staleness_hours=1.5,
        overlap_hours=1,
    )
    statuses, issues = refresher.refresh(["ALGOUSD"])
    assert not issues
    assert statuses and statuses[0].symbol == "ALGOUSD"
    assert len(calls) > 1
    assert all((end - start) <= timedelta(hours=100) for start, end in calls)


def test_fetch_binance_bars_parses_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)
    payload = [
        [int(start.timestamp() * 1000), "10", "11", "9", "10.5", "100", int((start.timestamp() + 3600) * 1000), "1050", 12, "60", "630", "0"],
        [int((start.timestamp() + 3600) * 1000), "11", "12", "10", "11.5", "120", int((start.timestamp() + 7200) * 1000), "1380", 8, "70", "805", "0"],
    ]

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return payload

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: _Resp())
    frame = fetch_binance_bars("BTCUSD", start, end)
    assert not frame.empty
    assert "close" in frame.columns
    assert frame.index[0].tzinfo == timezone.utc
