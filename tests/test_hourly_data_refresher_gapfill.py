from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.hourly_data_refresh import HourlyDataRefresher
from src.hourly_data_utils import HourlyDataValidator


def _write_hourly_csv(path: Path, symbol: str, timestamps: pd.DatetimeIndex) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1.0,
            "trade_count": 1,
            "vwap": 100.5,
            "symbol": symbol,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _dummy_fetcher(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    start_ts = start_ts.floor("h")
    end_ts = end_ts.floor("h")
    if start_ts >= end_ts:
        return pd.DataFrame()
    idx = pd.date_range(start_ts, end_ts, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1.0,
            "trade_count": 1,
            "vwap": 100.5,
            "symbol": symbol.upper(),
        },
        index=idx,
    )
    frame.index.name = "timestamp"
    return frame


def test_hourly_refresher_fills_stale_gaps_instead_of_appending_recent_window(tmp_path: Path) -> None:
    data_root = tmp_path
    symbol = "TESTUSD"
    csv_path = data_root / "crypto" / f"{symbol}.csv"

    existing_ts = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    _write_hourly_csv(csv_path, symbol, existing_ts)

    # Simulate a "stale" local file where the last bar is far older than the backfill window.
    now = datetime(2026, 1, 10, tzinfo=timezone.utc)
    validator = HourlyDataValidator(data_root, max_staleness_hours=10_000)
    refresher = HourlyDataRefresher(
        data_root,
        validator,
        stock_fetcher=_dummy_fetcher,
        crypto_fetcher=_dummy_fetcher,
        backfill_hours=48,
        overlap_hours=2,
        sleep_seconds=0.0,
    )

    refreshed = refresher._refresh_symbol(symbol, now)  # noqa: SLF001 - unit test targets gap-fill behavior
    assert refreshed is True

    out = pd.read_csv(csv_path)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    diffs = out["timestamp"].diff().dropna()

    # A stale file should be refreshed from its last known timestamp (not from now-backfill),
    # so we don't create a multi-day gap.
    assert diffs.max() <= pd.Timedelta(hours=1)
