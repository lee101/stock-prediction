from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from trainingdatadaily.download_crypto_daily import (
    DEFAULT_HISTORY_YEARS,
    download_and_save,
    parse_date,
    resolve_dates,
    resolve_symbols,
)


def test_parse_date_returns_utc():
    naive = "2024-01-01"
    parsed = parse_date(naive)
    assert parsed.tzinfo == timezone.utc
    assert parsed.year == 2024

    aware = "2024-01-01T05:00:00-05:00"
    parsed_aware = parse_date(aware)
    assert parsed_aware.tzinfo == timezone.utc
    assert parsed_aware.hour == 10  # shifted to UTC


def test_resolve_dates_history_window():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    start, end = resolve_dates(None, None, history_years=DEFAULT_HISTORY_YEARS, now=now)
    assert start < end
    expected_days = int(DEFAULT_HISTORY_YEARS * 365.25)
    assert (end - start).days in {expected_days, expected_days + 1}


def test_resolve_dates_start_after_end_raises():
    with pytest.raises(ValueError):
        resolve_dates("2024-01-02", "2024-01-01", history_years=1.0)


def test_resolve_symbols_defaults_match_universe():
    symbols = resolve_symbols(None)
    # Ensure the defaults contain representative crypto tickers and are sorted.
    assert "BTCUSD" in symbols
    assert symbols == sorted(symbols)


def _stub_fetch(symbol: str, start: datetime, end: datetime, include_latest: bool) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=3, freq="D", tz=timezone.utc)
    return pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
            "volume": [100, 200, 300],
            "symbol": symbol,
        },
        index=index,
    )


def test_download_and_save_writes_files(tmp_path: Path):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, tzinfo=timezone.utc)

    results = download_and_save(
        symbols=["BTCUSD"],
        start_dt=start,
        end_dt=end,
        output_dir=tmp_path,
        include_latest=False,
        sleep_seconds=0.0,
        fetch_fn=_stub_fetch,
    )

    assert results and results[0]["status"] == "ok"
    output_file = tmp_path / "BTCUSD.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file, index_col=0, parse_dates=True)
    assert len(df) == 3
    assert "symbol" not in df.columns

    summary = tmp_path / "summary.csv"
    assert summary.exists()
    summary_df = pd.read_csv(summary)
    assert summary_df.loc[0, "symbol"] == "BTCUSD"
