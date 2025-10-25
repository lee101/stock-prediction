from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from trainingdatahourly.download_hourly_bars import (
    DEFAULT_HOURLY_STOCK_SYMBOLS,
    DEFAULT_HISTORY_YEARS,
    SymbolSpec,
    download_and_save,
    parse_date,
    resolve_symbol_specs,
    resolve_window,
)


def _dummy_fetch(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=4, freq="h", tz=timezone.utc)
    return pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.05, 1.15, 1.25, 1.35],
            "volume": [10, 20, 30, 40],
            "symbol": symbol,
        },
        index=index,
    )


def test_parse_date_normalizes_to_utc():
    value = "2024-05-01T12:30:00-04:00"
    parsed = parse_date(value)
    assert parsed.tzinfo == timezone.utc
    assert parsed.hour == 16


def test_resolve_window_defaults():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    start, end = resolve_window(None, None, history_years=DEFAULT_HISTORY_YEARS, now=now)
    assert start < end
    expected_days = int(DEFAULT_HISTORY_YEARS * 365.25)
    assert (end - start).days in {expected_days, expected_days + 1}


def test_resolve_window_invalid_range():
    with pytest.raises(ValueError):
        resolve_window("2024-01-02", "2024-01-01", history_years=1)


def test_resolve_symbol_specs_defaults_include_crypto_and_stocks():
    specs = resolve_symbol_specs(symbols=None, include_crypto=True, include_stocks=True, stock_symbols=None)
    crypto_specs = [s for s in specs if s.asset_class == "crypto"]
    stock_specs = [s for s in specs if s.asset_class == "stock"]
    assert crypto_specs
    assert stock_specs
    resolved_stock_symbols = {spec.symbol for spec in stock_specs}
    expected_subset = set(DEFAULT_HOURLY_STOCK_SYMBOLS)
    assert resolved_stock_symbols.issubset(expected_subset)


def test_resolve_symbol_specs_with_filtering():
    specs = resolve_symbol_specs(symbols=["BTCUSD", "AAPL"], include_crypto=True, include_stocks=False)
    assert specs == [SymbolSpec(symbol="BTCUSD", asset_class="crypto")]


def test_download_and_save(tmp_path: Path):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 3, tzinfo=timezone.utc)
    specs = [SymbolSpec(symbol="BTCUSD", asset_class="crypto"), SymbolSpec(symbol="AAPL", asset_class="stock")]

    results = download_and_save(
        specs=specs,
        start_dt=start,
        end_dt=end,
        output_dir=tmp_path,
        sleep_seconds=0.0,
        crypto_fetcher=_dummy_fetch,
        stock_fetcher=_dummy_fetch,
    )

    assert len(results) == 2
    for entry in results:
        assert entry["status"] == "ok"

    crypto_file = tmp_path / "crypto" / "BTCUSD.csv"
    stock_file = tmp_path / "stock" / "AAPL.csv"
    assert crypto_file.exists()
    assert stock_file.exists()

    summary = tmp_path / "summary.csv"
    assert summary.exists()
    df = pd.read_csv(summary)
    assert set(df["symbol"]) == {"BTCUSD", "AAPL"}
