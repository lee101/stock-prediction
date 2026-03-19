from __future__ import annotations

from pathlib import Path

import pandas as pd

import src.forecast_cache_lookup as forecast_cache_lookup
from src.forecast_cache_lookup import load_latest_forecast_from_cache


def _write_forecast(path: Path, *, close_value: float, include_symbol: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-14T00:00:00Z"),
                "symbol": include_symbol,
                "predicted_close_p50": close_value,
                "predicted_high_p50": close_value + 5.0,
                "predicted_low_p50": close_value - 5.0,
            }
        ]
    )
    frame.to_parquet(path, index=False)


def test_load_latest_forecast_from_cache_uses_exact_symbol_first(tmp_path: Path) -> None:
    _write_forecast(tmp_path / "h24" / "BNBUSD.parquet", close_value=610.0, include_symbol="BNBUSD")
    _write_forecast(tmp_path / "h24" / "BNBUSDT.parquet", close_value=620.0, include_symbol="BNBUSDT")

    forecast = load_latest_forecast_from_cache("BNBUSD", 24, tmp_path)

    assert forecast is not None
    assert forecast["predicted_close_p50"] == 610.0


def test_load_latest_forecast_from_cache_falls_back_to_alias(tmp_path: Path) -> None:
    _write_forecast(tmp_path / "h24" / "BNBUSDT.parquet", close_value=615.0, include_symbol="BNBUSDT")

    forecast = load_latest_forecast_from_cache("BNBUSD", 24, tmp_path)

    assert forecast is not None
    assert forecast["predicted_close_p50"] == 615.0


def test_load_latest_forecast_from_cache_works_for_hourly_horizon(tmp_path: Path) -> None:
    _write_forecast(tmp_path / "h1" / "BNBUSDT.parquet", close_value=605.0, include_symbol="BNBUSDT")

    forecast = load_latest_forecast_from_cache("BNBUSD", 1, tmp_path)

    assert forecast is not None
    assert forecast["predicted_close_p50"] == 605.0


def test_load_latest_forecast_from_cache_uses_latest_row_available_at_as_of(tmp_path: Path) -> None:
    path = tmp_path / "h24" / "BNBUSDT.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-03-10T00:00:00Z"),
                "issued_at": pd.Timestamp("2026-03-09T00:00:00Z"),
                "symbol": "BNBUSDT",
                "predicted_close_p50": 600.0,
                "predicted_high_p50": 605.0,
                "predicted_low_p50": 595.0,
            },
            {
                "timestamp": pd.Timestamp("2026-03-12T00:00:00Z"),
                "issued_at": pd.Timestamp("2026-03-11T00:00:00Z"),
                "symbol": "BNBUSDT",
                "predicted_close_p50": 630.0,
                "predicted_high_p50": 635.0,
                "predicted_low_p50": 625.0,
            },
        ]
    )
    frame.to_parquet(path, index=False)

    forecast = load_latest_forecast_from_cache(
        "BNBUSD",
        24,
        tmp_path,
        as_of="2026-03-10T12:00:00Z",
    )

    assert forecast is not None
    assert forecast["predicted_close_p50"] == 600.0


def test_load_latest_forecast_from_cache_reuses_cached_parquet_reads(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "h1" / "BNBUSDT.parquet"
    _write_forecast(path, close_value=605.0, include_symbol="BNBUSDT")

    forecast_cache_lookup._read_forecast_frame.cache_clear()
    read_count = 0
    original_read_parquet = forecast_cache_lookup.pd.read_parquet

    def counting_read_parquet(*args, **kwargs):
        nonlocal read_count
        read_count += 1
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(forecast_cache_lookup.pd, "read_parquet", counting_read_parquet)

    forecast_a = load_latest_forecast_from_cache("BNBUSD", 1, tmp_path)
    forecast_b = load_latest_forecast_from_cache("BNBUSD", 1, tmp_path)

    assert forecast_a is not None
    assert forecast_b is not None
    assert read_count == 1
