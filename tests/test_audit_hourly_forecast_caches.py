from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_hourly_forecast_caches import audit_cache_pair, run_audit


def _write_hourly_csv(path: Path, timestamps: list[str]) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_cache(path: Path, timestamps: list[str]) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "symbol": "AAA",
            "predicted_close_p50": 100.0,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_audit_cache_pair_detects_missing_file(tmp_path: Path) -> None:
    _write_hourly_csv(
        tmp_path / "data" / "AAA.csv",
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"],
    )

    row = audit_cache_pair(
        symbol="AAA",
        horizon_hours=1,
        data_root=tmp_path / "data",
        forecast_cache_root=tmp_path / "cache",
    )

    assert row.missing_file is True
    assert row.cache_rows == 0
    assert row.has_issue is True


def test_audit_cache_pair_detects_stale_and_missing_timestamps(tmp_path: Path) -> None:
    _write_hourly_csv(
        tmp_path / "data" / "AAA.csv",
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
            "2026-01-01T03:00:00Z",
        ],
    )
    _write_cache(
        tmp_path / "cache" / "h1" / "AAA.parquet",
        [
            "2026-01-01T01:00:00Z",
            "2026-01-01T03:00:00Z",
        ],
    )

    row = audit_cache_pair(
        symbol="AAA",
        horizon_hours=1,
        data_root=tmp_path / "data",
        forecast_cache_root=tmp_path / "cache",
    )

    assert row.missing_file is False
    assert row.latest_gap_hours == 0.0
    assert row.missing_timestamps_in_cache_range == 1
    assert row.has_issue is True


def test_run_audit_handles_multiple_symbols_and_horizons(tmp_path: Path) -> None:
    _write_hourly_csv(
        tmp_path / "data" / "AAA.csv",
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
    )
    _write_hourly_csv(
        tmp_path / "data" / "BBB.csv",
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
    )
    _write_cache(
        tmp_path / "cache" / "h1" / "AAA.parquet",
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
    )

    rows = run_audit(
        symbols=["AAA", "BBB"],
        horizons=[1, 24],
        data_root=tmp_path / "data",
        forecast_cache_root=tmp_path / "cache",
    )

    assert len(rows) == 4
    assert any(row.symbol == "AAA" and row.horizon_hours == 1 and row.has_issue is False for row in rows)
    assert any(row.symbol == "BBB" and row.missing_file is True for row in rows)
