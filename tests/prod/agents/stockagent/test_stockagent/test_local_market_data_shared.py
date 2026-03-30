from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from stockagent.agentsimulator import local_market_data


def test_find_latest_local_symbol_files_returns_empty_when_directory_scan_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_iterdir = Path.iterdir

    def _broken_iterdir(self: Path):
        if self == tmp_path:
            raise OSError("permission denied")
        yield from original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _broken_iterdir)

    indexed = local_market_data.find_latest_local_symbol_files(
        symbols=["AAPL"],
        directories=[tmp_path],
    )

    assert indexed == {}


def test_load_local_data_file_returns_empty_on_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "AAPL.csv"
    csv_path.write_text("timestamp,close\n2025-01-01T00:00:00Z,100\n", encoding="utf-8")
    original_read_csv = pd.read_csv

    def _broken_read_csv(path: Path, *args, **kwargs):
        if Path(path) == csv_path:
            raise OSError("disk read failure")
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", _broken_read_csv)

    df = local_market_data.load_local_data_file(symbol="AAPL", path=csv_path)

    assert df.empty


def test_ensure_datetime_index_rejects_frames_without_timestamp_column() -> None:
    frame = pd.DataFrame({"close": [100.0, 101.0]})

    indexed = local_market_data.ensure_datetime_index(frame)

    assert indexed.empty


def test_ensure_datetime_index_accepts_datetime_index() -> None:
    frame = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2025-01-02T00:00:00Z")]),
    )

    indexed = local_market_data.ensure_datetime_index(frame)

    assert list(indexed["close"]) == [100.0, 101.0]
    assert isinstance(indexed.index, pd.DatetimeIndex)


def test_find_latest_local_symbol_files_requires_filename_boundary(tmp_path: Path) -> None:
    frame = pd.DataFrame({"timestamp": ["2025-01-01T00:00:00Z"], "close": [100.0]})
    eth_path = tmp_path / "ETH_snapshot.csv"
    ethusd_path = tmp_path / "ETHUSD_newer.csv"
    frame.to_csv(eth_path, index=False)
    frame.assign(close=[200.0]).to_csv(ethusd_path, index=False)

    indexed = local_market_data.find_latest_local_symbol_files(
        symbols=["ETH"],
        directories=[tmp_path],
    )

    assert indexed["ETH"] == eth_path


def test_find_latest_local_symbol_files_supports_symbols_with_internal_separators(tmp_path: Path) -> None:
    frame = pd.DataFrame({"timestamp": ["2025-01-01T00:00:00Z"], "close": [100.0]})
    brkb_path = tmp_path / "BRK-B_snapshot.csv"
    frame.to_csv(brkb_path, index=False)

    indexed = local_market_data.find_latest_local_symbol_files(
        symbols=["BRK-B"],
        directories=[tmp_path],
    )

    assert indexed["BRK-B"] == brkb_path


def test_shared_local_data_defaults_can_be_overridden_via_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_dir = tmp_path / "custom"
    monkeypatch.setenv(local_market_data.LOCAL_DATA_DIR_ENV, custom_dir.as_posix())
    monkeypatch.setenv(local_market_data.USE_FALLBACK_DATA_DIRS_ENV, "false")

    assert local_market_data.default_local_data_dir() == custom_dir
    assert local_market_data.default_use_fallback_data_dirs() is False


def test_shared_local_data_defaults_fall_back_on_invalid_bool_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(local_market_data.USE_FALLBACK_DATA_DIRS_ENV, "definitely")

    assert local_market_data.default_use_fallback_data_dirs() is True


def test_build_market_data_bundle_uses_default_symbols_and_remote_loader(tmp_path: Path) -> None:
    remote_calls: list[str] = []
    remote_frame = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2025-01-01T00:00:00Z"),
                pd.Timestamp("2025-01-02T00:00:00Z"),
                pd.Timestamp("2025-01-03T00:00:00Z"),
            ]
        ),
    )

    def _remote_loader(symbol: str) -> pd.DataFrame:
        remote_calls.append(symbol)
        return remote_frame

    bundle = local_market_data.build_market_data_bundle(
        symbols=[],
        default_symbols=["AAPL"],
        lookback_days=2,
        as_of=datetime(2025, 1, 5, tzinfo=timezone.utc),
        local_data_dir=tmp_path,
        fallback_data_dirs=(),
        remote_loader=_remote_loader,
    )

    assert remote_calls == ["AAPL"]
    assert list(bundle.get_symbol_bars("AAPL")["close"]) == [101.0, 102.0]
