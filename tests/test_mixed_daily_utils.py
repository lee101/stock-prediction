from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.mixed_daily_utils import align_daily_price_frames, latest_snapshot, summarize_symbol_coverage


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_align_daily_price_frames_normalizes_mixed_stock_crypto_timestamps(tmp_path: Path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    _write_csv(
        root / "AAPL.csv",
        [
            {"timestamp": "2025-01-01 05:00:00+00:00", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
            {"timestamp": "2025-01-02 05:00:00+00:00", "open": 102, "high": 103, "low": 101, "close": 102, "volume": 11},
        ],
    )
    _write_csv(
        root / "BTCUSD.csv",
        [
            {"timestamp": "2025-01-01 00:00:00+00:00", "open": 200, "high": 201, "low": 199, "close": 200, "volume": 20},
            {"timestamp": "2025-01-02 00:00:00+00:00", "open": 202, "high": 203, "low": 201, "close": 202, "volume": 21},
        ],
    )

    aligned = align_daily_price_frames(["AAPL", "BTCUSD"], data_root=root, min_days=2)

    assert list(aligned.index.strftime("%Y-%m-%d")) == ["2025-01-01", "2025-01-02"]
    assert aligned.tradable["AAPL"].tolist() == [True, True]
    assert aligned.tradable["BTCUSD"].tolist() == [True, True]

    snap = latest_snapshot(aligned)
    assert snap.as_of.strftime("%Y-%m-%d") == "2025-01-02"
    assert snap.feature_matrix.shape == (2, 16)
    assert snap.prices["AAPL"] == 102.0
    assert snap.prices["BTCUSD"] == 202.0


def test_align_daily_price_frames_marks_closed_days_non_tradable(tmp_path: Path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    _write_csv(
        root / "AAPL.csv",
        [
            {"timestamp": "2025-01-01 05:00:00+00:00", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
            {"timestamp": "2025-01-03 05:00:00+00:00", "open": 103, "high": 104, "low": 102, "close": 103, "volume": 13},
        ],
    )
    _write_csv(
        root / "BTCUSD.csv",
        [
            {"timestamp": "2025-01-01 00:00:00+00:00", "open": 200, "high": 201, "low": 199, "close": 200, "volume": 20},
            {"timestamp": "2025-01-02 00:00:00+00:00", "open": 201, "high": 202, "low": 200, "close": 201, "volume": 21},
            {"timestamp": "2025-01-03 00:00:00+00:00", "open": 202, "high": 203, "low": 201, "close": 202, "volume": 22},
        ],
    )

    aligned = align_daily_price_frames(["AAPL", "BTCUSD"], data_root=root, min_days=3)

    assert list(aligned.index.strftime("%Y-%m-%d")) == ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert aligned.tradable["AAPL"].tolist() == [True, False, True]
    assert aligned.prices["AAPL"].iloc[1]["close"] == 100.0
    assert aligned.prices["AAPL"].iloc[1]["volume"] == 0.0


def test_summarize_symbol_coverage_reports_bounds(tmp_path: Path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    _write_csv(
        root / "BTCUSD.csv",
        [
            {"timestamp": "2025-01-01 00:00:00+00:00", "open": 200, "high": 201, "low": 199, "close": 200, "volume": 20},
            {"timestamp": "2025-01-02 00:00:00+00:00", "open": 202, "high": 203, "low": 201, "close": 202, "volume": 21},
        ],
    )

    rows = summarize_symbol_coverage(["BTCUSD"], data_root=root)

    assert len(rows) == 1
    assert rows[0].symbol == "BTCUSD"
    assert rows[0].first_date.strftime("%Y-%m-%d") == "2025-01-01"
    assert rows[0].last_date.strftime("%Y-%m-%d") == "2025-01-02"
    assert rows[0].num_rows == 2


def test_align_daily_price_frames_coalesces_duplicate_shadow_ohlc_columns(tmp_path: Path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    (root / "NET.csv").write_text(
        "\n".join(
            [
                "timestamp,open,high,low,close,volume,Open,High,Low,Close",
                "2025-01-01T00:00:00Z,10,11,9,10.5,1000,,,,",
                "2025-01-02T00:00:00Z,10.5,11.5,10,11,1100,,,,",
            ]
        )
    )
    _write_csv(
        root / "BTCUSD.csv",
        [
            {"timestamp": "2025-01-01 00:00:00+00:00", "open": 200, "high": 201, "low": 199, "close": 200, "volume": 20},
            {"timestamp": "2025-01-02 00:00:00+00:00", "open": 202, "high": 203, "low": 201, "close": 202, "volume": 21},
        ],
    )

    aligned = align_daily_price_frames(["NET", "BTCUSD"], data_root=root, min_days=2)

    assert aligned.prices["NET"].iloc[-1]["close"] == 11.0
    assert aligned.features["NET"].shape[1] == 16
