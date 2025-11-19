import pandas as pd

from update_daily_data import (
    _merge_training_frames,
    _prepare_training_frame,
    _sync_symbol,
    _storage_symbol,
)


def test_prepare_training_frame_populates_optional_columns():
    raw = pd.DataFrame(
        {
            "Timestamp": ["2025-01-01 00:00:00+00:00"],
            "Open": [100],
            "High": [105],
            "Low": [95],
            "Close": [102],
        }
    )
    prepared = _prepare_training_frame(raw, "BTC/USD")
    assert list(prepared.columns[:9]) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
        "symbol",
    ]
    assert prepared.loc[0, "volume"] == 0.0
    assert prepared.loc[0, "trade_count"] == 0.0
    assert prepared.loc[0, "vwap"] == 102
    assert prepared.loc[0, "symbol"] == _storage_symbol("BTC/USD")


def test_merge_training_frames_deduplicates_and_counts_new_rows():
    existing = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01 00:00:00+00:00", "2025-01-02 00:00:00+00:00"], utc=True
            ),
            "open": [100, 101],
            "high": [110, 111],
            "low": [95, 96],
            "close": [105, 106],
            "volume": [10, 11],
            "trade_count": [1, 1],
            "vwap": [104, 105],
            "symbol": ["AAPL", "AAPL"],
        }
    )
    updates = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-02 00:00:00+00:00", "2025-01-03 00:00:00+00:00"], utc=True
            ),
            "open": [101, 102],
            "high": [111, 112],
            "low": [96, 97],
            "close": [106, 107],
            "volume": [12, 13],
            "trade_count": [1, 1],
            "vwap": [105, 106],
            "symbol": ["AAPL", "AAPL"],
        }
    )
    merged, new_rows = _merge_training_frames(existing, updates)
    assert new_rows == 1
    assert len(merged) == 3
    assert merged.iloc[-1]["close"] == 107


def test_sync_symbol_appends_snapshots(tmp_path):
    snapshot_dir = tmp_path / "data" / "train"
    training_dir = tmp_path / "trainingdata" / "train"
    snapshot_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)

    symbol = "AAPL"
    snapshot_path = snapshot_dir / f"{_storage_symbol(symbol)}-2025-11-17.csv"
    pd.DataFrame(
        {
            "symbol": [symbol, symbol],
            "timestamp": [
                "2025-11-16 00:00:00+00:00",
                "2025-11-17 00:00:00+00:00",
            ],
            "Open": [150, 152],
            "High": [155, 156],
            "Low": [149, 150],
            "Close": [154, 155],
            "Volume": [1_000_000, 1_100_000],
            "Trade_Count": [100, 120],
            "VWAP": [153, 154],
        }
    ).to_csv(snapshot_path, index=False)

    existing_path = training_dir / f"{_storage_symbol(symbol)}.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-11-15 00:00:00+00:00"],
            "open": [148],
            "high": [152],
            "low": [147],
            "close": [150],
            "volume": [900000],
            "trade_count": [90],
            "vwap": [149],
            "symbol": [symbol],
        }
    ).to_csv(existing_path, index=False)

    appended = _sync_symbol(symbol, snapshot_dir, training_dir)
    assert appended == 2

    synced = pd.read_csv(existing_path)
    assert len(synced) == 3
    assert synced.iloc[-1]["close"] == 155
