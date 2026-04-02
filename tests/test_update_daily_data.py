import pandas as pd

from update_daily_data import (
    _market_data_symbol,
    _merge_training_frames,
    _prepare_training_frame,
    _sync_symbol,
    _storage_symbol,
    build_sync_report,
    load_symbols_file,
    resolve_requested_symbols,
    resolve_symbol_set,
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


def test_prepare_training_frame_collapses_duplicate_normalized_columns():
    raw = pd.DataFrame(
        {
            "timestamp": ["2025-01-01 00:00:00+00:00"],
            "open": [100],
            "Open": [None],
            "high": [105],
            "High": [None],
            "low": [95],
            "Low": [None],
            "close": [102],
            "Close": [None],
        }
    )
    prepared = _prepare_training_frame(raw, "CRWD")
    assert prepared.loc[0, "open"] == 100
    assert prepared.loc[0, "high"] == 105
    assert prepared.loc[0, "low"] == 95
    assert prepared.loc[0, "close"] == 102


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


def test_sync_symbol_accepts_dotted_snapshot_for_hyphenated_symbol(tmp_path):
    snapshot_dir = tmp_path / "data" / "train"
    training_dir = tmp_path / "trainingdata" / "train"
    snapshot_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)

    symbol = "BRK-B"
    snapshot_path = snapshot_dir / "BRK.B-2026-04-01.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-03-31 00:00:00+00:00", "2026-04-01 00:00:00+00:00"],
            "Open": [500, 505],
            "High": [501, 506],
            "Low": [499, 504],
            "Close": [500.5, 505.5],
        }
    ).to_csv(snapshot_path, index=False)

    existing_path = training_dir / "BRK-B.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-03-30 00:00:00+00:00"],
            "open": [499],
            "high": [500],
            "low": [498],
            "close": [499.5],
            "volume": [0],
            "trade_count": [0],
            "vwap": [499.5],
            "symbol": ["BRK-B"],
        }
    ).to_csv(existing_path, index=False)

    appended = _sync_symbol(symbol, snapshot_dir, training_dir)
    assert appended == 2

    synced = pd.read_csv(existing_path)
    assert list(synced["symbol"].unique()) == ["BRK-B"]
    assert float(synced.iloc[-1]["close"]) == 505.5


def test_market_data_symbol_maps_share_classes_for_alpaca():
    assert _market_data_symbol("BRK-B") == "BRK.B"
    assert _market_data_symbol("BF-B") == "BF.B"
    assert _market_data_symbol("AAPL") == "AAPL"


def test_resolve_symbol_set_stock_expansion_contains_known_names():
    symbols = resolve_symbol_set("stock-expansion")
    assert "PLTR" in symbols
    assert "JPM" in symbols


def test_load_symbols_file_supports_comments_and_commas(tmp_path):
    path = tmp_path / "symbols.txt"
    path.write_text("pltr, nflx\n# ignore\njpm\n")
    assert load_symbols_file(path) == ["JPM", "NFLX", "PLTR"]


def test_resolve_requested_symbols_combines_sources(tmp_path):
    train_dir = tmp_path / "trainingdata" / "train"
    train_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": ["2025-01-01 00:00:00+00:00"],
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [0],
            "trade_count": [0],
            "vwap": [1],
            "symbol": ["AAPL"],
        }
    ).to_csv(train_dir / "AAPL.csv", index=False)
    symbols_path = tmp_path / "symbols.txt"
    symbols_path.write_text("nflx\n")

    resolved = resolve_requested_symbols(
        cli_symbols=["pltr"],
        symbol_set="alpaca-live8",
        symbols_file=symbols_path,
        training_dir=train_dir,
    )

    assert "PLTR" in resolved
    assert "NFLX" in resolved
    assert "NVDA" in resolved


def test_build_sync_report_returns_freshness(tmp_path):
    train_dir = tmp_path / "trainingdata" / "train"
    train_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": ["2026-03-29 00:00:00+00:00", "2026-03-31 00:00:00+00:00"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [1, 2],
            "volume": [0, 0],
            "trade_count": [0, 0],
            "vwap": [1, 2],
            "symbol": ["PLTR", "PLTR"],
        }
    ).to_csv(train_dir / "PLTR.csv", index=False)

    rows = build_sync_report(
        ["PLTR", "NFLX"],
        {"PLTR": 2, "NFLX": 0},
        training_dir=train_dir,
        as_of=pd.Timestamp("2026-04-01 00:00:00+00:00"),
    )

    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["PLTR"]["exists"] is True
    assert by_symbol["PLTR"]["stale_days"] == 1
    assert by_symbol["PLTR"]["appended_rows"] == 2
    assert by_symbol["NFLX"]["exists"] is False
