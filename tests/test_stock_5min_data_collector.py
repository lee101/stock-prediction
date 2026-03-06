from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.stock_5min_data_collector import (
    append_stock_bars,
    collect_recent_cycle,
    extract_stock_symbols_from_supervisor,
    normalize_stock_bars,
    parse_symbol_tokens,
    resolve_fetch_start,
    resolve_stock_symbols,
)


def test_parse_symbol_tokens_normalizes_and_deduplicates() -> None:
    assert parse_symbol_tokens(["nvda, pltr", "GOOG", "pltr"]) == ["NVDA", "PLTR", "GOOG"]


def test_extract_stock_symbols_from_supervisor_command(tmp_path: Path) -> None:
    config_path = tmp_path / "unified-stock-trader.conf"
    config_path.write_text(
        "[program:unified-stock-trader]\n"
        "command=/usr/bin/python trade.py --stock-symbols NVDA,PLTR,GOOG --loop\n",
        encoding="utf-8",
    )
    assert extract_stock_symbols_from_supervisor(config_path) == ["NVDA", "PLTR", "GOOG"]


def test_resolve_stock_symbols_prefers_explicit_values(tmp_path: Path) -> None:
    config_path = tmp_path / "unified-stock-trader.conf"
    config_path.write_text(
        "[program:x]\ncommand=/usr/bin/python trade.py --stock-symbols NVDA,PLTR\n",
        encoding="utf-8",
    )
    resolved = resolve_stock_symbols(
        symbols=["mtch, trip", "goog"],
        supervisor_config=config_path,
        fallback=("AAPL",),
    )
    assert resolved == ["MTCH", "TRIP", "GOOG"]


def test_normalize_stock_bars_drops_incomplete_latest_bar() -> None:
    index = pd.to_datetime(
        [
            "2026-03-06T14:50:00Z",
            "2026-03-06T14:55:00Z",
            "2026-03-06T15:00:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [10, 20, 30],
        },
        index=index,
    )
    normalized = normalize_stock_bars(
        "NVDA",
        frame,
        now=datetime(2026, 3, 6, 15, 2, tzinfo=timezone.utc),
    )
    assert list(normalized.index.astype(str)) == [
        "2026-03-06 14:50:00+00:00",
        "2026-03-06 14:55:00+00:00",
    ]
    assert list(normalized["trade_count"]) == [0, 0]
    assert list(normalized["vwap"]) == [1.05, 1.15]


def test_append_stock_bars_deduplicates_overlap(tmp_path: Path) -> None:
    path = tmp_path / "trainingdata5min" / "stocks" / "NVDA.csv"
    existing = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-06T14:45:00Z", "2026-03-06T14:50:00Z"], utc=True),
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [10, 20],
            "trade_count": [1, 2],
            "vwap": [1.04, 1.14],
            "symbol": ["NVDA", "NVDA"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    existing.to_csv(path, index=False)

    new = pd.DataFrame(
        {
            "open": [1.11, 1.2],
            "high": [1.21, 1.3],
            "low": [1.01, 1.1],
            "close": [1.16, 1.25],
            "volume": [21, 30],
            "trade_count": [3, 4],
            "vwap": [1.15, 1.24],
            "symbol": ["NVDA", "NVDA"],
        },
        index=pd.to_datetime(["2026-03-06T14:50:00Z", "2026-03-06T14:55:00Z"], utc=True),
    )
    new.index.name = "timestamp"

    stats = append_stock_bars(path, new)
    merged = pd.read_csv(path, parse_dates=["timestamp"])
    assert stats["appended"] == 1
    assert len(merged) == 3
    assert merged.loc[merged["timestamp"] == pd.Timestamp("2026-03-06T14:50:00Z"), "trade_count"].item() == 3


def test_resolve_fetch_start_bootstraps_missing_file(tmp_path: Path) -> None:
    now = datetime(2026, 3, 6, 15, 0, tzinfo=timezone.utc)
    start = resolve_fetch_start(
        tmp_path / "trainingdata5min" / "stocks" / "NVDA.csv",
        now=now,
        recent_minutes=15,
        overlap_minutes=10,
        bootstrap_days=3,
    )
    assert start == datetime(2026, 3, 3, 15, 0, tzinfo=timezone.utc)


def test_collect_recent_cycle_bootstraps_missing_symbols(tmp_path: Path) -> None:
    calls: list[tuple[str, datetime, datetime]] = []

    def _fake_fetch(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        calls.append((symbol, start, end))
        index = pd.to_datetime(["2026-03-06T14:50:00Z", "2026-03-06T14:55:00Z"], utc=True)
        frame = pd.DataFrame(
            {
                "open": [1.0, 1.1],
                "high": [1.1, 1.2],
                "low": [0.9, 1.0],
                "close": [1.05, 1.15],
                "volume": [10, 20],
                "trade_count": [1, 2],
                "vwap": [1.04, 1.14],
                "symbol": [symbol, symbol],
            },
            index=index,
        )
        frame.index.name = "timestamp"
        return frame

    now = datetime(2026, 3, 6, 15, 0, tzinfo=timezone.utc)
    results = collect_recent_cycle(
        symbols=["NVDA"],
        out_root=tmp_path / "trainingdata5min",
        now=now,
        recent_minutes=15,
        overlap_minutes=10,
        bootstrap_days=2,
        fetcher=_fake_fetch,
    )

    assert len(results) == 1
    assert calls[0][0] == "NVDA"
    assert calls[0][1] == datetime(2026, 3, 4, 15, 0, tzinfo=timezone.utc)
    assert calls[0][2] == now
    assert (tmp_path / "trainingdata5min" / "stocks" / "NVDA.csv").exists()
