from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import build_stock_universe as cli_mod
from src.stock_universe_builder import is_candidate_stock_symbol, rank_stock_universe, summarize_daily_stock_csv


def _write_daily_csv(path: Path, *, close: float, volume: float, rows: int = 300) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="D", tz="UTC"),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
            "symbol": path.stem.upper(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_is_candidate_stock_symbol_filters_crypto_and_meta() -> None:
    assert is_candidate_stock_symbol("AAPL") is True
    assert is_candidate_stock_symbol("BTCUSD") is False
    assert is_candidate_stock_symbol("correlation_matrix") is False


def test_summarize_daily_stock_csv_requires_liquidity_and_history(tmp_path: Path) -> None:
    _write_daily_csv(tmp_path / "AAPL.csv", close=100.0, volume=2_000_000.0, rows=300)
    _write_daily_csv(tmp_path / "THIN.csv", close=2.0, volume=10_000.0, rows=300)
    _write_daily_csv(tmp_path / "NEW.csv", close=50.0, volume=2_000_000.0, rows=100)

    aapl = summarize_daily_stock_csv(tmp_path / "AAPL.csv")
    thin = summarize_daily_stock_csv(tmp_path / "THIN.csv")
    new = summarize_daily_stock_csv(tmp_path / "NEW.csv")

    assert aapl is not None
    assert aapl.symbol == "AAPL"
    assert aapl.median_dollar_volume == 200_000_000.0
    assert thin is None
    assert new is None


def test_summarize_daily_stock_csv_respects_min_last_timestamp(tmp_path: Path) -> None:
    _write_daily_csv(tmp_path / "AAPL.csv", close=100.0, volume=2_000_000.0, rows=300)
    fresh = summarize_daily_stock_csv(
        tmp_path / "AAPL.csv",
        min_last_timestamp="2025-10-01T00:00:00+00:00",
    )
    stale = summarize_daily_stock_csv(
        tmp_path / "AAPL.csv",
        min_last_timestamp="2026-12-01T00:00:00+00:00",
    )
    assert fresh is not None
    assert stale is None


def test_rank_stock_universe_prefers_liquidity_then_history(tmp_path: Path) -> None:
    _write_daily_csv(tmp_path / "AAPL.csv", close=100.0, volume=2_000_000.0, rows=400)
    _write_daily_csv(tmp_path / "MSFT.csv", close=100.0, volume=1_500_000.0, rows=500)
    _write_daily_csv(tmp_path / "SPY.csv", close=500.0, volume=1_000_000.0, rows=450)

    ranked = rank_stock_universe(sorted(tmp_path.glob("*.csv")), top_n=2)

    assert [candidate.symbol for candidate in ranked] == ["SPY", "AAPL"]


def test_cli_main_writes_symbol_file_and_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "train"
    _write_daily_csv(data_dir / "AAPL.csv", close=100.0, volume=2_000_000.0, rows=400)
    _write_daily_csv(data_dir / "MSFT.csv", close=100.0, volume=1_500_000.0, rows=500)
    _write_daily_csv(data_dir / "BTCUSD.csv", close=100000.0, volume=1_000.0, rows=500)
    output_file = tmp_path / "stocks.txt"
    json_out = tmp_path / "stocks.json"

    result = cli_mod.main(
        [
            "--data-dir",
            str(data_dir),
            "--top-n",
            "2",
            "--output-file",
            str(output_file),
            "--json-out",
            str(json_out),
        ]
    )

    assert result == 0
    assert output_file.read_text(encoding="utf-8").splitlines() == ["AAPL", "MSFT"]
    assert '"selected_count": 2' in json_out.read_text(encoding="utf-8")


def test_cli_main_applies_min_last_date_filter(tmp_path: Path) -> None:
    data_dir = tmp_path / "train"
    _write_daily_csv(data_dir / "AAPL.csv", close=100.0, volume=2_000_000.0, rows=400)
    output_file = tmp_path / "stocks.txt"
    json_out = tmp_path / "stocks.json"

    result = cli_mod.main(
        [
            "--data-dir",
            str(data_dir),
            "--top-n",
            "10",
            "--min-last-date",
            "2026-12-01",
            "--output-file",
            str(output_file),
            "--json-out",
            str(json_out),
        ]
    )

    assert result == 0
    assert output_file.read_text(encoding="utf-8") == ""
    assert '"selected_count": 0' in json_out.read_text(encoding="utf-8")
