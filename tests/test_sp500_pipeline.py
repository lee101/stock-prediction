"""Unit tests for the S&P500 data download and export pipeline."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_sp500_data import (
    download_symbols,
    fetch_sp500_symbols_from_wikipedia,
    load_symbols_from_file,
    main as download_main,
    save_symbol_list,
)
from scripts.export_sp500_daily import (
    find_csv_symbols,
    group_into_batches,
    main as export_main,
)
from src.alpaca_stock_expansion import get_sp500_symbols


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_html_table(symbols: list[str]) -> list[pd.DataFrame]:
    """Return a list containing a single DataFrame that looks like the Wikipedia table."""
    return [pd.DataFrame({"Symbol": symbols, "Security": [f"Company {s}" for s in symbols]})]


def _write_daily_csv(path: Path, n_rows: int = 300) -> None:
    """Write a minimal daily OHLCV CSV with n_rows rows."""
    import numpy as np

    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": np.linspace(100, 110, n_rows),
            "high": np.linspace(101, 111, n_rows),
            "low": np.linspace(99, 109, n_rows),
            "close": np.linspace(100, 110, n_rows),
            "volume": np.full(n_rows, 1_000_000),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# get_sp500_symbols tests
# ---------------------------------------------------------------------------


def test_get_sp500_symbols_reads_cache_file(tmp_path):
    cache = tmp_path / "symbols.txt"
    cache.write_text("AAPL\nMSFT\nGOOGL\n", encoding="utf-8")
    symbols = get_sp500_symbols(use_cache=True, cache_file=str(cache))
    assert symbols == ["AAPL", "MSFT", "GOOGL"]


def test_get_sp500_symbols_skips_comments_in_cache(tmp_path):
    cache = tmp_path / "symbols.txt"
    cache.write_text("# comment\nAAPL\n\nMSFT\n", encoding="utf-8")
    symbols = get_sp500_symbols(use_cache=True, cache_file=str(cache))
    assert symbols == ["AAPL", "MSFT"]


def test_get_sp500_symbols_fetches_wikipedia_when_no_cache(tmp_path):
    cache = tmp_path / "symbols.txt"
    mock_tables = _make_mock_html_table(["AAPL", "MSFT", "GOOG"])
    with patch("src.alpaca_stock_expansion.pd.read_html", return_value=mock_tables) as mock_rh:
        symbols = get_sp500_symbols(use_cache=False, cache_file=str(cache))
    mock_rh.assert_called_once()
    assert symbols == ["AAPL", "MSFT", "GOOG"]
    # Should NOT have written cache
    assert not cache.exists()


def test_get_sp500_symbols_writes_cache_after_wikipedia_fetch(tmp_path):
    cache = tmp_path / "symbols.txt"
    mock_tables = _make_mock_html_table(["AAPL", "MSFT"])
    with patch("src.alpaca_stock_expansion.pd.read_html", return_value=mock_tables):
        symbols = get_sp500_symbols(use_cache=True, cache_file=str(cache))
    assert symbols == ["AAPL", "MSFT"]
    assert cache.exists()
    assert "AAPL" in cache.read_text()


def test_get_sp500_symbols_returns_list_of_strings(tmp_path):
    cache = tmp_path / "symbols.txt"
    mock_tables = _make_mock_html_table(["AAPL", "GOOG", "BRK-B"])
    with patch("src.alpaca_stock_expansion.pd.read_html", return_value=mock_tables):
        symbols = get_sp500_symbols(use_cache=True, cache_file=str(cache))
    assert all(isinstance(s, str) for s in symbols)
    assert len(symbols) == 3


# ---------------------------------------------------------------------------
# fetch_sp500_symbols_from_wikipedia tests
# ---------------------------------------------------------------------------


def test_fetch_sp500_symbols_from_wikipedia_returns_symbols():
    mock_tables = _make_mock_html_table(["AAPL", "MSFT", "AMZN", "TSLA"])
    with patch("scripts.download_sp500_data.pd.read_html", return_value=mock_tables):
        symbols = fetch_sp500_symbols_from_wikipedia()
    assert symbols == ["AAPL", "MSFT", "AMZN", "TSLA"]


def test_fetch_sp500_symbols_replaces_dots_with_dashes():
    mock_tables = _make_mock_html_table(["BRK.B", "BF.B"])
    with patch("scripts.download_sp500_data.pd.read_html", return_value=mock_tables):
        symbols = fetch_sp500_symbols_from_wikipedia()
    assert "BRK-B" in symbols
    assert "BF-B" in symbols


# ---------------------------------------------------------------------------
# load_symbols_from_file tests
# ---------------------------------------------------------------------------


def test_load_symbols_from_file_reads_one_per_line(tmp_path):
    f = tmp_path / "syms.txt"
    f.write_text("aapl\nMSFT\ngoogl\n", encoding="utf-8")
    assert load_symbols_from_file(str(f)) == ["AAPL", "MSFT", "GOOGL"]


def test_load_symbols_from_file_skips_blank_and_comments(tmp_path):
    f = tmp_path / "syms.txt"
    f.write_text("# this is a comment\n\nAAPL\n  \nMSFT\n", encoding="utf-8")
    assert load_symbols_from_file(str(f)) == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# save_symbol_list tests
# ---------------------------------------------------------------------------


def test_save_symbol_list_creates_file(tmp_path):
    save_symbol_list(["AAPL", "MSFT"], tmp_path)
    dest = tmp_path / "sp500_symbols.txt"
    assert dest.exists()
    content = dest.read_text()
    assert "AAPL" in content
    assert "MSFT" in content


# ---------------------------------------------------------------------------
# download_symbols dry-run tests
# ---------------------------------------------------------------------------


def test_download_symbols_dry_run_creates_no_files(tmp_path):
    symbols = ["AAPL", "MSFT", "GOOG"]
    downloaded, skipped = download_symbols(
        symbols,
        tmp_path,
        start_date="2024-01-01",
        end_date="2024-12-31",
        dry_run=True,
    )
    csv_files = list(tmp_path.glob("*.csv"))
    assert csv_files == [], "dry-run must not create files"
    assert downloaded == len(symbols)
    assert skipped == 0


def test_download_symbols_skips_existing_without_force(tmp_path):
    (tmp_path / "AAPL.csv").write_text("date,open,high,low,close,volume\n", encoding="utf-8")
    downloaded, skipped = download_symbols(
        ["AAPL"],
        tmp_path,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    assert skipped == 1
    assert downloaded == 0


def test_download_symbols_limit_via_main_dry_run(tmp_path, capsys):
    mock_tables = _make_mock_html_table([f"SYM{i}" for i in range(20)])
    with patch("scripts.download_sp500_data.pd.read_html", return_value=mock_tables):
        rc = download_main(["--output-dir", str(tmp_path), "--limit", "5", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Limiting to first 5 symbols" in out


def test_download_symbols_symbols_file_arg(tmp_path, capsys):
    syms_file = tmp_path / "my_symbols.txt"
    syms_file.write_text("AAPL\nMSFT\n", encoding="utf-8")

    mock_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "open": [150.0, 151.0],
            "high": [152.0, 153.0],
            "low": [149.0, 150.0],
            "close": [151.0, 152.0],
            "volume": [1000000, 1100000],
        }
    )
    # yfinance is imported lazily inside download_symbols as `import yfinance as yf`
    # so we patch yfinance.download at the module level
    with patch("yfinance.download", return_value=mock_df):
        rc = download_main(
            ["--output-dir", str(tmp_path), "--symbols-file", str(syms_file), "--force"]
        )
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 symbols" in out


# ---------------------------------------------------------------------------
# group_into_batches tests
# ---------------------------------------------------------------------------


def test_group_into_batches_503_symbols():
    symbols = [f"SYM{i}" for i in range(503)]
    batches = group_into_batches(symbols, 50)
    assert len(batches) == math.ceil(503 / 50)  # 11 batches
    assert sum(len(b) for b in batches) == 503
    for b in batches[:-1]:
        assert len(b) == 50
    assert len(batches[-1]) == 503 % 50  # 3


def test_group_into_batches_exact_multiple():
    symbols = [f"SYM{i}" for i in range(100)]
    batches = group_into_batches(symbols, 50)
    assert len(batches) == 2
    assert all(len(b) == 50 for b in batches)


def test_group_into_batches_single_symbol():
    batches = group_into_batches(["AAPL"], 50)
    assert batches == [["AAPL"]]


def test_group_into_batches_preserves_order():
    symbols = ["A", "B", "C", "D", "E"]
    batches = group_into_batches(symbols, 2)
    flattened = [s for batch in batches for s in batch]
    assert flattened == symbols


# ---------------------------------------------------------------------------
# find_csv_symbols tests
# ---------------------------------------------------------------------------


def test_find_csv_symbols_returns_stems(tmp_path):
    for name in ["AAPL", "MSFT", "GOOG"]:
        (tmp_path / f"{name}.csv").write_text("date,close\n", encoding="utf-8")
    symbols = find_csv_symbols(tmp_path)
    assert sorted(symbols) == ["AAPL", "GOOG", "MSFT"]


def test_find_csv_symbols_excludes_sp500_symbols_txt(tmp_path):
    (tmp_path / "AAPL.csv").write_text("date,close\n", encoding="utf-8")
    (tmp_path / "sp500_symbols.csv").write_text("symbols\n", encoding="utf-8")
    symbols = find_csv_symbols(tmp_path)
    assert "SP500_SYMBOLS" not in symbols
    assert "AAPL" in symbols


# ---------------------------------------------------------------------------
# export_main dry-run tests
# ---------------------------------------------------------------------------


def test_export_main_dry_run_no_files_written(tmp_path, capsys):
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    for sym in ["AAPL", "MSFT", "GOOG"]:
        _write_daily_csv(data_dir / f"{sym}.csv")

    output_dir = tmp_path / "output"
    rc = export_main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not output_dir.exists() or list(output_dir.glob("*.bin")) == []
    out = capsys.readouterr().out
    assert "dry-run" in out


def test_export_main_missing_data_dir_returns_1(tmp_path, capsys):
    rc = export_main(
        [
            "--data-dir",
            str(tmp_path / "nonexistent"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 1


def test_export_main_batch_size_capped_at_64(tmp_path, capsys):
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    (data_dir / "AAPL.csv").write_text("date,open,high,low,close,volume\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    rc = export_main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "100",
            "--dry-run",
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "capping to 64" in err


# ---------------------------------------------------------------------------
# Integration: export real binary for a small batch
# ---------------------------------------------------------------------------


def test_export_main_writes_bin_for_real_csv(tmp_path):
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    for sym in ["AAPL", "MSFT"]:
        _write_daily_csv(data_dir / f"{sym}.csv", n_rows=400)

    output_dir = tmp_path / "output"
    rc = export_main(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "5",
            "--val-split-date",
            "2021-01-01",
        ]
    )
    assert rc == 0
    bins = list(output_dir.glob("*.bin"))
    assert len(bins) >= 1, "Expected at least one .bin file"
