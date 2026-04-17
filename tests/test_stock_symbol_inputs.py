from __future__ import annotations

from pathlib import Path

import pytest

from src.stock_symbol_inputs import (
    load_symbols_file,
    normalize_stock_symbol,
    normalize_stock_symbol_list,
    normalize_symbols,
)


def test_normalize_stock_symbol_accepts_standard_share_class_symbols() -> None:
    assert normalize_stock_symbol(" brk.b ") == "BRK.B"
    assert normalize_stock_symbol("bf-b") == "BF-B"


def test_normalize_stock_symbol_rejects_path_like_input() -> None:
    with pytest.raises(ValueError, match=r"Unsupported symbol: \.\./AAPL"):
        normalize_stock_symbol("../AAPL")


def test_normalize_symbols_reports_duplicates_and_blanks() -> None:
    normalized, removed_duplicates, ignored_inputs = normalize_symbols(
        [" aapl ", "MSFT", "aapl", "", "  "]
    )

    assert normalized == ["AAPL", "MSFT"]
    assert removed_duplicates == ["AAPL"]
    assert ignored_inputs == ["<blank>", "<blank>"]


def test_normalize_stock_symbol_list_preserves_order() -> None:
    assert normalize_stock_symbol_list(["msft", "aapl"]) == ["MSFT", "AAPL"]


def test_normalize_stock_symbol_list_treats_string_as_single_symbol() -> None:
    assert normalize_stock_symbol_list(" msft ") == ["MSFT"]


def test_load_symbols_file_supports_comments_and_commas(tmp_path: Path) -> None:
    path = tmp_path / "symbols.txt"
    path.write_text("aapl, msft\n# comment\nnvda\n", encoding="utf-8")

    assert load_symbols_file(path) == ["AAPL", "MSFT", "NVDA"]


def test_load_symbols_file_rejects_comment_only_files(tmp_path: Path) -> None:
    path = tmp_path / "symbols.txt"
    path.write_text("# comment only\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"No valid symbols found in {path}"):
        load_symbols_file(path)


def test_normalize_symbols_treats_string_as_single_symbol() -> None:
    normalized, removed_duplicates, ignored_inputs = normalize_symbols(" aapl ")

    assert normalized == ["AAPL"]
    assert removed_duplicates == []
    assert ignored_inputs == []
