from __future__ import annotations

from pathlib import Path

from src.symbol_file_utils import (
    SymbolFileOrdering,
    load_symbols_from_file,
    parse_symbols_text,
)


def test_parse_symbols_text_supports_comments_commas_and_custom_normalizer() -> None:
    raw_text = "aapl, msft\n# ignore\n brk.b \n"

    values = parse_symbols_text(raw_text, normalize_symbol=lambda symbol: symbol.strip().upper())

    assert values == ["AAPL", "MSFT", "BRK.B"]


def test_load_symbols_from_file_dedupes_by_default(tmp_path: Path) -> None:
    path = tmp_path / "symbols.txt"
    path.write_text("aapl, msft\nAAPL\nnvda\n", encoding="utf-8")

    assert load_symbols_from_file(path) == ["AAPL", "MSFT", "NVDA"]


def test_load_symbols_from_file_can_preserve_duplicates_and_sort_output(tmp_path: Path) -> None:
    path = tmp_path / "symbols.txt"
    path.write_text("msft\naapl\nMSFT\n", encoding="utf-8")

    assert load_symbols_from_file(path, dedupe=False) == ["MSFT", "AAPL", "MSFT"]
    assert load_symbols_from_file(path, ordering=SymbolFileOrdering.SORTED) == ["AAPL", "MSFT"]
