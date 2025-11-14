from __future__ import annotations

from pathlib import Path

import pytest

from strategytraining.symbol_sources import load_trade_stock_symbols


def test_load_trade_stock_symbols_reads_repo_default():
    symbols = load_trade_stock_symbols(Path("trade_stock_e2e.py"))
    assert symbols, "Expected at least one symbol from trade_stock_e2e.py"
    assert symbols[0] == "EQIX"
    assert {"BTCUSD", "ETHUSD", "UNIUSD"}.issubset(set(symbols))


def test_load_trade_stock_symbols_dedupes_and_normalizes(tmp_path: Path):
    script = tmp_path / "trade_stock_e2e.py"
    script.write_text(
        "def main():\n"
        "    symbols = [\n"
        "        'foo',\n"
        "        'BAR',\n"
        "        'foo',\n"
        "    ]\n"
    )

    symbols = load_trade_stock_symbols(script)
    assert symbols == ["FOO", "BAR"]


def test_load_trade_stock_symbols_missing_file(tmp_path: Path):
    missing = tmp_path / "missing_trade_stock.py"
    with pytest.raises(FileNotFoundError):
        load_trade_stock_symbols(missing)
