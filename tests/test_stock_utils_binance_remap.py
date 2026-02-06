from __future__ import annotations

from src.stock_utils import binance_remap_symbols


def test_binance_remap_symbols_defaults_to_usdt(monkeypatch) -> None:
    monkeypatch.delenv("BINANCE_TLD", raising=False)
    monkeypatch.delenv("BINANCE_DEFAULT_QUOTE", raising=False)
    assert binance_remap_symbols("BTCUSD") == "BTCUSDT"


def test_binance_remap_symbols_respects_default_quote(monkeypatch) -> None:
    monkeypatch.delenv("BINANCE_TLD", raising=False)
    monkeypatch.setenv("BINANCE_DEFAULT_QUOTE", "FDUSD")
    assert binance_remap_symbols("BTCUSD") == "BTCFDUSD"


def test_binance_remap_symbols_keeps_stable_quote_pairs(monkeypatch) -> None:
    monkeypatch.delenv("BINANCE_TLD", raising=False)
    monkeypatch.setenv("BINANCE_DEFAULT_QUOTE", "FDUSD")
    assert binance_remap_symbols("SOLFDUSD") == "SOLFDUSD"
    assert binance_remap_symbols("ETHUSDC") == "ETHUSDC"


def test_binance_remap_symbols_keeps_usd_pairs_on_binance_us(monkeypatch) -> None:
    monkeypatch.setenv("BINANCE_TLD", "us")
    monkeypatch.setenv("BINANCE_DEFAULT_QUOTE", "FDUSD")
    assert binance_remap_symbols("BTCUSD") == "BTCUSD"


def test_binance_remap_symbols_ignores_stocks(monkeypatch) -> None:
    monkeypatch.delenv("BINANCE_TLD", raising=False)
    monkeypatch.setenv("BINANCE_DEFAULT_QUOTE", "FDUSD")
    assert binance_remap_symbols("AAPL") == "AAPL"

