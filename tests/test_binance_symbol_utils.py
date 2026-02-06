from __future__ import annotations

import pytest

from src.binance_symbol_utils import (
    proxy_symbol_to_usd,
    split_stable_quote_symbol,
    stable_quote_aliases_from_usd,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BTCUSDT", ("BTC", "USDT")),
        ("BTC/USDT", ("BTC", "USDT")),
        ("btc_usdc", ("BTC", "USDC")),
        ("SOLFDUSD", ("SOL", "FDUSD")),
        ("ETHUSD", ("ETH", "USD")),
        ("AAPL", ("AAPL", "")),
        ("", ("", "")),
        ("   ", ("", "")),
    ],
)
def test_split_stable_quote_symbol(raw: str, expected: tuple[str, str]) -> None:
    assert split_stable_quote_symbol(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BTCUSDT", "BTCUSD"),
        ("BTC/USDT", "BTCUSD"),
        ("SOLFDUSD", "SOLUSD"),
        ("ETHUSD", "ETHUSD"),
        ("AAPL", "AAPL"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_proxy_symbol_to_usd(raw: str, expected: str) -> None:
    assert proxy_symbol_to_usd(raw) == expected


def test_stable_quote_aliases_from_usd() -> None:
    aliases = stable_quote_aliases_from_usd("BTCUSD")
    assert "BTCUSDT" in aliases
    assert "BTCUSDC" in aliases
    assert "BTCFDUSD" in aliases
    assert "BTCUSD" not in aliases


def test_stable_quote_aliases_requires_usd_quote() -> None:
    assert stable_quote_aliases_from_usd("AAPL") == []
    assert stable_quote_aliases_from_usd("BTCUSDT") == []

