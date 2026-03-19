from __future__ import annotations

import pytest

from src.binance_symbol_utils import (
    forecast_cache_symbol_candidates,
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


def test_forecast_cache_symbol_candidates_prefers_usdt_for_usd_proxy() -> None:
    candidates = forecast_cache_symbol_candidates("BNBUSD")
    assert candidates[:3] == ["BNBUSD", "BNBUSDT", "BNBFDUSD"]


def test_forecast_cache_symbol_candidates_keeps_exact_symbol_first() -> None:
    candidates = forecast_cache_symbol_candidates("BNBFDUSD")
    assert candidates[:3] == ["BNBFDUSD", "BNBUSD", "BNBUSDT"]


def test_forecast_cache_symbol_candidates_supports_render_aliases() -> None:
    candidates = forecast_cache_symbol_candidates("RNDRUSD")
    assert candidates[:2] == ["RNDRUSD", "RNDRUSDT"]
    assert "RENDERUSD" in candidates
    assert "RENDERUSDT" in candidates
