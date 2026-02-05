from __future__ import annotations

from alpacaconstrainedexp.symbols import (
    DEFAULT_LONG_CRYPTO,
    LONGABLE_STOCKS,
    SHORTABLE_STOCKS,
    build_longable_symbols,
    build_shortable_symbols,
    normalize_symbols,
    split_symbols_by_constraint,
)


def test_normalize_symbols_dedup():
    raw = ["nvda", "NVDA", "", " msft ", "BTCUSD", "btcusd"]
    assert normalize_symbols(raw) == ["NVDA", "MSFT", "BTCUSD"]


def test_build_longable_defaults():
    symbols = build_longable_symbols()
    for sym in LONGABLE_STOCKS:
        assert sym in symbols
    for sym in DEFAULT_LONG_CRYPTO:
        assert sym in symbols


def test_build_shortable_defaults():
    symbols = build_shortable_symbols()
    for sym in SHORTABLE_STOCKS:
        assert sym in symbols


def test_split_symbols_by_constraint():
    symbols = ["BTCUSD", "NVDA", "YELP", "MSFT", "ANGI", "SOLUSD"]
    longable, shortable = split_symbols_by_constraint(symbols)
    assert "BTCUSD" in longable
    assert "SOLUSD" in longable
    assert "NVDA" in longable
    assert "MSFT" in longable
    assert "YELP" in shortable
    assert "ANGI" in shortable
    assert "YELP" not in longable
