from __future__ import annotations

import pytest

from src.alpaca_stock_universes import (
    available_stock_universe_names,
    merge_symbols_with_stock_universe,
    resolve_stock_universe,
)
from src.trade_directions import resolve_trade_directions


def test_available_stock_universe_names_include_new_broad_presets() -> None:
    names = available_stock_universe_names()
    assert "ai_long_short22" in names
    assert "software_short11" in names
    assert "stock19" in names
    assert "stock21_plus_pltr_nflx" in names


def test_resolve_stock_universe_software_short11_matches_requested_symbols() -> None:
    universe = resolve_stock_universe("software_short11")
    assert universe is not None
    assert universe.short_symbols == (
        "YELP",
        "EBAY",
        "TRIP",
        "MTCH",
        "KIND",
        "ANGI",
        "Z",
        "EXPE",
        "BKNG",
        "NWSA",
        "NYT",
    )
    assert universe.long_symbols == ()


def test_merge_symbols_with_stock_universe_appends_and_sets_direction_overrides() -> None:
    symbols, long_only, short_only = merge_symbols_with_stock_universe(
        base_symbols=["BTCUSD", "ETHUSD"],
        stock_universe="ai_long_short22",
        long_only_symbols=["SHOP"],
        short_only_symbols=["DBX"],
    )

    assert symbols[:2] == ["BTCUSD", "ETHUSD"]
    assert "NVDA" in symbols
    assert "YELP" in symbols
    assert "SHOP" in long_only
    assert "NVDA" in long_only
    assert "DBX" in short_only
    assert "KIND" in short_only

    nvda_dirs = resolve_trade_directions(
        "NVDA",
        allow_short=True,
        long_only_symbols=long_only,
        short_only_symbols=short_only,
        use_default_groups=False,
    )
    assert nvda_dirs.can_long is True
    assert nvda_dirs.can_short is False

    kind_dirs = resolve_trade_directions(
        "KIND",
        allow_short=True,
        long_only_symbols=long_only,
        short_only_symbols=short_only,
        use_default_groups=False,
    )
    assert kind_dirs.can_long is False
    assert kind_dirs.can_short is True


def test_merge_symbols_with_stock_universe_can_replace_base_symbols() -> None:
    symbols, long_only, short_only = merge_symbols_with_stock_universe(
        base_symbols=["SOLUSD", "BTCUSD"],
        stock_universe="stock19",
        universe_only=True,
    )

    assert "SOLUSD" not in symbols
    assert "BTCUSD" not in symbols
    assert "NVDA" in symbols
    assert "YELP" in symbols
    assert "NVDA" in long_only
    assert "YELP" in short_only


def test_resolve_stock_universe_stock21_plus_pltr_nflx_includes_extra_ai_longs() -> None:
    universe = resolve_stock_universe("stock21_plus_pltr_nflx")
    assert universe is not None
    assert "PLTR" in universe.long_symbols
    assert "NFLX" in universe.long_symbols
    assert "YELP" in universe.short_symbols
    assert "KIND" not in universe.short_symbols


def test_resolve_stock_universe_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        resolve_stock_universe("unknown_preset")
