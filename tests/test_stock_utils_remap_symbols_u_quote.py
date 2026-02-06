from __future__ import annotations

import pytest

from src.stock_utils import remap_symbols, unmap_symbols


@pytest.mark.parametrize("symbol", ["MU", "LULU", "BIDU"])
def test_remap_symbols_does_not_split_stock_tickers_ending_with_u(symbol: str) -> None:
    # "U" is a Binance stable-quote, but many stock tickers end with the same letter.
    assert remap_symbols(symbol) == symbol


def test_remap_symbols_splits_known_u_quote_crypto_pairs() -> None:
    assert remap_symbols("BTCU") == "BTC/U"
    assert remap_symbols("ETHU") == "ETH/U"
    assert remap_symbols("SOLU") == "SOL/U"
    assert remap_symbols("BNBU") == "BNB/U"


def test_unmap_symbols_does_not_collapse_unknown_u_pairs() -> None:
    # If something generated "M/U", keep it as-is; do not silently convert to "MU".
    assert unmap_symbols("M/U") == "M/U"


def test_unmap_symbols_round_trips_known_u_quote_crypto_pairs() -> None:
    assert unmap_symbols("BTC/U") == "BTCU"
