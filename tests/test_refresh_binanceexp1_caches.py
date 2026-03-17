from scripts.refresh_binanceexp1_caches import SYMBOLS, USDT_FALLBACK


def test_refresh_binanceexp1_caches_includes_live_alt_symbols() -> None:
    assert "LINKUSD" in SYMBOLS
    assert "UNIUSD" in SYMBOLS
    assert USDT_FALLBACK["LINKUSD"] == "LINKUSDT"
    assert USDT_FALLBACK["UNIUSD"] == "UNIUSDT"
