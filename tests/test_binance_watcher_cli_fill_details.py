from __future__ import annotations

from binanceneural.binance_watcher_cli import _extract_fill_details


def test_extract_fill_details_uses_quote_over_limit_price():
    order = {
        "executedQty": "1.5",
        "cummulativeQuoteQty": "150.0",
        "price": "999.0",
    }
    details = _extract_fill_details(order)
    assert details is not None
    assert details["fill_qty"] == 1.5
    assert details["fill_quote"] == 150.0
    assert details["fill_price"] == 100.0


def test_extract_fill_details_falls_back_to_limit_price_when_quote_missing():
    order = {
        "executedQty": "2",
        "cummulativeQuoteQty": "0",
        "price": "50.0",
    }
    details = _extract_fill_details(order)
    assert details is not None
    assert details["fill_qty"] == 2.0
    assert details["fill_quote"] == 0.0
    assert details["fill_price"] == 50.0


def test_extract_fill_details_returns_none_for_zero_exec_qty():
    assert _extract_fill_details({"executedQty": "0", "cummulativeQuoteQty": "1", "price": "1"}) is None

