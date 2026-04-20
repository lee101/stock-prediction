"""Tests for BRK-B / BF-B → BRK.B / BF.B ticker translation on the trading path.

Our symbol lists (and XGB training data) use dash-form tickers like ``BRK-B``
and ``BF-B`` because the training-data CSV filenames store them that way.
Alpaca's trading and data APIs only accept dot-form (``BRK.B``, ``BF.B``).

If the XGB ensemble picks one of these tickers as top_n=1, the pre-fix
trading submit path sent the raw dash form straight to Alpaca, which is
rejected at the broker and crashes the session.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.live_trader import _submit_market_order, _to_alpaca_symbol  # noqa: E402


def test_to_alpaca_symbol_translates_brk_and_bf() -> None:
    assert _to_alpaca_symbol("BRK-B") == "BRK.B"
    assert _to_alpaca_symbol("BF-B") == "BF.B"


def test_to_alpaca_symbol_passthrough_plain_ticker() -> None:
    assert _to_alpaca_symbol("AAPL") == "AAPL"
    assert _to_alpaca_symbol("MSFT") == "MSFT"


def test_to_alpaca_symbol_handles_empty() -> None:
    assert _to_alpaca_symbol("") == ""


def _capture_market_order_symbol(symbol: str, qty: float, side: str) -> tuple[str, float]:
    """Run _submit_market_order and capture what MarketOrderRequest was called with."""
    client = MagicMock()
    captured: dict = {}

    def _fake_req(**kw):
        captured.update(kw)
        return MagicMock()

    with patch("alpaca.trading.requests.MarketOrderRequest", side_effect=_fake_req):
        _submit_market_order(client, symbol=symbol, qty=qty, side=side)
    return captured["symbol"], captured["qty"]


def test_submit_market_order_translates_dash_to_dot_for_buy() -> None:
    sym, qty = _capture_market_order_symbol("BRK-B", 1.25, "buy")
    assert sym == "BRK.B"
    assert qty == 1.25


def test_submit_market_order_translates_dash_to_dot_for_sell() -> None:
    sym, _ = _capture_market_order_symbol("BF-B", 2.5, "sell")
    assert sym == "BF.B"


def test_submit_market_order_passthrough_plain_ticker() -> None:
    sym, qty = _capture_market_order_symbol("AAPL", 0.5, "buy")
    assert sym == "AAPL"
    assert qty == 0.5


def test_submit_market_order_rounds_qty_to_4dp() -> None:
    _, qty = _capture_market_order_symbol("MSFT", 1.23456789, "buy")
    assert qty == 1.2346
