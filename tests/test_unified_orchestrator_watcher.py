"""Tests for AlpacaCryptoWatcher intra-hour order management."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from unified_orchestrator.alpaca_watcher import AlpacaCryptoWatcher, OrderPair


class _FakeAlpacaClient:
    """Minimal fake for Alpaca TradingClient used by watcher tests."""

    def __init__(self, positions=None, open_orders=None):
        self.positions = list(positions or [])
        self._open_orders = list(open_orders or [])
        self.submitted_orders: list[SimpleNamespace] = []
        self.canceled_ids: list[str] = []
        self._next_id = 1

    def get_all_positions(self):
        return list(self.positions)

    def get_orders(self, request):
        return list(self._open_orders)

    def submit_order(self, request):
        oid = f"order-{self._next_id}"
        self._next_id += 1
        self.submitted_orders.append(request)
        result = SimpleNamespace(id=oid)
        # Track as open order
        self._open_orders.append(SimpleNamespace(
            id=oid,
            symbol=request.symbol,
            side=SimpleNamespace(value=request.side if isinstance(request.side, str) else "buy"),
            qty=request.qty,
            limit_price=request.limit_price,
        ))
        return result

    def cancel_order_by_id(self, order_id: str):
        self.canceled_ids.append(order_id)
        self._open_orders = [o for o in self._open_orders if str(o.id) != order_id]


@pytest.fixture(autouse=True)
def _patch_alpaca_imports(monkeypatch):
    """Patch Alpaca SDK imports used inside the watcher."""
    import alpaca.trading.enums as enums
    import alpaca.trading.requests as requests

    class _OrderSide:
        BUY = "buy"
        SELL = "sell"

    class _TimeInForce:
        GTC = "gtc"

    class _QueryOrderStatus:
        OPEN = "open"

    def _limit_order_request(**kwargs):
        return SimpleNamespace(**kwargs)

    def _get_orders_request(**kwargs):
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(enums, "OrderSide", _OrderSide, raising=False)
    monkeypatch.setattr(enums, "TimeInForce", _TimeInForce, raising=False)
    monkeypatch.setattr(enums, "QueryOrderStatus", _QueryOrderStatus, raising=False)
    monkeypatch.setattr(requests, "LimitOrderRequest", _limit_order_request, raising=False)
    monkeypatch.setattr(requests, "GetOrdersRequest", _get_orders_request, raising=False)


def test_watcher_places_sell_on_buy_fill():
    """When a buy order fills (disappears from open + position exists), watcher places TP sell."""
    client = _FakeAlpacaClient()
    pair = OrderPair(
        symbol="BTCUSD",
        buy_price=69000.0,
        sell_price=70000.0,
        target_qty=0.15,
        buy_order_id="buy-1",
    )

    watcher = AlpacaCryptoWatcher(
        alpaca_client=client,
        pairs=[pair],
        expiry_minutes=55,
        poll_interval=1,
    )

    # Simulate: buy order no longer in open orders + position exists
    client._open_orders = []  # buy-1 gone
    client.positions = [SimpleNamespace(symbol="BTC/USD", qty=0.15)]

    # Poll once
    watcher._start_time = time.time()
    watcher._poll_orders()

    # Should have placed a sell order
    assert len(client.submitted_orders) == 1
    sell = client.submitted_orders[0]
    assert sell.side == "sell"
    assert sell.limit_price == 70000.0
    assert sell.qty < 0.15  # floor trick
    assert pair.sell_order_id is not None
    assert len(watcher.fill_log) == 1
    assert watcher.fill_log[0]["side"] == "buy"


def test_watcher_oscillates_buy_sell():
    """After sell fills, watcher re-enters buy if time remains."""
    client = _FakeAlpacaClient()
    pair = OrderPair(
        symbol="ETHUSD",
        buy_price=2000.0,
        sell_price=2050.0,
        target_qty=5.0,
        current_qty=5.0,
        sell_order_id="sell-1",
        max_oscillations=3,
    )

    watcher = AlpacaCryptoWatcher(
        alpaca_client=client,
        pairs=[pair],
        expiry_minutes=55,
        poll_interval=1,
    )
    watcher._start_time = time.time()

    # Simulate: sell order gone + no position
    client._open_orders = []
    client.positions = []

    watcher._poll_orders()

    # Should have placed a re-buy
    assert pair.oscillations == 1
    assert len(client.submitted_orders) == 1
    buy = client.submitted_orders[0]
    assert buy.side == "buy"
    assert buy.limit_price == 2000.0
    assert pair.buy_order_id is not None
    assert len(watcher.fill_log) == 1
    assert watcher.fill_log[0]["side"] == "sell"


def test_watcher_stops_at_expiry():
    """Watcher should self-terminate when expired."""
    client = _FakeAlpacaClient()
    pair = OrderPair(
        symbol="SOLUSD",
        buy_price=100.0,
        sell_price=105.0,
        target_qty=10.0,
        buy_order_id="buy-active",
    )
    # Put the buy order in the open list so it doesn't look filled
    client._open_orders = [SimpleNamespace(
        id="buy-active", symbol="SOL/USD",
        side=SimpleNamespace(value="buy"), qty=10.0, limit_price=100.0,
    )]

    watcher = AlpacaCryptoWatcher(
        alpaca_client=client,
        pairs=[pair],
        expiry_minutes=0,  # Expire immediately
        poll_interval=1,
    )
    watcher._start_time = time.time() - 60  # Started 60s ago

    assert watcher.is_expired

    # Run should exit quickly and cancel open orders
    watcher.run()

    assert "buy-active" in client.canceled_ids


def test_watcher_does_not_reenter_at_max_oscillations():
    """After max oscillations, watcher should not place new buys."""
    client = _FakeAlpacaClient()
    pair = OrderPair(
        symbol="BTCUSD",
        buy_price=69000.0,
        sell_price=70000.0,
        target_qty=0.1,
        current_qty=0.1,
        sell_order_id="sell-final",
        oscillations=4,
        max_oscillations=5,
    )

    watcher = AlpacaCryptoWatcher(
        alpaca_client=client,
        pairs=[pair],
        expiry_minutes=55,
        poll_interval=1,
    )
    watcher._start_time = time.time()

    # Simulate sell fill
    client._open_orders = []
    client.positions = []

    watcher._poll_orders()

    # oscillation incremented to 5 (max), should NOT re-enter
    assert pair.oscillations == 5
    assert len(client.submitted_orders) == 0
    assert pair.buy_order_id is None


def test_watcher_cancel_all_on_stop():
    """Watcher cleanup cancels all open orders for its pairs."""
    client = _FakeAlpacaClient()
    pair = OrderPair(
        symbol="LTCUSD",
        buy_price=80.0,
        sell_price=85.0,
        target_qty=5.0,
        buy_order_id="buy-ltc",
        sell_order_id="sell-ltc",
    )

    watcher = AlpacaCryptoWatcher(
        alpaca_client=client,
        pairs=[pair],
        expiry_minutes=55,
    )

    watcher._cancel_all_open()

    assert "buy-ltc" in client.canceled_ids
    assert "sell-ltc" in client.canceled_ids
    assert pair.buy_order_id is None
    assert pair.sell_order_id is None


def test_watcher_dry_run_no_real_orders():
    """In dry run mode, watcher should not submit real orders."""
    client = _FakeAlpacaClient()
    pair = OrderPair(
        symbol="AVAXUSD",
        buy_price=30.0,
        sell_price=32.0,
        target_qty=10.0,
        buy_order_id="buy-dry",
    )

    watcher = AlpacaCryptoWatcher(
        alpaca_client=client,
        pairs=[pair],
        expiry_minutes=55,
        dry_run=True,
    )
    watcher._start_time = time.time()

    # Simulate buy fill
    client._open_orders = []
    client.positions = [SimpleNamespace(symbol="AVAX/USD", qty=10.0)]

    watcher._poll_orders()

    # Should NOT submit real order, but should track fill
    assert len(client.submitted_orders) == 0
    assert pair.sell_order_id.startswith("dry-")
    assert len(watcher.fill_log) == 1
