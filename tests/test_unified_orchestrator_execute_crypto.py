from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from llm_hourly_trader.gemini_wrapper import TradePlan
from unified_orchestrator.orchestrator import execute_crypto_signals, _place_crypto_tp_sell
from unified_orchestrator.state import Position, UnifiedPortfolioSnapshot


@dataclass
class _FakeOrder:
    symbol: str
    side: SimpleNamespace
    qty: float
    limit_price: float
    id: str


class _FakeAlpacaClient:
    def __init__(
        self,
        *,
        cash: float,
        buying_power: float,
        positions: list[SimpleNamespace],
        open_orders: list[_FakeOrder],
    ) -> None:
        self.cash = cash
        self.buying_power = buying_power
        self.positions = list(positions)
        self.open_orders = list(open_orders)
        self.canceled_ids: list[str] = []
        self.submitted_orders: list[SimpleNamespace] = []

    def get_account(self):
        return SimpleNamespace(cash=self.cash, buying_power=self.buying_power)

    def get_all_positions(self):
        return list(self.positions)

    def get_orders(self, request):
        return list(self.open_orders)

    def cancel_order_by_id(self, order_id: str):
        self.canceled_ids.append(order_id)
        self.open_orders = [order for order in self.open_orders if order.id != order_id]

    def submit_order(self, request):
        self.submitted_orders.append(request)
        return SimpleNamespace(id=f"submitted-{len(self.submitted_orders)}")


@pytest.fixture(autouse=True)
def _patch_alpaca_order_types(monkeypatch):
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


def test_execute_crypto_signals_refreshes_cash_and_positions_after_cancel():
    snapshot = UnifiedPortfolioSnapshot(
        alpaca_cash=12_115.35,
        alpaca_buying_power=24_230.70,
        alpaca_positions={
            "ETHUSD": Position(
                symbol="ETHUSD",
                qty=6.900140951,
                avg_price=2060.0,
                current_price=2060.0,
                unrealized_pnl=0.0,
                broker="alpaca",
            )
        },
    )
    signals = {
        "BTCUSD": TradePlan(
            direction="long",
            buy_price=69_710.0,
            sell_price=70_755.0,
            confidence=0.50,
            reasoning="btc edge",
        ),
    }
    stale_buy = _FakeOrder(
        symbol="BTC/USD",
        side=SimpleNamespace(value="buy"),
        qty=0.20328129,
        limit_price=69_959.5,
        id="old-btc-buy",
    )
    client = _FakeAlpacaClient(
        cash=26_336.05,
        buying_power=52_672.10,
        positions=[],
        open_orders=[stale_buy],
    )

    orders = execute_crypto_signals(signals, snapshot, dry_run=False, alpaca_client=client)

    assert client.canceled_ids == ["old-btc-buy"]
    assert snapshot.alpaca_cash == pytest.approx(26_336.05)
    assert snapshot.alpaca_buying_power == pytest.approx(52_672.10)
    assert snapshot.alpaca_positions == {}
    assert len(client.submitted_orders) == 1
    submitted = client.submitted_orders[0]
    expected_trade_size = snapshot.alpaca_cash * 0.40
    assert submitted.limit_price == pytest.approx(69_710.0)
    assert submitted.qty == pytest.approx(expected_trade_size / 69_710.0)
    assert orders == [
        {
            "symbol": "BTCUSD",
            "action": "buy",
            "price": 69_710.0,
            "qty": pytest.approx(expected_trade_size / 69_710.0),
            "order_id": "submitted-1",
        }
    ]


def test_execute_crypto_signals_dry_run_does_not_cancel_open_orders():
    snapshot = UnifiedPortfolioSnapshot(
        alpaca_cash=1_000.0,
        alpaca_buying_power=2_000.0,
    )
    signals = {
        "BTCUSD": TradePlan(
            direction="long",
            buy_price=50_000.0,
            sell_price=50_500.0,
            confidence=0.60,
            reasoning="dry-run-only",
        ),
    }
    open_order = _FakeOrder(
        symbol="BTC/USD",
        side=SimpleNamespace(value="buy"),
        qty=0.02,
        limit_price=49_900.0,
        id="live-order",
    )
    client = _FakeAlpacaClient(
        cash=1_000.0,
        buying_power=2_000.0,
        positions=[],
        open_orders=[open_order],
    )

    orders = execute_crypto_signals(signals, snapshot, dry_run=True, alpaca_client=client)

    assert client.canceled_ids == []
    assert client.submitted_orders == []
    assert client.open_orders == [open_order]
    assert orders == [
        {
            "symbol": "BTCUSD",
            "action": "buy",
            "price": 50_000.0,
            "qty": pytest.approx(0.008),
            "dry_run": True,
        }
    ]


def test_tp_sell_placed_for_existing_position_at_max():
    """When a position is at max size, a TP sell should be placed."""
    snapshot = UnifiedPortfolioSnapshot(
        alpaca_cash=1_000.0,
        alpaca_buying_power=2_000.0,
        alpaca_positions={
            "ETHUSD": Position(
                symbol="ETHUSD",
                qty=7.657842454,
                avg_price=2112.0,
                current_price=2090.0,
                unrealized_pnl=-168.0,
                broker="alpaca",
            )
        },
    )
    signals = {
        "ETHUSD": TradePlan(
            direction="long",
            buy_price=2085.0,
            sell_price=2134.0,
            confidence=0.50,
            reasoning="eth edge",
        ),
    }
    client = _FakeAlpacaClient(
        cash=1_000.0,
        buying_power=2_000.0,
        positions=[
            SimpleNamespace(
                symbol="ETH/USD", qty=7.657842454,
                avg_entry_price=2112.0, current_price=2090.0,
                unrealized_pl=-168.0,
            )
        ],
        open_orders=[],
    )

    orders = execute_crypto_signals(signals, snapshot, dry_run=False, alpaca_client=client)

    # Should have placed a TP sell order
    sell_orders = [o for o in client.submitted_orders if o.side == "sell"]
    assert len(sell_orders) == 1
    assert sell_orders[0].limit_price == 2134.0
    # sell_qty should be slightly less than position qty (floor trick)
    import math
    expected_sell_qty = math.floor(7.657842454 * 1e8 - 1) / 1e8
    assert sell_orders[0].qty == pytest.approx(expected_sell_qty)
    # Orders list should include the TP sell
    tp_entries = [o for o in orders if o.get("action") == "sell_tp"]
    assert len(tp_entries) == 1
    assert tp_entries[0]["price"] == 2134.0


def test_tp_sell_cancels_old_sell_before_placing_new():
    """When updating TP sell price, old sell order should be canceled first."""
    snapshot = UnifiedPortfolioSnapshot(
        alpaca_cash=500.0,
        alpaca_buying_power=1_000.0,
        alpaca_positions={
            "BTCUSD": Position(
                symbol="BTCUSD",
                qty=0.085,
                avg_price=71000.0,
                current_price=71500.0,
                unrealized_pnl=42.5,
                broker="alpaca",
            )
        },
    )
    signals = {
        "BTCUSD": TradePlan(
            direction="long",
            buy_price=71400.0,
            sell_price=72000.0,
            confidence=0.60,
            reasoning="btc bullish",
        ),
    }
    old_sell = _FakeOrder(
        symbol="BTC/USD",
        side=SimpleNamespace(value="sell"),
        qty=0.08499999,
        limit_price=71800.0,
        id="old-tp-sell",
    )
    client = _FakeAlpacaClient(
        cash=500.0,
        buying_power=1_000.0,
        positions=[
            SimpleNamespace(
                symbol="BTC/USD", qty=0.085,
                avg_entry_price=71000.0, current_price=71500.0,
                unrealized_pl=42.5,
            )
        ],
        open_orders=[old_sell],
    )

    orders = execute_crypto_signals(signals, snapshot, dry_run=False, alpaca_client=client)

    # Old sell should be canceled
    assert "old-tp-sell" in client.canceled_ids
    # New TP sell should be placed at updated price
    sell_orders = [o for o in client.submitted_orders if o.side == "sell"]
    assert len(sell_orders) == 1
    assert sell_orders[0].limit_price == 72000.0


def test_tp_sell_not_placed_when_sell_price_below_current():
    """TP sell should NOT be placed if sell_price <= current_price."""
    snapshot = UnifiedPortfolioSnapshot(
        alpaca_cash=1_000.0,
        alpaca_buying_power=2_000.0,
        alpaca_positions={
            "SOLUSD": Position(
                symbol="SOLUSD",
                qty=100.0,
                avg_price=90.0,
                current_price=88.0,
                unrealized_pnl=-200.0,
                broker="alpaca",
            )
        },
    )
    signals = {
        "SOLUSD": TradePlan(
            direction="long",
            buy_price=87.5,
            sell_price=87.0,  # Below current!
            confidence=0.50,
            reasoning="sol edge",
        ),
    }
    client = _FakeAlpacaClient(
        cash=1_000.0,
        buying_power=2_000.0,
        positions=[
            SimpleNamespace(
                symbol="SOL/USD", qty=100.0,
                avg_entry_price=90.0, current_price=88.0,
                unrealized_pl=-200.0,
            )
        ],
        open_orders=[],
    )

    orders = execute_crypto_signals(signals, snapshot, dry_run=False, alpaca_client=client)

    sell_orders = [o for o in client.submitted_orders if getattr(o, 'side', None) == "sell"]
    assert len(sell_orders) == 0


def test_tp_sell_qty_avoids_insufficient_balance():
    """Sell qty must be strictly less than position qty to avoid rounding errors."""
    import math
    # Simulate the exact Alpaca scenario
    pos_qty = 7.657842454
    sell_qty = math.floor(pos_qty * 1e8 - 1) / 1e8
    assert sell_qty < pos_qty
    assert sell_qty > 0
    # Should be very close but strictly less
    assert pos_qty - sell_qty < 1e-7
    assert pos_qty - sell_qty > 0
