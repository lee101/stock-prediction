from __future__ import annotations

from types import SimpleNamespace

from cancel_multi_orders import cancel_duplicate_opening_orders


def test_cancel_duplicate_opening_orders_only_cancels_flat_entry_duplicates() -> None:
    orders = [
        SimpleNamespace(id="eth-buy-1", symbol="ETH/USD", side="buy", created_at=1),
        SimpleNamespace(id="eth-buy-2", symbol="ETHUSD", side="buy", created_at=2),
        SimpleNamespace(id="eth-sell-1", symbol="ETHUSD", side="sell", created_at=3),
        SimpleNamespace(id="nvda-sell-1", symbol="NVDA", side="sell", created_at=4),
        SimpleNamespace(id="nvda-sell-2", symbol="NVDA", side="sell", created_at=5),
    ]
    positions = [
        SimpleNamespace(symbol="NVDA", qty="5", current_price="900.0"),
    ]
    cancelled: list[str] = []

    result = cancel_duplicate_opening_orders(
        orders,
        positions,
        cancel_order_fn=lambda order: cancelled.append(order.id),
    )

    assert result == ["eth-buy-1"]
    assert cancelled == ["eth-buy-1"]


def test_cancel_duplicate_opening_orders_treats_crypto_dust_as_flat() -> None:
    orders = [
        SimpleNamespace(id="eth-buy-1", symbol="ETHUSD", side="buy", created_at=1),
        SimpleNamespace(id="eth-buy-2", symbol="ETHUSD", side="buy", created_at=2),
    ]
    positions = [
        SimpleNamespace(symbol="ETHUSD", qty="0.000000001", current_price="2500.0"),
    ]
    cancelled: list[str] = []

    result = cancel_duplicate_opening_orders(
        orders,
        positions,
        cancel_order_fn=lambda order: cancelled.append(order.id),
    )

    assert result == ["eth-buy-1"]
    assert cancelled == ["eth-buy-1"]


def test_cancel_duplicate_opening_orders_keeps_duplicate_exit_orders_for_live_position() -> None:
    orders = [
        SimpleNamespace(id="nvda-sell-1", symbol="NVDA", side="sell", created_at=1),
        SimpleNamespace(id="nvda-sell-2", symbol="NVDA", side="sell", created_at=2),
    ]
    positions = [
        SimpleNamespace(symbol="NVDA", qty="5", current_price="900.0"),
    ]
    cancelled: list[str] = []

    result = cancel_duplicate_opening_orders(
        orders,
        positions,
        cancel_order_fn=lambda order: cancelled.append(order.id),
    )

    assert result == []
    assert cancelled == []
