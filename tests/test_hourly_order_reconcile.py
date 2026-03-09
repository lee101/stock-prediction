from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.hourly_order_reconcile import orders_to_cancel_for_live_symbol
from src.hourly_trader_utils import OrderIntent


def _order(*, order_id: str, side: str, qty: float, limit_price: float, created_at: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=order_id,
        side=side,
        qty=qty,
        limit_price=limit_price,
        created_at=pd.Timestamp(created_at),
    )


def test_orders_to_cancel_for_live_symbol_cancels_stale_exit_price() -> None:
    intents = [OrderIntent(side="sell", qty=1.0, limit_price=110.0, kind="exit")]
    orders = [
        _order(order_id="matching", side="sell", qty=1.0, limit_price=110.0, created_at="2026-03-09T11:00:00Z"),
        _order(order_id="stale", side="sell", qty=1.0, limit_price=108.0, created_at="2026-03-09T10:00:00Z"),
    ]

    cancelled = orders_to_cancel_for_live_symbol(orders, position_qty=1.0, intents=intents)

    assert cancelled == [(orders[1], "stale_working_order")]


def test_orders_to_cancel_for_live_symbol_cancels_duplicate_matching_orders() -> None:
    intents = [OrderIntent(side="sell", qty=2.0, limit_price=110.0, kind="exit")]
    orders = [
        _order(order_id="newest", side="sell", qty=2.0, limit_price=110.0, created_at="2026-03-09T11:00:00Z"),
        _order(order_id="older", side="sell", qty=2.0, limit_price=110.0, created_at="2026-03-09T10:00:00Z"),
    ]

    cancelled = orders_to_cancel_for_live_symbol(orders, position_qty=2.0, intents=intents)

    assert cancelled == [(orders[1], "duplicate_working_order")]


def test_orders_to_cancel_for_live_symbol_keeps_similar_exit_qty_within_tolerance() -> None:
    intents = [OrderIntent(side="sell", qty=10.0, limit_price=50.0, kind="exit")]
    orders = [
        _order(order_id="matching", side="sell", qty=10.4, limit_price=50.0, created_at="2026-03-09T11:00:00Z"),
    ]

    cancelled = orders_to_cancel_for_live_symbol(orders, position_qty=10.0, intents=intents)

    assert cancelled == []
