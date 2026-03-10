from __future__ import annotations

import pandas as pd

from newnanoalpacahourlyexp.export_alpaca_initial_state import (
    reconstruct_cash_at_start,
    reconstruct_open_orders_at_start,
    reconstruct_positions_at_start,
)


def test_reconstruct_positions_and_cash_at_start_from_window_fills(monkeypatch) -> None:
    monkeypatch.setattr("newnanoalpacahourlyexp.export_alpaca_initial_state.get_fee_for_symbol", lambda _symbol: 0.0)

    orders = pd.DataFrame(
        [
            {
                "symbol": "ETHUSD",
                "filled_at": pd.Timestamp("2026-01-01T01:00:00Z"),
                "filled_qty": 1.0,
                "filled_price": 200.0,
                "side": "sell",
            }
        ]
    )
    window_start = pd.Timestamp("2026-01-01T00:00:00Z")
    window_end = pd.Timestamp("2026-01-01T02:00:00Z")

    positions = reconstruct_positions_at_start(
        current_positions={"ETHUSD": 0.5},
        orders=orders,
        window_start=window_start,
        window_end=window_end,
        symbols=["ETHUSD"],
    )
    cash = reconstruct_cash_at_start(
        current_cash=1000.0,
        orders=orders,
        window_start=window_start,
        window_end=window_end,
    )

    assert positions == {"ETHUSD": 1.5}
    assert cash == 800.0


def test_reconstruct_open_orders_at_start_merges_same_side_orders(monkeypatch) -> None:
    monkeypatch.setattr("newnanoalpacahourlyexp.export_alpaca_initial_state.get_fee_for_symbol", lambda _symbol: 0.0)

    window_start = pd.Timestamp("2026-01-01T00:00:00Z")
    orders = pd.DataFrame(
        [
            {
                "symbol": "ETH/USD",
                "created_at": pd.Timestamp("2025-12-31T23:00:00Z"),
                "filled_at": pd.NaT,
                "canceled_at": pd.Timestamp("2026-01-01T01:00:00Z"),
                "side": "buy",
                "qty": 1.0,
                "limit_price": 100.0,
            },
            {
                "symbol": "ETHUSD",
                "created_at": pd.Timestamp("2025-12-31T23:30:00Z"),
                "filled_at": pd.Timestamp("2026-01-01T02:00:00Z"),
                "canceled_at": pd.NaT,
                "side": "buy",
                "qty": 2.0,
                "limit_price": 110.0,
            },
            {
                "symbol": "ETHUSD",
                "created_at": pd.Timestamp("2025-12-31T23:15:00Z"),
                "filled_at": pd.NaT,
                "canceled_at": pd.NaT,
                "side": "sell",
                "qty": 0.5,
                "limit_price": 120.0,
            },
        ]
    )

    open_orders = reconstruct_open_orders_at_start(
        orders=orders,
        positions_at_start={"ETHUSD": 1.0},
        window_start=window_start,
        reserve_buy_notional=True,
    )

    assert len(open_orders) == 2
    buy_order = next(order for order in open_orders if order["side"] == "buy")
    sell_order = next(order for order in open_orders if order["side"] == "sell")

    assert buy_order["symbol"] == "ETHUSD"
    assert buy_order["kind"] == "entry"
    assert buy_order["quantity"] == 3.0
    assert buy_order["price"] == 106.66666666666667
    assert buy_order["reserved_cash"] == 320.0
    assert buy_order["created_at"] == "2025-12-31T23:00:00+00:00"

    assert sell_order["kind"] == "exit"
    assert sell_order["quantity"] == 0.5
