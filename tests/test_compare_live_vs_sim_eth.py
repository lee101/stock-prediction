from __future__ import annotations

import pandas as pd

from fastalgorithms.eth_risk_ppo.compare_live_vs_sim_eth import _build_report


def test_compare_live_vs_sim_counts_fill_from_pre_window_order() -> None:
    window_start = pd.Timestamp("2026-03-05T08:00:00Z")
    window_end = pd.Timestamp("2026-03-06T20:00:00Z")

    sim = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-03-05T13:00:00Z"), "symbol": "ETHUSD", "side": "buy", "quantity": 1.0, "price": 2000.0},
            {"timestamp": pd.Timestamp("2026-03-06T21:00:00Z"), "symbol": "ETHUSD", "side": "sell", "quantity": 1.0, "price": 2100.0},
            {"timestamp": pd.Timestamp("2026-03-05T13:00:00Z"), "symbol": "BTCUSD", "side": "buy", "quantity": 1.0, "price": 90000.0},
        ]
    )
    live = pd.DataFrame(
        [
            {
                "id": "pre_window_fill",
                "symbol": "ETH/USD",
                "created_at": pd.Timestamp("2026-03-05T07:00:00Z"),
                "filled_at": pd.Timestamp("2026-03-06T13:55:52Z"),
                "canceled_at": pd.NaT,
                "side": "buy",
                "status": "filled",
                "qty": 5.0,
                "filled_qty": 5.0,
                "limit_price": 2029.88,
                "filled_price": 2029.88,
            },
            {
                "id": "still_open",
                "symbol": "ETH/USD",
                "created_at": pd.Timestamp("2026-03-05T17:00:07Z"),
                "filled_at": pd.NaT,
                "canceled_at": pd.NaT,
                "side": "buy",
                "status": "new",
                "qty": 6.0,
                "filled_qty": 0.0,
                "limit_price": 1928.73,
                "filled_price": 0.0,
            },
            {
                "id": "canceled_before_window",
                "symbol": "ETH/USD",
                "created_at": pd.Timestamp("2026-03-04T17:00:07Z"),
                "filled_at": pd.NaT,
                "canceled_at": pd.Timestamp("2026-03-05T07:30:00Z"),
                "side": "buy",
                "status": "canceled",
                "qty": 6.0,
                "filled_qty": 0.0,
                "limit_price": 1950.0,
                "filled_price": 0.0,
            },
        ]
    )

    report = _build_report(
        sim=sim[sim["symbol"] == "ETHUSD"],
        live=live,
        window_start=window_start,
        window_end=window_end,
    )

    assert report["summary"]["live_orders_total"] == 2
    assert report["summary"]["live_orders_fetched_total"] == 3
    assert report["summary"]["live_filled_total"] == 1
    assert report["summary"]["live_open_total"] == 1
    assert report["summary"]["sim_fills_total"] == 1
    assert report["live_filled_orders"][0]["id"] == "pre_window_fill"
    assert report["live_open_orders"][0]["id"] == "still_open"
