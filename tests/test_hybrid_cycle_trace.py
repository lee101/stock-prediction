from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.binan import hybrid_cycle_trace as trace


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_cycle_snapshots_filters_window_and_live_mode(tmp_path: Path) -> None:
    log_path = tmp_path / f"{trace.TRACE_TAG}_20260313.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "event": "cycle_snapshot",
                "cycle_started_at": "2026-03-13T00:00:00+00:00",
                "mode": "live",
                "cycle_id": "live-a",
            },
            {
                "event": "cycle_snapshot",
                "cycle_started_at": "2026-03-13T01:00:00+00:00",
                "mode": "dry_run",
                "cycle_id": "dry-b",
            },
            {
                "event": "other",
                "cycle_started_at": "2026-03-13T02:00:00+00:00",
                "mode": "live",
                "cycle_id": "ignored",
            },
        ],
    )

    snapshots = trace.load_cycle_snapshots(
        log_dir=tmp_path,
        start="2026-03-13T00:30:00+00:00",
        end="2026-03-13T01:30:00+00:00",
        live_only=False,
    )
    assert [row["cycle_id"] for row in snapshots] == ["dry-b"]

    live_only = trace.load_cycle_snapshots(
        log_dir=tmp_path,
        start="2026-03-12T23:00:00+00:00",
        end="2026-03-13T01:30:00+00:00",
        live_only=True,
    )
    assert [row["cycle_id"] for row in live_only] == ["live-a"]


def test_extract_expected_orders_uses_placed_and_existing_working_actions() -> None:
    snapshots = [
        {
            "cycle_id": "c1",
            "cycle_started_at": "2026-03-13T00:00:00+00:00",
            "cycle_kind": "per_symbol",
            "mode": "live",
            "symbols_detail": [
                {
                    "symbol": "BTCUSD",
                    "market_symbol": "BTCUSDT",
                    "actions": [
                        {
                            "kind": "sell_take_profit",
                            "side": "SELL",
                            "status": "already_working",
                            "matched_open_orders": [
                                {
                                    "order_id": 101,
                                    "symbol": "BTCUSDT",
                                    "side": "SELL",
                                    "price": 71016.02,
                                    "orig_qty": 0.01473,
                                }
                            ],
                        },
                        {
                            "kind": "buy_entry",
                            "side": "BUY",
                            "status": "placed",
                            "placed_order": {
                                "order_id": 202,
                                "symbol": "BTCUSDT",
                                "side": "BUY",
                                "price": 70150.0,
                                "orig_qty": 0.01,
                            },
                        },
                    ],
                }
            ],
        }
    ]

    expected = trace.extract_expected_orders(snapshots)

    assert len(expected) == 2
    assert expected[0]["order_id"] == 101
    assert expected[0]["source"] == "already_working"
    assert expected[1]["order_id"] == 202
    assert expected[1]["source"] == "placed"


def test_match_expected_orders_matches_by_order_id_and_flags_unexpected() -> None:
    expected_orders = [
        {"cycle_id": "c1", "cycle_started_at": "2026-03-13T00:00:00+00:00", "symbol": "BTCUSDT", "side": "SELL", "order_id": 101, "price": 71016.02, "qty": 0.01473},
        {"cycle_id": "c2", "cycle_started_at": "2026-03-13T01:00:00+00:00", "symbol": "ETHUSDT", "side": "BUY", "order_id": 202, "price": 2050.0, "qty": 0.5},
    ]
    actual_orders = [
        {"orderId": 101, "symbol": "BTCUSDT", "side": "SELL", "status": "NEW", "price": "71016.02", "origQty": "0.01473", "executedQty": "0"},
        {"orderId": 303, "symbol": "SOLUSDT", "side": "BUY", "status": "NEW", "price": "85.0", "origQty": "10", "executedQty": "0"},
    ]

    payload = trace.match_expected_orders(expected_orders, actual_orders)

    assert payload["matched_count"] == 1
    assert payload["missing_count"] == 1
    assert len(payload["unexpected_orders"]) == 1
    assert payload["unexpected_orders"][0]["order_id"] == 303


def test_normalize_exchange_order_interprets_binance_millisecond_timestamps() -> None:
    normalized = trace.normalize_exchange_order(
        {
            "orderId": 101,
            "symbol": "BTCUSDT",
            "side": "SELL",
            "status": "NEW",
            "price": "71016.02",
            "origQty": "0.01473",
            "executedQty": "0",
            "time": 1741822201387,
            "updateTime": 1741822223014,
        }
    )

    assert normalized["time"] == "2025-03-12T23:30:01.387000+00:00"
    assert normalized["update_time"] == "2025-03-12T23:30:23.014000+00:00"


def test_order_price_touch_summary_and_touch_rollup() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-13T00:00:00+00:00",
                    "2026-03-13T00:05:00+00:00",
                    "2026-03-13T00:10:00+00:00",
                ]
            ),
            "open": [100.0, 100.5, 101.0],
            "high": [100.4, 101.3, 101.5],
            "low": [99.8, 100.2, 100.7],
            "close": [100.2, 101.0, 101.2],
            "volume": [1000.0, 1200.0, 1500.0],
        }
    )

    sell_touch = trace.order_price_touch_summary("SELL", 101.25, bars)
    buy_touch = trace.order_price_touch_summary("BUY", 99.7, bars)

    assert sell_touch["touched"] is True
    assert sell_touch["first_touch_ts"] == "2026-03-13T00:05:00+00:00"
    assert buy_touch["touched"] is False

    rollup = trace.summarize_touch_results(
        [
            {"touched": True, "filled": True},
            {"touched": False, "filled": False},
            {"touched": False, "filled": True},
        ]
    )
    assert rollup == {"touched": 1, "untouched": 2, "filled_without_touch": 1}
