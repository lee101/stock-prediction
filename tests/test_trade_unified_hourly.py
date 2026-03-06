from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import unified_hourly_experiment.trade_unified_hourly as live


def test_log_event_writes_jsonl(tmp_path: Path, monkeypatch) -> None:
    event_log = tmp_path / "stock_event_log.jsonl"
    monkeypatch.setattr(live, "EVENT_LOG", event_log)

    live.log_event("unit_test_event", symbol="NVDA", payload={"qty": 5, "side": "buy"})

    rows = event_log.read_text().strip().splitlines()
    assert len(rows) == 1
    payload = json.loads(rows[0])
    assert payload["event_type"] == "unit_test_event"
    assert payload["symbol"] == "NVDA"
    assert payload["payload"] == {"qty": 5, "side": "buy"}
    assert "logged_at" in payload
    assert payload["pid"] > 0


def test_manage_positions_replaces_open_entry_order_with_protective_exit(monkeypatch) -> None:
    state = {
        "positions": {
            "NVDA": {
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "exit_price": 110.0,
                "hold_hours": 100.0,
            }
        }
    }
    calls: dict[str, object] = {"cancelled": [], "placed": []}

    monkeypatch.setattr(
        live,
        "get_current_positions",
        lambda api: {"NVDA": {"qty": 5.0, "price": 101.0}},
    )
    monkeypatch.setattr(
        live,
        "get_open_orders",
        lambda api: {"NVDA": [SimpleNamespace(id="entry-1", side="buy")]},
    )
    monkeypatch.setattr(
        live,
        "cancel_symbol_orders",
        lambda api, symbol, orders_by_symbol: calls["cancelled"].append(
            (symbol, [getattr(order, "id", None) for order in orders_by_symbol.get(symbol, [])])
        ),
    )
    monkeypatch.setattr(
        live,
        "place_exit_order",
        lambda api, symbol, qty, sell_price, side="sell": calls["placed"].append(
            (symbol, qty, sell_price, side)
        )
        or "exit-1",
    )

    live.manage_positions(object(), state, max_hold_hours=6, active_symbols={"NVDA"})

    assert calls["cancelled"] == [("NVDA", ["entry-1"])]
    assert calls["placed"] == [("NVDA", 5.0, 110.0, "sell")]
    assert state["positions"]["NVDA"]["exit_order_id"] == "exit-1"


def test_manage_positions_keeps_existing_protective_exit(monkeypatch) -> None:
    state = {
        "positions": {
            "NVDA": {
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "exit_price": 110.0,
                "hold_hours": 100.0,
            }
        }
    }
    calls: dict[str, object] = {"cancelled": [], "placed": []}

    monkeypatch.setattr(
        live,
        "get_current_positions",
        lambda api: {"NVDA": {"qty": 5.0, "price": 101.0}},
    )
    monkeypatch.setattr(
        live,
        "get_open_orders",
        lambda api: {"NVDA": [SimpleNamespace(id="exit-1", side="sell")]},
    )
    monkeypatch.setattr(
        live,
        "cancel_symbol_orders",
        lambda api, symbol, orders_by_symbol: calls["cancelled"].append(symbol),
    )
    monkeypatch.setattr(
        live,
        "place_exit_order",
        lambda api, symbol, qty, sell_price, side="sell": calls["placed"].append(
            (symbol, qty, sell_price, side)
        )
        or "exit-2",
    )

    live.manage_positions(object(), state, max_hold_hours=6, active_symbols={"NVDA"})

    assert calls["cancelled"] == []
    assert calls["placed"] == []


def test_execute_trades_emits_entry_lifecycle_events(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []
    state = {"positions": {}}

    fake_requests = types.ModuleType("alpaca.trading.requests")

    class _LimitOrderRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_requests.LimitOrderRequest = _LimitOrderRequest

    fake_enums = types.ModuleType("alpaca.trading.enums")
    fake_enums.OrderSide = SimpleNamespace(BUY="buy", SELL="sell")
    fake_enums.TimeInForce = SimpleNamespace(DAY="day")

    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", fake_enums)

    class _DummyAPI:
        def submit_order(self, order):
            return SimpleNamespace(id="entry-1")

    monkeypatch.setattr(
        live,
        "get_account_info",
        lambda api: {"equity": 10_000.0, "buying_power": 10_000.0, "cash": 5_000.0},
    )
    monkeypatch.setattr(
        live,
        "entry_intensity_fraction",
        lambda *args, **kwargs: (50.0, 0.5),
    )
    monkeypatch.setattr(live, "log_trade", lambda event: None)
    monkeypatch.setattr(live, "log_event", lambda event_type, **fields: events.append((event_type, fields)))

    live.execute_trades(
        _DummyAPI(),
        {
            "NVDA": {
                "buy_price": 100.0,
                "sell_price": 110.0,
                "buy_amount": 50.0,
                "sell_amount": 0.0,
                "edge": 0.02,
                "hold_hours": 4.0,
            }
        },
        state,
        max_positions=5,
    )

    event_types = [event_type for event_type, _ in events]
    assert "execute_trades_start" in event_types
    assert "entry_candidate" in event_types
    assert "entry_order_submit_requested" in event_types
    assert "entry_order_submit_succeeded" in event_types
    assert "position_tracking_created" in event_types
    assert "execute_trades_complete" in event_types
    assert state["positions"]["NVDA"]["entry_order_id"] == "entry-1"
