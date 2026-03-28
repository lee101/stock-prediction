from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_calendar_hours_between_counts_wall_clock_hours() -> None:
    start = datetime(2026, 3, 7, 23, 0, tzinfo=timezone.utc)
    end = datetime(2026, 3, 9, 5, 0, tzinfo=timezone.utc)

    assert live.calendar_hours_between(start, end) == pytest.approx(30.0)


def test_manage_positions_uses_calendar_hours_for_crypto(monkeypatch) -> None:
    now = datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc)
    entry_time = now - timedelta(hours=7)
    state = {
        "positions": {
            "BTCUSD": {
                "entry_time": entry_time.isoformat(),
                "exit_price": 110000.0,
                "hold_hours": 6.0,
            }
        }
    }
    forced: list[tuple[str, float, float]] = []
    events: list[tuple[str, dict]] = []

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz is None else now.astimezone(tz)

    monkeypatch.setattr(live, "datetime", FakeDateTime)
    monkeypatch.setattr(
        live,
        "get_current_positions",
        lambda api: {"BTCUSD": {"qty": 0.5, "price": 95_000.0}},
    )
    monkeypatch.setattr(live, "get_open_orders", lambda api: {})
    monkeypatch.setattr(live, "cancel_symbol_orders", lambda *args, **kwargs: None)
    monkeypatch.setattr(live.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        live,
        "force_close_position",
        lambda api, symbol, qty, current_price=0: forced.append((symbol, qty, current_price)),
    )
    monkeypatch.setattr(live, "log_trade", lambda event: None)
    monkeypatch.setattr(live, "log_event", lambda event_type, **fields: events.append((event_type, fields)))

    live.manage_positions(object(), state, max_hold_hours=6, active_symbols={"BTCUSD"})

    assert forced == [("BTCUSD", 0.5, 95_000.0)]
    assert any(
        event_type == "manage_position_action"
        and fields.get("reason") == "hold_timeout"
        and fields.get("hold_clock") == "calendar"
        for event_type, fields in events
    )


def test_manage_positions_keeps_market_hours_for_stocks(monkeypatch) -> None:
    now = datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc)
    entry_time = now - timedelta(hours=7)
    state = {
        "positions": {
            "AAPL": {
                "entry_time": entry_time.isoformat(),
                "exit_price": 210.0,
                "hold_hours": 6.0,
            }
        }
    }
    forced: list[tuple[str, float, float]] = []

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz is None else now.astimezone(tz)

    monkeypatch.setattr(live, "datetime", FakeDateTime)
    monkeypatch.setattr(
        live,
        "get_current_positions",
        lambda api: {"AAPL": {"qty": 5.0, "price": 205.0}},
    )
    monkeypatch.setattr(
        live,
        "get_open_orders",
        lambda api: {"AAPL": [SimpleNamespace(id="exit-1", side="sell")]},
    )
    monkeypatch.setattr(
        live,
        "force_close_position",
        lambda api, symbol, qty, current_price=0: forced.append((symbol, qty, current_price)),
    )
    monkeypatch.setattr(live, "cancel_symbol_orders", lambda *args, **kwargs: None)

    live.manage_positions(object(), state, max_hold_hours=6, active_symbols={"AAPL"})

    assert forced == []


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
    calls: dict[str, object] = {"placed": []}
    cancelled: list[str] = []

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
        "place_exit_order",
        lambda api, symbol, qty, sell_price, side="sell": calls["placed"].append(
            (symbol, qty, sell_price, side)
        )
        or "exit-1",
    )

    class _DummyAPI:
        def cancel_order_by_id(self, order_id):
            cancelled.append(str(order_id))

    live.manage_positions(_DummyAPI(), state, max_hold_hours=6, active_symbols={"NVDA"})

    assert cancelled == ["entry-1"]
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


def test_manage_positions_reconciles_exit_price_to_broker_entry_basis(monkeypatch) -> None:
    state = {
        "positions": {
            "PLTR": {
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "entry_price": 150.14,
                "exit_price": 150.74,
                "hold_hours": 6.0,
                "fee_rate": 0.001,
            }
        }
    }
    placed: list[tuple[str, float, float, str]] = []
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        live,
        "get_current_positions",
        lambda api: {"PLTR": {"qty": 107.0, "price": 152.38, "avg_entry_price": 152.78}},
    )
    monkeypatch.setattr(live, "get_open_orders", lambda api: {})
    monkeypatch.setattr(
        live,
        "place_exit_order",
        lambda api, symbol, qty, sell_price, side="sell": placed.append((symbol, qty, sell_price, side)) or "exit-1",
    )
    monkeypatch.setattr(live, "log_trade", lambda event: None)
    monkeypatch.setattr(live, "log_event", lambda event_type, **fields: events.append((event_type, fields)))

    live.manage_positions(object(), state, max_hold_hours=6, active_symbols={"PLTR"})

    assert len(placed) == 1
    symbol, qty, sell_price, side = placed[0]
    assert (symbol, qty, side) == ("PLTR", 107.0, "sell")
    assert sell_price == pytest.approx(152.93278)
    assert state["positions"]["PLTR"]["entry_price"] == 152.78
    assert state["positions"]["PLTR"]["exit_price"] == pytest.approx(152.93278)
    assert any(
        event_type == "manage_position_action" and fields.get("action") == "entry_price_reconciled"
        for event_type, fields in events
    )
    assert any(
        event_type == "manage_position_action" and fields.get("action") == "exit_price_reconciled_to_entry_basis"
        for event_type, fields in events
    )


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
    monkeypatch.setattr(live, "get_current_positions", lambda api: {})
    monkeypatch.setattr(live, "get_open_orders", lambda api: {})
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


def test_execute_trades_uses_short_side_for_short_only_symbol(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []
    submitted: list[object] = []
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
            submitted.append(order)
            return SimpleNamespace(id="entry-short-1")

    monkeypatch.setattr(
        live,
        "get_account_info",
        lambda api: {"equity": 10_000.0, "buying_power": 10_000.0, "cash": 5_000.0},
    )
    monkeypatch.setattr(live, "get_current_positions", lambda api: {})
    monkeypatch.setattr(live, "get_open_orders", lambda api: {})
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
            "DBX": {
                "buy_price": 20.0,
                "sell_price": 21.0,
                "buy_amount": 0.0,
                "sell_amount": 50.0,
                "edge": 0.03,
                "hold_hours": 4.0,
            }
        },
        state,
        max_positions=5,
    )

    assert submitted
    assert submitted[0].kwargs["side"] == "sell"
    assert state["positions"]["DBX"]["side"] == "short"
    assert state["positions"]["DBX"]["qty"] < 0


def test_execute_trades_uses_market_entry_reference_price_for_market_orders(monkeypatch) -> None:
    submitted: list[object] = []
    state = {"positions": {}}

    fake_requests = types.ModuleType("alpaca.trading.requests")

    class _MarketOrderRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _LimitOrderRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_requests.MarketOrderRequest = _MarketOrderRequest
    fake_requests.LimitOrderRequest = _LimitOrderRequest

    fake_enums = types.ModuleType("alpaca.trading.enums")
    fake_enums.OrderSide = SimpleNamespace(BUY="buy", SELL="sell")
    fake_enums.TimeInForce = SimpleNamespace(DAY="day")

    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", fake_enums)

    class _DummyAPI:
        def submit_order(self, order):
            submitted.append(order)
            return SimpleNamespace(id="entry-market-1")

    monkeypatch.setattr(
        live,
        "get_account_info",
        lambda api: {"equity": 10_000.0, "buying_power": 10_000.0, "cash": 5_000.0},
    )
    monkeypatch.setattr(live, "get_current_positions", lambda api: {})
    monkeypatch.setattr(live, "get_open_orders", lambda api: {})
    monkeypatch.setattr(live, "is_market_open_now", lambda: True)
    monkeypatch.setattr(
        live,
        "entry_intensity_fraction",
        lambda *args, **kwargs: (50.0, 0.5),
    )
    monkeypatch.setattr(live, "log_trade", lambda event: None)
    monkeypatch.setattr(live, "log_event", lambda *args, **kwargs: None)

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
                "market_entry_reference_price": 105.0,
                "market_entry_reference_source": "quote_ask",
            }
        },
        state,
        max_positions=5,
        market_order_entry=True,
    )

    assert submitted
    assert submitted[0].kwargs["qty"] == 51
    assert state["positions"]["NVDA"]["entry_price"] == 105.0


def test_execute_trades_concentrated_allocator_overweights_stronger_signal(monkeypatch) -> None:
    submitted: list[object] = []
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
            submitted.append(order)
            return SimpleNamespace(id=f"entry-{len(submitted)}")

    monkeypatch.setattr(
        live,
        "get_account_info",
        lambda api: {"equity": 10_000.0, "buying_power": 10_000.0, "cash": 5_000.0},
    )
    monkeypatch.setattr(live, "get_current_positions", lambda api: {})
    monkeypatch.setattr(live, "get_open_orders", lambda api: {})
    monkeypatch.setattr(
        live,
        "entry_intensity_fraction",
        lambda *args, **kwargs: (100.0, 1.0),
    )
    monkeypatch.setattr(live, "log_trade", lambda event: None)
    monkeypatch.setattr(live, "log_event", lambda *args, **kwargs: None)

    live.execute_trades(
        _DummyAPI(),
        {
            "AAA": {
                "buy_price": 100.0,
                "sell_price": 110.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "edge": 0.10,
                "hold_hours": 4.0,
            },
            "BBB": {
                "buy_price": 100.0,
                "sell_price": 106.0,
                "buy_amount": 100.0,
                "sell_amount": 0.0,
                "edge": 0.05,
                "hold_hours": 4.0,
            },
        },
        state,
        max_positions=5,
    )

    assert [order.kwargs["symbol"] for order in submitted] == ["AAA", "BBB"]
    assert [order.kwargs["qty"] for order in submitted] == [48, 42]


def test_place_exit_order_supports_fractional_stock_qty(monkeypatch) -> None:
    submitted: list[object] = []
    events: list[tuple[str, dict]] = []

    fake_requests = types.ModuleType("alpaca.trading.requests")

    class _LimitOrderRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_requests.LimitOrderRequest = _LimitOrderRequest

    fake_enums = types.ModuleType("alpaca.trading.enums")
    fake_enums.OrderSide = SimpleNamespace(BUY="buy", SELL="sell")
    fake_enums.TimeInForce = SimpleNamespace(DAY="day", GTC="gtc")

    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", fake_enums)
    monkeypatch.setattr(live, "log_event", lambda event_type, **fields: events.append((event_type, fields)))

    class _DummyAPI:
        def submit_order(self, order):
            submitted.append(order)
            return SimpleNamespace(id="exit-frac-1")

    order_id = live.place_exit_order(_DummyAPI(), "MSFT", 0.94, 397.37, side="sell")

    assert order_id == "exit-frac-1"
    assert submitted
    assert submitted[0].kwargs["qty"] == pytest.approx(0.94)
    assert submitted[0].kwargs["time_in_force"] == "day"
    assert any(event_type == "exit_order_submit_succeeded" for event_type, _ in events)


def test_force_close_position_supports_fractional_stock_qty(monkeypatch) -> None:
    submitted: list[object] = []
    events: list[tuple[str, dict]] = []

    fake_requests = types.ModuleType("alpaca.trading.requests")

    class _LimitOrderRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_requests.LimitOrderRequest = _LimitOrderRequest

    fake_enums = types.ModuleType("alpaca.trading.enums")
    fake_enums.OrderSide = SimpleNamespace(BUY="buy", SELL="sell")
    fake_enums.TimeInForce = SimpleNamespace(DAY="day", GTC="gtc")

    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", fake_enums)
    monkeypatch.setattr(live, "log_event", lambda event_type, **fields: events.append((event_type, fields)))
    monkeypatch.setattr(live, "log_trade", lambda payload: None)

    class _DummyAPI:
        def submit_order(self, order):
            submitted.append(order)
            return SimpleNamespace(id="force-close-frac-1")

    live.force_close_position(_DummyAPI(), "MSFT", 0.94, 397.1146)

    assert submitted
    assert submitted[0].kwargs["qty"] == pytest.approx(0.94)
    assert submitted[0].kwargs["time_in_force"] == "day"
    assert submitted[0].kwargs["limit_price"] == pytest.approx(round(397.1146 * 0.997, 2))
    assert any(event_type == "force_close_submit_succeeded" for event_type, _ in events)


def test_manage_positions_keeps_pending_entry_order(monkeypatch) -> None:
    state = {
        "positions": {
            "NVDA": {
                "qty": 5,
                "side": "long",
                "entry_price": 100.0,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "entry_order_id": "entry-1",
                "exit_price": 110.0,
                "hold_hours": 6.0,
            }
        }
    }
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(live, "get_current_positions", lambda api: {})
    monkeypatch.setattr(
        live,
        "get_open_orders",
        lambda api: {
            "NVDA": [
                SimpleNamespace(
                    id="entry-1",
                    side="buy",
                    qty="5",
                    limit_price="100.0",
                    created_at=datetime.now(timezone.utc),
                )
            ]
        },
    )
    monkeypatch.setattr(live, "log_trade", lambda event: None)
    monkeypatch.setattr(live, "log_event", lambda event_type, **fields: events.append((event_type, fields)))

    class _DummyAPI:
        def cancel_order_by_id(self, order_id):
            raise AssertionError("pending entry should not be cancelled")

    live.manage_positions(_DummyAPI(), state, max_hold_hours=6, active_symbols={"NVDA"})

    assert "NVDA" in state["positions"]
    assert any(event_type == "manage_position_action" and fields.get("action") == "pending_entry_present" for event_type, fields in events)


def test_execute_trades_skips_existing_matching_open_entry_order(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []
    submitted: list[object] = []
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
            submitted.append(order)
            return SimpleNamespace(id="should-not-submit")

        def cancel_order_by_id(self, order_id):
            raise AssertionError("matching order should not be cancelled")

    monkeypatch.setattr(live, "get_current_positions", lambda api: {})
    monkeypatch.setattr(
        live,
        "get_open_orders",
        lambda api: {
            "NVDA": [
                SimpleNamespace(
                    id="entry-1",
                    symbol="NVDA",
                    side="buy",
                    qty="54",
                    limit_price="100.0",
                    created_at=datetime.now(timezone.utc),
                )
            ]
        },
    )
    monkeypatch.setattr(
        live,
        "get_account_info",
        lambda api: {"equity": 10_000.0, "buying_power": 10_000.0, "cash": 5_000.0},
    )
    monkeypatch.setattr(live, "entry_intensity_fraction", lambda *args, **kwargs: (50.0, 0.5))
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

    assert submitted == []
    assert state["positions"]["NVDA"]["entry_order_id"] == "entry-1"
    assert any(event_type == "entry_skipped" and fields.get("reason") == "existing_open_entry_order" for event_type, fields in events)


def test_execute_trades_waits_for_stale_entry_cancel_before_replacing(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []
    submitted: list[object] = []
    cancelled: list[str] = []
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
            submitted.append(order)
            return SimpleNamespace(id="should-not-submit")

        def cancel_order_by_id(self, order_id):
            cancelled.append(str(order_id))

    monkeypatch.setattr(live, "get_current_positions", lambda api: {})
    monkeypatch.setattr(
        live,
        "get_open_orders",
        lambda api: {
            "NVDA": [
                SimpleNamespace(
                    id="entry-stale-1",
                    symbol="NVDA",
                    side="buy",
                    qty="54",
                    limit_price="101.0",
                    created_at=datetime.now(timezone.utc),
                )
            ]
        },
    )
    monkeypatch.setattr(
        live,
        "get_account_info",
        lambda api: {"equity": 10_000.0, "buying_power": 10_000.0, "cash": 5_000.0},
    )
    monkeypatch.setattr(live, "entry_intensity_fraction", lambda *args, **kwargs: (50.0, 0.5))
    monkeypatch.setattr(live, "_wait_for_entry_order_cancel_ack", lambda *args, **kwargs: False)
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

    assert submitted == []
    assert cancelled == ["entry-stale-1"]
    assert state["positions"] == {}
    assert any(event_type == "entry_skipped" and fields.get("reason") == "waiting_for_entry_order_cancel" for event_type, fields in events)


def test_execute_trades_replaces_stale_entry_after_cancel_ack(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []
    submitted: list[object] = []
    cancelled: list[str] = []
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
            submitted.append(order)
            return SimpleNamespace(id="new-entry-1")

        def cancel_order_by_id(self, order_id):
            cancelled.append(str(order_id))

    monkeypatch.setattr(live, "get_current_positions", lambda api: {})
    monkeypatch.setattr(
        live,
        "get_open_orders",
        lambda api: {
            "NVDA": [
                SimpleNamespace(
                    id="entry-stale-1",
                    symbol="NVDA",
                    side="buy",
                    qty="54",
                    limit_price="101.0",
                    created_at=datetime.now(timezone.utc),
                )
            ]
        },
    )
    monkeypatch.setattr(
        live,
        "get_account_info",
        lambda api: {"equity": 10_000.0, "buying_power": 10_000.0, "cash": 5_000.0},
    )
    monkeypatch.setattr(live, "entry_intensity_fraction", lambda *args, **kwargs: (50.0, 0.5))
    monkeypatch.setattr(live, "_wait_for_entry_order_cancel_ack", lambda *args, **kwargs: True)
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

    assert cancelled == ["entry-stale-1"]
    assert len(submitted) == 1
    assert state["positions"]["NVDA"]["entry_order_id"] == "new-entry-1"
    assert not any(event_type == "entry_skipped" and fields.get("reason") == "waiting_for_entry_order_cancel" for event_type, fields in events)


def test_execute_trades_blocks_replacement_when_entry_cancel_fails(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []
    submitted: list[object] = []
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
            submitted.append(order)
            return SimpleNamespace(id="should-not-submit")

        def cancel_order_by_id(self, order_id):
            raise RuntimeError("broker cancel failed")

    monkeypatch.setattr(live, "get_current_positions", lambda api: {})
    monkeypatch.setattr(
        live,
        "get_open_orders",
        lambda api: {
            "NVDA": [
                SimpleNamespace(
                    id="entry-stale-1",
                    symbol="NVDA",
                    side="buy",
                    qty="54",
                    limit_price="101.0",
                    created_at=datetime.now(timezone.utc),
                )
            ]
        },
    )
    monkeypatch.setattr(
        live,
        "get_account_info",
        lambda api: {"equity": 10_000.0, "buying_power": 10_000.0, "cash": 5_000.0},
    )
    monkeypatch.setattr(live, "entry_intensity_fraction", lambda *args, **kwargs: (50.0, 0.5))
    monkeypatch.setattr(live, "_wait_for_entry_order_cancel_ack", lambda *args, **kwargs: False)
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

    assert submitted == []
    assert state["positions"] == {}
    assert any(event_type == "order_cancel_failed" for event_type, _ in events)
    assert any(event_type == "entry_skipped" and fields.get("reason") == "waiting_for_entry_order_cancel" for event_type, fields in events)


def test_manage_positions_force_closes_untracked_fractional_stock_only(monkeypatch) -> None:
    state = {"positions": {}, "pending_close": []}
    forced: list[tuple[str, float, float]] = []

    monkeypatch.setattr(
        live,
        "get_current_positions",
        lambda api: {
            "MSFT": {"qty": 0.94, "price": 397.1146},
            "ETHUSD": {"qty": 1.8e-08, "price": 2236.3},
        },
    )
    monkeypatch.setattr(live, "get_open_orders", lambda api: {})
    monkeypatch.setattr(
        live,
        "force_close_position",
        lambda api, symbol, qty, current_price=0: forced.append((symbol, qty, current_price)),
    )
    monkeypatch.setattr(live, "log_event", lambda *args, **kwargs: None)

    live.manage_positions(object(), state, max_hold_hours=6, active_symbols={"NVDA", "PLTR"})

    assert forced == [("MSFT", 0.94, 397.1146)]
    assert state["pending_close"] == ["MSFT"]


def test_poll_broker_events_logs_and_dedupes(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []
    request_calls: list[tuple[str, dict]] = []
    now = datetime(2026, 3, 6, 15, 0, tzinfo=timezone.utc)
    order_ts = datetime(2026, 3, 6, 14, 30, tzinfo=timezone.utc)
    activity_ts = datetime(2026, 3, 6, 14, 45, tzinfo=timezone.utc)

    fake_requests = types.ModuleType("alpaca.trading.requests")

    class _GetOrdersRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_requests.GetOrdersRequest = _GetOrdersRequest

    fake_enums = types.ModuleType("alpaca.trading.enums")
    fake_enums.QueryOrderStatus = SimpleNamespace(CLOSED="closed")

    fake_wrapper = types.ModuleType("alpaca_wrapper")

    def _get_account_activities(api, **kwargs):
        request_calls.append(("activities", kwargs))
        return [
            {
                "id": "fill-1",
                "activity_type": "FILL",
                "symbol": "NVDA",
                "side": "buy",
                "qty": "5",
                "transaction_time": activity_ts.isoformat(),
            }
        ]

    fake_wrapper.get_account_activities = _get_account_activities

    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", fake_enums)
    monkeypatch.setitem(sys.modules, "alpaca_wrapper", fake_wrapper)
    monkeypatch.setattr(live, "log_event", lambda event_type, **fields: events.append((event_type, fields)))

    class _DummyAPI:
        def get_orders(self, request):
            request_calls.append(("orders", request.kwargs))
            return [
                SimpleNamespace(
                    id="order-1",
                    symbol="NVDA",
                    side="buy",
                    status="filled",
                    qty="5",
                    filled_qty="5",
                    filled_avg_price="101.25",
                    limit_price="101.0",
                    updated_at=order_ts,
                    filled_at=order_ts,
                )
            ]

    state: dict[str, object] = {}
    api = _DummyAPI()

    live.poll_broker_events(api, state, reason="pre_cycle", now=now, lookback_hours=24)

    event_types = [event_type for event_type, _ in events]
    assert "broker_event_poll_start" in event_types
    assert "broker_closed_order" in event_types
    assert "broker_fill_activity" in event_types
    assert "broker_event_poll_complete" in event_types
    assert request_calls[0] == (
        "orders",
        {
            "status": "closed",
            "after": datetime(2026, 3, 5, 15, 0, tzinfo=timezone.utc),
            "until": now,
            "direction": "asc",
            "limit": 500,
        },
    )
    assert request_calls[1] == (
        "activities",
        {
            "activity_types": ["FILL"],
            "date": "2026-03-05",
            "direction": "asc",
            "page_size": 100,
        },
    )
    assert request_calls[2] == (
        "activities",
        {
            "activity_types": ["FILL"],
            "date": "2026-03-06",
            "direction": "asc",
            "page_size": 100,
        },
    )
    broker_cursor = state["broker_event_cursor"]
    assert broker_cursor["closed_orders_after"] == order_ts.isoformat()
    assert broker_cursor["fill_activities_after"] == activity_ts.isoformat()
    assert broker_cursor["recent_order_event_keys"]
    assert broker_cursor["recent_activity_event_keys"]

    events.clear()
    live.poll_broker_events(api, state, reason="post_cycle", now=now, lookback_hours=24)

    event_types = [event_type for event_type, _ in events]
    assert "broker_event_poll_start" in event_types
    assert "broker_event_poll_complete" in event_types
    assert "broker_closed_order" not in event_types
    assert "broker_fill_activity" not in event_types


def test_run_cycle_polls_broker_events_pre_and_post_cycle(monkeypatch) -> None:
    poll_reasons: list[str] = []
    args = SimpleNamespace(
        ignore_market_hours=False,
        dry_run=False,
        min_edge=0.01,
        fee_rate=0.001,
        stock_data_root=Path("trainingdatahourly/stocks"),
        stock_cache_root=Path("unified_hourly_experiment/forecast_cache"),
    )

    monkeypatch.setattr(live, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(live, "poll_broker_events", lambda api, state, reason, **kwargs: poll_reasons.append(reason))
    monkeypatch.setattr(live, "manage_positions", lambda *args, **kwargs: 0)
    monkeypatch.setattr(live, "generate_signal_for_symbol", lambda *args, **kwargs: None)
    monkeypatch.setattr(live, "save_state", lambda state: None)
    monkeypatch.setattr(live, "is_market_open_now", lambda: False)

    live.run_cycle(
        model=object(),
        feature_columns=[],
        sequence_length=1,
        device=None,
        stocks=["NVDA"],
        args=args,
        api=object(),
        state={"positions": {}},
        max_positions=5,
        max_hold_hours=6,
        normalizer=None,
    )

    assert poll_reasons == ["pre_cycle", "post_cycle"]
