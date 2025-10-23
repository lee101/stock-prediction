from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import trade_stock_e2e


def make_position(
    symbol: str,
    qty: float,
    price: float,
    side: str = "long",
    unrealized_pl: float = 0.0,
) -> SimpleNamespace:
    market_value = qty * price
    return SimpleNamespace(
        symbol=symbol,
        qty=qty,
        current_price=price,
        side=side,
        market_value=market_value,
        unrealized_pl=unrealized_pl,
    )


def test_describe_probe_state_transition_ready():
    now = datetime(2025, 10, 15, 14, 0, tzinfo=timezone.utc)
    started = datetime(2025, 10, 14, 14, 30, tzinfo=timezone.utc)

    summary = trade_stock_e2e._describe_probe_state(
        {"probe_active": True, "probe_started_at": started.isoformat()},
        now=now,
    )

    assert summary["probe_transition_ready"] is True
    assert summary["probe_expired"] is False
    assert summary["probe_started_at"] == started.isoformat()
    assert summary["probe_expires_at"] == (started + trade_stock_e2e.PROBE_MAX_DURATION).isoformat()
    assert summary["probe_age_seconds"] == pytest.approx((now - started).total_seconds())


def test_describe_probe_state_expired():
    now = datetime(2025, 10, 15, 16, 0, tzinfo=timezone.utc)
    started = now - trade_stock_e2e.PROBE_MAX_DURATION - timedelta(minutes=1)

    summary = trade_stock_e2e._describe_probe_state(
        {"probe_active": True, "probe_started_at": started.isoformat()},
        now=now,
    )

    assert summary["probe_expired"] is True
    assert summary["probe_transition_ready"] is True  # expiry implies readiness


def test_describe_probe_state_inactive():
    now = datetime.now(timezone.utc)
    summary = trade_stock_e2e._describe_probe_state({}, now=now)
    assert summary["probe_transition_ready"] is False
    assert summary["probe_expired"] is False


def test_manage_positions_promotes_probe(monkeypatch):
    module = trade_stock_e2e
    symbol = "TEST"

    positions = [make_position(symbol, qty=1.0, price=10.0, side="long")]
    module.alpaca_wrapper.equity = 1000.0

    monkeypatch.setattr(module.alpaca_wrapper, "get_all_positions", lambda: positions)
    monkeypatch.setattr(module, "filter_to_realistic_positions", lambda pos: pos)
    monkeypatch.setattr(module, "_handle_live_drawdown", lambda *_: None)
    monkeypatch.setattr(module, "is_nyse_trading_day_now", lambda: True)
    monkeypatch.setattr(module, "is_nyse_trading_day_ending", lambda: True)

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(module, "StockHistoricalDataClient", DummyClient)
    monkeypatch.setattr(module, "download_exchange_latest_data", lambda client, sym: None)
    monkeypatch.setattr(module, "get_bid", lambda sym: 9.5)
    monkeypatch.setattr(module, "get_ask", lambda sym: 10.0)
    monkeypatch.setattr(module, "get_qty", lambda sym, price, _positions: 5.0)
    monkeypatch.setattr(module, "spawn_close_position_at_takeprofit", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "backout_near_market", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.alpaca_wrapper, "open_order_at_price_or_all", lambda *args, **kwargs: None)

    ramp_calls = []
    monkeypatch.setattr(
        module,
        "ramp_into_position",
        lambda sym, side, target_qty=None: ramp_calls.append((sym, side, target_qty)),
    )

    transition_calls = []
    monkeypatch.setattr(
        module,
        "_mark_probe_transitioned",
        lambda sym, side, qty: (transition_calls.append((sym, side, qty)) or {}),
    )

    probe_active_calls = []
    monkeypatch.setattr(
        module,
        "_mark_probe_active",
        lambda sym, side, qty: (probe_active_calls.append((sym, side, qty)) or {}),
    )

    active_trade_updates = []
    monkeypatch.setattr(
        module,
        "_update_active_trade",
        lambda sym, side, mode, qty, strategy=None: active_trade_updates.append(
            (sym, side, mode, qty, strategy)
        ),
    )

    monkeypatch.setattr(module, "_mark_probe_pending", lambda sym, side: {})
    monkeypatch.setattr(
        module,
        "record_portfolio_snapshot",
        lambda total_value, observed_at=None: SimpleNamespace(
            observed_at=datetime.now(timezone.utc),
            portfolio_value=total_value,
            risk_threshold=1.0,
        ),
    )

    current_pick = {
        "trade_mode": "probe",
        "probe_transition_ready": True,
        "probe_expired": False,
        "side": "buy",
        "strategy": "simple",
        "predicted_high": 12.0,
        "predicted_low": 8.0,
        "trade_blocked": False,
        "pending_probe": False,
        "probe_active": True,
        "predicted_movement": 1.0,
        "composite_score": 1.0,
    }
    current_picks = {symbol: current_pick}
    analyzed_results = {symbol: copy.deepcopy(current_pick)}

    module.manage_positions(current_picks, previous_picks={}, all_analyzed_results=analyzed_results)

    assert len(transition_calls) == 1
    trans_symbol, trans_side, trans_qty = transition_calls[0]
    assert (trans_symbol, trans_side) == (symbol, "buy")
    assert trans_qty == pytest.approx(5.0)
    assert probe_active_calls == []
    assert len(active_trade_updates) >= 1
    act_symbol, act_side, act_mode, act_qty = active_trade_updates[-1]
    assert (act_symbol, act_side, act_mode) == (symbol, "buy", "probe_transition")
    assert act_qty == pytest.approx(5.0)
    assert len(ramp_calls) == 1
    ramp_symbol, ramp_side, ramp_qty = ramp_calls[0]
    assert (ramp_symbol, ramp_side) == (symbol, "buy")
    assert ramp_qty == pytest.approx(5.0)


def test_manage_positions_backouts_expired_probe(monkeypatch):
    module = trade_stock_e2e
    symbol = "TEST"

    positions = [make_position(symbol, qty=1.0, price=10.0, side="long")]
    module.alpaca_wrapper.equity = 1000.0

    monkeypatch.setattr(module.alpaca_wrapper, "get_all_positions", lambda: positions)
    monkeypatch.setattr(module, "filter_to_realistic_positions", lambda pos: pos)
    monkeypatch.setattr(module, "_handle_live_drawdown", lambda *_: None)
    monkeypatch.setattr(module, "is_nyse_trading_day_now", lambda: True)
    monkeypatch.setattr(module, "is_nyse_trading_day_ending", lambda: True)

    record_calls = []
    monkeypatch.setattr(
        module,
        "_record_trade_outcome",
        lambda pos, reason: record_calls.append((pos.symbol, reason)),
    )

    backout_calls = []
    monkeypatch.setattr(module, "backout_near_market", lambda sym: backout_calls.append(sym))

    monkeypatch.setattr(module, "ramp_into_position", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "spawn_close_position_at_takeprofit", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "StockHistoricalDataClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "download_exchange_latest_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "get_bid", lambda sym: 9.5)
    monkeypatch.setattr(module, "get_ask", lambda sym: 10.0)
    monkeypatch.setattr(module, "get_qty", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(module, "_mark_probe_transitioned", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_mark_probe_active", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_mark_probe_pending", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_update_active_trade", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "record_portfolio_snapshot",
        lambda total_value, observed_at=None: SimpleNamespace(
            observed_at=datetime.now(timezone.utc),
            portfolio_value=total_value,
            risk_threshold=0.05,
        ),
    )

    current_pick = {
        "trade_mode": "probe",
        "probe_transition_ready": False,
        "probe_expired": True,
        "side": "buy",
        "strategy": "simple",
        "trade_blocked": False,
        "pending_probe": True,
        "probe_active": True,
        "predicted_movement": 0.5,
        "composite_score": 0.1,
    }
    current_picks = {symbol: current_pick}
    analyzed_results = {symbol: copy.deepcopy(current_pick)}

    module.manage_positions(current_picks, previous_picks={}, all_analyzed_results=analyzed_results)

    assert record_calls == [(symbol, "probe_duration_exceeded")]
    assert backout_calls == [symbol]
