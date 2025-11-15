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

    monkeypatch.setattr(module, "ENABLE_PROBE_TRADES", True)
    monkeypatch.setattr(module, "_recent_trade_pnl_pcts", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "_recent_trade_pnls", lambda *args, **kwargs: [])
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
        lambda sym, side, target_qty=None, **kwargs: ramp_calls.append((sym, side, target_qty)),
    )

    transition_calls = []
    monkeypatch.setattr(
        module,
        "_mark_probe_transitioned",
        lambda sym, side, qty, strategy=None: (transition_calls.append((sym, side, qty)) or {}),
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
    monkeypatch.setattr(
        module,
        "_evaluate_trade_block",
        lambda sym, side, strategy=None: {},
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
    assert trans_qty == pytest.approx(1.0)
    assert probe_active_calls == []
    assert len(active_trade_updates) >= 1
    act_symbol, act_side, act_mode, act_qty = active_trade_updates[-1]
    assert (act_symbol, act_side, act_mode) == (symbol, "buy", "probe_transition")
    assert act_qty == pytest.approx(1.0)
    assert ramp_calls == []


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
    monkeypatch.setattr(module, "backout_near_market", lambda sym, **kwargs: backout_calls.append(sym))

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
    monkeypatch.setattr(
        module,
        "_evaluate_trade_block",
        lambda sym, side, strategy=None: {},
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


def test_evaluate_trade_block_forces_probe_with_recent_losses(monkeypatch):
    module = trade_stock_e2e
    timestamp = datetime(2025, 11, 14, 16, 0, tzinfo=timezone.utc).isoformat()
    monkeypatch.setattr(module, "PROBE_TRADE_MODE", True)
    monkeypatch.setattr(module, "ENABLE_PROBE_TRADES", True)
    monkeypatch.setattr(module, "_load_trade_outcome", lambda *args, **kwargs: {"pnl": -5.0, "closed_at": timestamp})
    monkeypatch.setattr(module, "_load_learning_state", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_recent_trade_pnl_pcts", lambda *args, **kwargs: [-0.02, -0.03])

    result = module._evaluate_trade_block("ETHUSD", "buy")

    assert result["trade_mode"] == "probe"
    assert result["pending_probe"] is True
    assert result["blocked"] is False


def test_evaluate_trade_block_remains_blocked_when_recent_positive(monkeypatch):
    module = trade_stock_e2e
    timestamp = datetime(2025, 11, 14, 16, 0, tzinfo=timezone.utc).isoformat()
    monkeypatch.setattr(module, "PROBE_TRADE_MODE", True)
    monkeypatch.setattr(module, "ENABLE_PROBE_TRADES", True)
    monkeypatch.setattr(module, "_load_trade_outcome", lambda *args, **kwargs: {"pnl": -5.0, "closed_at": timestamp})
    monkeypatch.setattr(module, "_load_learning_state", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_recent_trade_pnl_pcts", lambda *args, **kwargs: [0.05, -0.01])

    result = module._evaluate_trade_block("ETHUSD", "buy")

    assert result["blocked"] is True
    assert result["trade_mode"] == "normal"


def test_manage_positions_promotes_large_notional_probe(monkeypatch):
    module = trade_stock_e2e
    symbol = "NVDA"

    monkeypatch.setattr(module, "PROBE_NOTIONAL_LIMIT", 300.0)

    positions = [make_position(symbol, qty=12.0, price=191.0, side="long")]
    module.alpaca_wrapper.equity = 25000.0

    monkeypatch.setattr(module.alpaca_wrapper, "get_all_positions", lambda: positions)
    monkeypatch.setattr(module, "filter_to_realistic_positions", lambda pos: pos)
    monkeypatch.setattr(module, "_handle_live_drawdown", lambda *_: None)
    monkeypatch.setattr(module, "is_nyse_trading_day_now", lambda: True)
    monkeypatch.setattr(module, "is_nyse_trading_day_ending", lambda: True)

    account = SimpleNamespace(equity=25000.0, last_equity=24000.0)
    monkeypatch.setattr(module.alpaca_wrapper, "get_account", lambda: account)

    monkeypatch.setattr(
        module,
        "record_portfolio_snapshot",
        lambda total_value, **_: SimpleNamespace(
            observed_at=datetime.now(timezone.utc),
            portfolio_value=total_value,
            risk_threshold=1.0,
        ),
    )

    monkeypatch.setattr(module, "StockHistoricalDataClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "download_exchange_latest_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "get_bid", lambda sym: 191.0)
    monkeypatch.setattr(module, "get_ask", lambda sym: 191.5)
    monkeypatch.setattr(module, "get_qty", lambda sym, price, _positions: 12.0)
    monkeypatch.setattr(module, "spawn_close_position_at_takeprofit", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "ramp_into_position", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "backout_near_market", lambda *args, **kwargs: None)

    record_calls = []
    monkeypatch.setattr(
        module,
        "_record_trade_outcome",
        lambda pos, reason: record_calls.append((pos.symbol, reason)),
    )

    active_trade_updates = []
    monkeypatch.setattr(
        module,
        "_update_active_trade",
        lambda sym, side, mode, qty, strategy=None: active_trade_updates.append(
            (sym, side, mode, qty, strategy)
        ),
    )

    monkeypatch.setattr(
        module,
        "_get_active_trade",
        lambda sym, side: {"entry_strategy": "simple", "qty": 6.0},
    )

    probe_state = {
        "pending_probe": True,
        "probe_active": True,
        "probe_expired": True,
        "trade_mode": "probe",
        "probe_transition_ready": False,
    }

    transition_calls = []

    def fake_mark_probe_transitioned(sym, side, qty, strategy=None):
        transition_calls.append((sym, side, qty))
        probe_state.update(
            pending_probe=False,
            probe_active=False,
            probe_expired=False,
            trade_mode="normal",
            probe_transition_ready=False,
        )
        return dict(probe_state)

    monkeypatch.setattr(module, "_mark_probe_transitioned", fake_mark_probe_transitioned)
    monkeypatch.setattr(module, "_mark_probe_active", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_mark_probe_pending", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "_normalize_active_trade_patch", lambda *_: None)

    monkeypatch.setattr(
        module,
        "_evaluate_trade_block",
        lambda sym, side, strategy=None: dict(probe_state),
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
        "composite_score": 0.7,
    }
    current_picks = {symbol: current_pick}
    analyzed_results = {symbol: dict(current_pick)}

    module.manage_positions(current_picks, previous_picks={}, all_analyzed_results=analyzed_results)

    assert record_calls == []
    assert len(transition_calls) == 1
    trans_symbol, trans_side, trans_qty = transition_calls[0]
    assert (trans_symbol, trans_side) == (symbol, "buy")
    assert trans_qty == pytest.approx(12.0)
    assert probe_state["pending_probe"] is False
    assert probe_state["probe_active"] is False
    assert probe_state["trade_mode"] == "normal"
    assert active_trade_updates
    act_symbol, act_side, act_mode, act_qty, _ = active_trade_updates[-1]
    assert (act_symbol, act_side, act_mode) == (symbol, "buy", "probe_transition")
    assert act_qty == pytest.approx(12.0)


def test_handle_live_drawdown_marks_position_in_drawdown(monkeypatch):
    """Test that _handle_live_drawdown marks position for probe when PnL drops below threshold"""
    module = trade_stock_e2e
    symbol = "TEST"

    # Create position with PnL below LIVE_DRAWDOWN_TRIGGER (-500)
    position = make_position(symbol, qty=10.0, price=100.0, side="long", unrealized_pl=-600.0)

    # Setup mocks
    state_updates = []
    def mock_update_learning_state(sym, side, **kwargs):
        state_updates.append((sym, side, kwargs))
        return {"pending_probe": kwargs.get("pending_probe", False), "probe_active": False}

    monkeypatch.setattr(module, "_load_learning_state", lambda sym, side, strategy=None: {"pending_probe": False, "probe_active": False})
    monkeypatch.setattr(module, "_update_learning_state", mock_update_learning_state)
    monkeypatch.setattr(module, "_normalize_side_for_key", lambda side: "buy")

    # Call _handle_live_drawdown
    module._handle_live_drawdown(position)

    # Verify probe was marked
    assert len(state_updates) == 1
    assert state_updates[0][0] == symbol
    assert state_updates[0][2]["pending_probe"] is True


def test_handle_live_drawdown_clears_probe_on_recovery(monkeypatch):
    """Test that _handle_live_drawdown clears probe flag when position recovers"""
    module = trade_stock_e2e
    symbol = "TEST"

    # Create position with PnL above LIVE_DRAWDOWN_TRIGGER (recovered)
    position = make_position(symbol, qty=10.0, price=100.0, side="long", unrealized_pl=-400.0)

    # Setup mocks - position was previously marked for probe
    state_updates = []
    def mock_update_learning_state(sym, side, **kwargs):
        state_updates.append((sym, side, kwargs))
        return {"pending_probe": kwargs.get("pending_probe", False), "probe_active": False}

    monkeypatch.setattr(module, "_load_learning_state", lambda sym, side, strategy=None: {"pending_probe": True, "probe_active": False})
    monkeypatch.setattr(module, "_update_learning_state", mock_update_learning_state)
    monkeypatch.setattr(module, "_normalize_side_for_key", lambda side: "buy")

    # Call _handle_live_drawdown
    module._handle_live_drawdown(position)

    # Verify probe flag was cleared
    assert len(state_updates) == 1
    assert state_updates[0][0] == symbol
    assert state_updates[0][2]["pending_probe"] is False


def test_handle_live_drawdown_handles_multiple_fluctuations(monkeypatch):
    """Test that multiple PnL fluctuations are handled correctly"""
    module = trade_stock_e2e
    symbol = "TEST"

    state = {"pending_probe": False, "probe_active": False}
    state_updates = []

    def mock_load_learning_state(sym, side, strategy=None):
        return dict(state)

    def mock_update_learning_state(sym, side, **kwargs):
        state_updates.append((sym, side, dict(kwargs)))
        state.update(kwargs)
        return dict(state)

    monkeypatch.setattr(module, "_load_learning_state", mock_load_learning_state)
    monkeypatch.setattr(module, "_update_learning_state", mock_update_learning_state)
    monkeypatch.setattr(module, "_normalize_side_for_key", lambda side: "buy")

    # Fluctuation 1: Drop below threshold
    position = make_position(symbol, qty=10.0, price=100.0, side="long", unrealized_pl=-600.0)
    module._handle_live_drawdown(position)
    assert state_updates[-1][2]["pending_probe"] is True
    assert state["pending_probe"] is True

    # Fluctuation 2: Recover above threshold
    position = make_position(symbol, qty=10.0, price=100.0, side="long", unrealized_pl=-400.0)
    module._handle_live_drawdown(position)
    assert state_updates[-1][2]["pending_probe"] is False
    assert state["pending_probe"] is False

    # Fluctuation 3: Drop again
    position = make_position(symbol, qty=10.0, price=100.0, side="long", unrealized_pl=-700.0)
    module._handle_live_drawdown(position)
    assert state_updates[-1][2]["pending_probe"] is True
    assert state["pending_probe"] is True

    # Verify we had 3 state updates
    assert len(state_updates) == 3


def test_handle_live_drawdown_does_not_clear_active_probe(monkeypatch):
    """Test that recovery doesn't clear probe flag when probe is already active"""
    module = trade_stock_e2e
    symbol = "TEST"

    # Create position with PnL above threshold (recovered)
    position = make_position(symbol, qty=10.0, price=100.0, side="long", unrealized_pl=-400.0)

    # Setup mocks - position has active probe (already executing)
    state_updates = []
    def mock_update_learning_state(sym, side, **kwargs):
        state_updates.append((sym, side, kwargs))
        return {"pending_probe": False, "probe_active": True}

    monkeypatch.setattr(module, "_load_learning_state", lambda sym, side, strategy=None: {"pending_probe": True, "probe_active": True})
    monkeypatch.setattr(module, "_update_learning_state", mock_update_learning_state)
    monkeypatch.setattr(module, "_normalize_side_for_key", lambda side: "buy")

    # Call _handle_live_drawdown
    module._handle_live_drawdown(position)

    # Verify no state updates (probe is active, don't interfere)
    assert len(state_updates) == 0
