import importlib
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest


def _install_stub(monkeypatch, *, minutes_to_close: float = 60.0):
    """Provide a lightweight alpaca_wrapper stub before importing the script."""

    def _clock():
        return SimpleNamespace(next_close=datetime.now(timezone.utc) + timedelta(minutes=minutes_to_close))

    captured = {"limit": [], "market": []}

    stub = SimpleNamespace(
        get_clock_internal=_clock,
        close_position_near_market=lambda pos, pct_above_market=0.0: captured["limit"].append(
            (pos.symbol, pct_above_market)
        ),
        close_position_violently=lambda pos: captured["market"].append(pos.symbol),
        get_account=lambda: SimpleNamespace(equity="100000"),
        get_all_positions=lambda: [],
    )

    monkeypatch.setitem(sys.modules, "alpaca_wrapper", stub)
    if "scripts.deleverage_account_day_end" in sys.modules:
        del sys.modules["scripts.deleverage_account_day_end"]
    module = importlib.import_module("scripts.deleverage_account_day_end")
    return module, captured


def _position(symbol: str, side: str, qty: float, price: float) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        side=side,
        qty=str(qty),
        market_value=str(qty * price),
    )


def test_filter_equity_positions_excludes_crypto(monkeypatch):
    module, _ = _install_stub(monkeypatch)

    positions = [
        _position("AAPL", "long", 10, 200),
        _position("BTCUSD", "long", 1, 30000),
        _position("MSFT", "short", 5, 300),
    ]

    equities = module._filter_equity_positions(positions)
    symbols = {p.symbol for p in equities}

    assert symbols == {"AAPL", "MSFT"}


def test_build_reduction_plan_generates_partial_exit(monkeypatch):
    module, _ = _install_stub(monkeypatch)
    positions = [ _position("AAPL", "long", 10, 200) ]

    plan = module._build_reduction_plan(positions, target_notional=1000, use_market=False, progress=0.0)
    assert len(plan) == 1
    order = plan[0]
    assert order.symbol == "AAPL"
    assert order.use_market is False
    # Half the position should remain (target 1000 out of 2000 exposure)
    assert pytest.approx(order.qty, rel=1e-3) == 5
    assert order.limit_offset > 0  # start of ramp sells slightly above bid


def test_build_reduction_plan_switches_to_market(monkeypatch):
    module, _ = _install_stub(monkeypatch)
    positions = [ _position("MSFT", "short", 20, 150) ]

    plan = module._build_reduction_plan(positions, target_notional=0, use_market=True, progress=1.0)
    assert len(plan) == 1
    order = plan[0]
    assert order.use_market is True
    assert order.limit_offset > 0  # short cover prefers crossing through ask


def test_apply_orders_routes_to_wrapper(monkeypatch):
    module, captured = _install_stub(monkeypatch)
    orders = [
        module.ReductionOrder(symbol="AAPL", side="long", qty=1, notional=200, use_market=False, limit_offset=0.01),
        module.ReductionOrder(symbol="MSFT", side="short", qty=2, notional=300, use_market=True, limit_offset=-0.02),
    ]

    module._apply_orders(orders)

    assert captured["limit"] == [("AAPL", 0.01)]
    assert captured["market"] == ["MSFT"]
