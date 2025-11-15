import pandas as pd
import pytest

import hourlycrypto.trade_stock_crypto_hourly as module


def test_build_trading_plan_handles_frame():
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
            "buy_price": [10.0],
            "sell_price": [11.0],
            "buy_amount": [1.5],
            "sell_amount": [0.5],
        }
    )
    plan = module._build_trading_plan(df)
    assert plan is not None
    assert plan.buy_price == 10.0


def test_spawn_watchers_clamps_by_cash(monkeypatch):
    plan = module.TradingPlan(
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        buy_price=10.0,
        sell_price=12.0,
        buy_amount=5.0,
        sell_amount=3.0,
    )
    monkeypatch.setattr(module, "_available_cash", lambda: 25.0)
    monkeypatch.setattr(module, "_current_inventory", lambda symbol: 1.0)
    calls = {}

    def fake_spawn_open(symbol, side, price, qty, **kwargs):
        calls["buy_qty"] = qty

    def fake_spawn_close(symbol, side, price, **kwargs):
        calls["sell_price"] = price

    monkeypatch.setattr(module, "spawn_open_position_at_maxdiff_takeprofit", fake_spawn_open)
    monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit", fake_spawn_close)
    module._spawn_watchers(plan, dry_run=False)
    assert calls["buy_qty"] == pytest.approx(2.5)
    assert calls["sell_price"] == 12.0
