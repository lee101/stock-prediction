from types import SimpleNamespace

import pandas as pd
import pytest

import hourlycrypto.trade_stock_crypto_hourly as module


def test_build_trading_plan_handles_frame():
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
            "buy_price": [10.0],
            "sell_price": [11.0],
            "trade_amount": [0.6],
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
        trade_amount=0.5,
    )
    monkeypatch.setattr(module, "_available_cash", lambda: 25.0)
    monkeypatch.setattr(module, "_current_inventory", lambda symbol: 1.0)
    calls = {}

    def fake_spawn_open(symbol, side, price, qty, **kwargs):
        calls["buy_qty"] = qty

    def fake_spawn_close(symbol, side, price, **kwargs):
        calls["sell_price"] = price
        calls["sell_qty"] = kwargs.get("target_qty")

    monkeypatch.setattr(module, "spawn_open_position_at_maxdiff_takeprofit", fake_spawn_open)
    monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit", fake_spawn_close)
    module._spawn_watchers(plan, dry_run=False, symbol="LINKUSD")
    assert calls["buy_qty"] == pytest.approx(1.25)
    assert calls["sell_price"] == 12.0
    assert calls["sell_qty"] == pytest.approx(0.5)


def test_main_skips_trading_when_mode_train(monkeypatch):
    args = SimpleNamespace(
        mode="train",
        log_level="INFO",
        dry_run=True,
        daemon=False,
        window_hours=24,
        epochs=1,
        sequence_length=72,
        batch_size=16,
        checkpoint_root=None,
        checkpoint_path=None,
        force_retrain=False,
        training_symbols=None,
        price_offset_pct=None,
        price_offset_span_multiplier=0.0,
        price_offset_max_pct=0.003,
        symbol="LINKUSD",
    )
    monkeypatch.setattr(module, "_parse_args", lambda: args)
    monkeypatch.setattr(module, "_configure_logging", lambda level: None)
    config = SimpleNamespace(force_retrain=False, price_offset_pct=0.0003)
    monkeypatch.setattr(module, "_build_training_config", lambda parsed: config)
    monkeypatch.setattr(module, "_ensure_forecasts", lambda config, cache_only=False: None)
    monkeypatch.setattr(module, "_load_pretrained_policy", lambda config, **kwargs: None)
    monkeypatch.setattr(module, "_train_policy", lambda config, **kwargs: ("data", "policy", None))
    monkeypatch.setattr(
        module,
        "_infer_actions",
        lambda policy, data, config, offset_params=None: pd.DataFrame(),
    )
    simulate_calls = {"count": 0}

    def fake_simulate(actions, data_module, window):
        simulate_calls["count"] += 1

    monkeypatch.setattr(module, "_simulate", fake_simulate)
    plan_called = {"called": False}

    def fake_build_plan(actions):
        plan_called["called"] = True
        return module.TradingPlan(
            timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
            buy_price=10.0,
            sell_price=11.0,
            trade_amount=1.0,
        )

    monkeypatch.setattr(module, "_build_trading_plan", fake_build_plan)
    spawn_calls = {"count": 0}

    def fake_spawn_watchers(plan, dry_run, symbol):
        spawn_calls["count"] += 1

    monkeypatch.setattr(module, "_spawn_watchers", fake_spawn_watchers)

    module.main()

    assert simulate_calls["count"] == 1
    assert plan_called["called"] is False
    assert spawn_calls["count"] == 0
