from types import SimpleNamespace

import pandas as pd
import pytest

import hourlycrypto.trade_stock_crypto_hourly as module


def _mock_spawn_deps(monkeypatch):
    """Monkeypatch all helper functions that _spawn_watchers calls."""
    monkeypatch.setattr(module, "_current_avg_price", lambda symbol: 9.0)
    monkeypatch.setattr(module, "_current_max_takeprofit", lambda symbol: 0.0)
    monkeypatch.setattr(module, "_cancel_existing_orders", lambda symbol: None)
    monkeypatch.setattr(module, "_latest_quote", lambda symbol: (None, None, None, None))
    monkeypatch.setattr(module, "_adjust_for_maker_liquidity", lambda plan, bid, ask, mid: plan)
    monkeypatch.setattr(module, "_get_min_order_notional", lambda symbol: 1.0)
    monkeypatch.setattr(module, "record_buy", lambda symbol, price: None)
    monkeypatch.setattr(module, "record_sell", lambda symbol, price: None)
    monkeypatch.setattr(module, "enforce_gap", lambda symbol, buy, sell, min_gap_pct=0: (buy, sell))
    # Clear position cap so each test starts fresh
    monkeypatch.setattr(module, "get_position_cap", lambda symbol, side: None)
    cap_calls = {}
    monkeypatch.setattr(
        module,
        "set_position_cap",
        lambda symbol, side, max_qty, buy_signal_qty: cap_calls.update(
            {f"{symbol}_{side}": max_qty}
        ),
    )
    return cap_calls


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
        buy_amount=0.5,
        sell_amount=0.5,
    )
    monkeypatch.setattr(module, "_available_cash", lambda: 25.0)
    monkeypatch.setattr(module, "_current_inventory", lambda symbol: 1.0)
    _mock_spawn_deps(monkeypatch)
    calls = {}

    def fake_spawn_open(symbol, side, price, qty, **kwargs):
        calls["buy_qty"] = qty

    def fake_spawn_close(symbol, side, price, **kwargs):
        calls["sell_price"] = price
        calls["sell_qty"] = kwargs.get("target_qty")

    monkeypatch.setattr(module, "spawn_open_position_at_maxdiff_takeprofit", fake_spawn_open)
    monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit", fake_spawn_close)
    module._spawn_watchers(plan, dry_run=False, symbol="LINKUSD")
    # buy_qty = buy_amt(0.5) * cash(25) / price(10) = 1.25
    # target_buy_qty = buy_qty(1.25) + inventory(1.0) = 2.25
    assert calls["buy_qty"] == pytest.approx(2.25)
    assert calls["sell_price"] == 12.0
    # sell_qty = sell_amt(0.5) * inventory(1.0) = 0.5
    assert calls["sell_qty"] == pytest.approx(0.5)


def test_spawn_watchers_cap_prevents_stacking(monkeypatch):
    """When a position cap already exists, target should not grow beyond it."""
    plan = module.TradingPlan(
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        buy_price=10.0,
        sell_price=12.0,
        buy_amount=0.5,
        sell_amount=0.0,
    )
    monkeypatch.setattr(module, "_available_cash", lambda: 50.0)
    monkeypatch.setattr(module, "_current_inventory", lambda symbol: 5.0)
    _mock_spawn_deps(monkeypatch)

    # Simulate existing cap from a previous signal (set when inventory was 0)
    monkeypatch.setattr(module, "get_position_cap", lambda symbol, side: 5.0)

    targets = []

    def fake_spawn_open(symbol, side, price, qty, **kwargs):
        targets.append(qty)

    monkeypatch.setattr(module, "spawn_open_position_at_maxdiff_takeprofit", fake_spawn_open)
    monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit", lambda *a, **kw: None)

    module._spawn_watchers(plan, dry_run=False, symbol="SOLUSD")

    # Without cap: target = 0.5 * 50/10 + 5 = 7.5
    # With cap = 5.0: target is capped to 5.0
    assert targets[0] == pytest.approx(5.0)


def test_spawn_watchers_sets_cap_on_first_signal(monkeypatch):
    """First signal sets the position cap for future calls."""
    plan = module.TradingPlan(
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        buy_price=10.0,
        sell_price=12.0,
        buy_amount=0.5,
        sell_amount=0.0,
    )
    monkeypatch.setattr(module, "_available_cash", lambda: 100.0)
    monkeypatch.setattr(module, "_current_inventory", lambda symbol: 0.0)
    cap_calls = _mock_spawn_deps(monkeypatch)

    def fake_spawn_open(symbol, side, price, qty, **kwargs):
        pass

    monkeypatch.setattr(module, "spawn_open_position_at_maxdiff_takeprofit", fake_spawn_open)
    monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit", lambda *a, **kw: None)

    module._spawn_watchers(plan, dry_run=False, symbol="SOLUSD")

    # buy_qty = 0.5 * 100/10 = 5.0, target = 5.0 + 0 = 5.0
    assert cap_calls.get("SOLUSD_buy") == pytest.approx(5.0)


def test_spawn_watchers_no_stacking_across_cycles(monkeypatch):
    """Simulate multiple hourly cycles: cap should freeze at first signal's target."""
    monkeypatch.setattr(module, "_current_avg_price", lambda symbol: 9.0)
    monkeypatch.setattr(module, "_current_max_takeprofit", lambda symbol: 0.0)
    monkeypatch.setattr(module, "_cancel_existing_orders", lambda symbol: None)
    monkeypatch.setattr(module, "_latest_quote", lambda symbol: (None, None, None, None))
    monkeypatch.setattr(module, "_adjust_for_maker_liquidity", lambda plan, bid, ask, mid: plan)
    monkeypatch.setattr(module, "_get_min_order_notional", lambda symbol: 1.0)
    monkeypatch.setattr(module, "record_buy", lambda symbol, price: None)
    monkeypatch.setattr(module, "record_sell", lambda symbol, price: None)
    monkeypatch.setattr(module, "enforce_gap", lambda symbol, buy, sell, min_gap_pct=0: (buy, sell))

    targets = []
    stored_cap = {"value": None}

    def mock_get_cap(symbol, side):
        return stored_cap["value"]

    def mock_set_cap(symbol, side, max_qty, buy_signal_qty):
        stored_cap["value"] = max_qty

    monkeypatch.setattr(module, "get_position_cap", mock_get_cap)
    monkeypatch.setattr(module, "set_position_cap", mock_set_cap)

    def fake_spawn_open(symbol, side, price, qty, **kwargs):
        targets.append(qty)

    monkeypatch.setattr(module, "spawn_open_position_at_maxdiff_takeprofit", fake_spawn_open)
    monkeypatch.setattr(module, "spawn_close_position_at_maxdiff_takeprofit", lambda *a, **kw: None)

    # Cycle 1: no inventory, plenty of cash
    monkeypatch.setattr(module, "_available_cash", lambda: 100.0)
    monkeypatch.setattr(module, "_current_inventory", lambda symbol: 0.0)
    plan = module.TradingPlan(
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
        buy_price=10.0,
        sell_price=12.0,
        buy_amount=0.5,
        sell_amount=0.0,
    )
    module._spawn_watchers(plan, dry_run=False, symbol="SOLUSD")

    # Cycle 2: inventory grew from fills, cash decreased
    monkeypatch.setattr(module, "_available_cash", lambda: 50.0)
    monkeypatch.setattr(module, "_current_inventory", lambda symbol: 5.0)
    module._spawn_watchers(plan, dry_run=False, symbol="SOLUSD")

    # Cycle 3: more inventory, even less cash
    monkeypatch.setattr(module, "_available_cash", lambda: 25.0)
    monkeypatch.setattr(module, "_current_inventory", lambda symbol: 7.5)
    module._spawn_watchers(plan, dry_run=False, symbol="SOLUSD")

    # Cycle 1 target: buy_qty(5) + inv(0) = 5. Cap set to 5.
    assert targets[0] == pytest.approx(5.0)
    # Cycle 2: computed target = 2.5 + 5 = 7.5, but cap = 5 → capped to 5
    assert targets[1] == pytest.approx(5.0)
    # Cycle 3: computed target = 1.25 + 7.5 = 8.75, but cap = 5 → capped to 5
    assert targets[2] == pytest.approx(5.0)

    # All targets should be identical (frozen cap)
    assert all(t == pytest.approx(5.0) for t in targets)


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
        retrain_to_keep_up_to_date=False,
        learning_rate=None,
        buy_price_offset_pct=None,
        sell_price_offset_pct=None,
        buy_price_offset_span_multiplier=None,
        sell_price_offset_span_multiplier=None,
        buy_price_offset_max_pct=None,
        sell_price_offset_max_pct=None,
        cache_only_forecasts=False,
        dry_train_steps=None,
        ema_decay=None,
        no_compile=False,
        enable_compile=False,
        use_amp=False,
        amp_dtype="bfloat16",
        preload_checkpoint=None,
        dropout=None,
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
            buy_amount=1.0,
            sell_amount=0.0,
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
