from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from binanceleveragesui import trade_margin_meta as meta


@pytest.fixture(autouse=True)
def _disable_meta_runtime_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARGIN_META_DISABLE_LOG", "1")
    monkeypatch.setattr(meta, "_log_event", lambda *args, **kwargs: None)


def test_select_model_winner_uses_highest_score(monkeypatch: pytest.MonkeyPatch) -> None:
    histories = {
        "doge": SimpleNamespace(tag="doge"),
        "aave": SimpleNamespace(tag="aave"),
    }
    score_by_name = {"doge": 0.12, "aave": 0.35}

    def fake_score(history, lookback, maker_fee, max_leverage, metric):
        return score_by_name[history.tag]

    monkeypatch.setattr(meta, "_run_hypothetical_score", fake_score)

    selected = meta.select_model(
        histories,
        lookback=12,
        max_leverage=2.0,
        metric="sortino",
        selection_mode="winner",
        cash_threshold=0.0,
    )
    assert selected == "aave"


def test_select_model_winner_cash_can_stay_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    histories = {
        "doge": SimpleNamespace(tag="doge"),
        "aave": SimpleNamespace(tag="aave"),
    }
    score_by_name = {"doge": -0.02, "aave": -0.01}

    def fake_score(history, lookback, maker_fee, max_leverage, metric):
        return score_by_name[history.tag]

    monkeypatch.setattr(meta, "_run_hypothetical_score", fake_score)

    selected = meta.select_model(
        histories,
        lookback=12,
        max_leverage=2.0,
        metric="calmar",
        selection_mode="winner_cash",
        cash_threshold=0.0,
    )
    assert selected == ""


def test_select_model_rejects_unknown_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    histories = {
        "doge": SimpleNamespace(tag="doge"),
        "aave": SimpleNamespace(tag="aave"),
    }

    monkeypatch.setattr(meta, "_run_hypothetical_score", lambda *args, **kwargs: 0.0)

    with pytest.raises(ValueError, match="Unsupported selection metric"):
        meta.select_model(
            histories,
            lookback=12,
            max_leverage=2.0,
            metric="unknown_metric",
            selection_mode="winner",
            cash_threshold=0.0,
        )


def test_select_model_switch_margin_keeps_current_model(monkeypatch: pytest.MonkeyPatch) -> None:
    histories = {
        "doge": SimpleNamespace(tag="doge"),
        "aave": SimpleNamespace(tag="aave"),
    }
    score_by_name = {"doge": 0.20, "aave": 0.205}

    def fake_score(history, lookback, maker_fee, max_leverage, metric):
        return score_by_name[history.tag]

    monkeypatch.setattr(meta, "_run_hypothetical_score", fake_score)

    selected = meta.select_model(
        histories,
        lookback=12,
        max_leverage=2.0,
        metric="sortino",
        selection_mode="winner",
        cash_threshold=0.0,
        current_model="doge",
        switch_margin=0.01,
    )
    assert selected == "doge"


def test_select_model_cash_hysteresis_keeps_current_model(monkeypatch: pytest.MonkeyPatch) -> None:
    histories = {
        "doge": SimpleNamespace(tag="doge"),
        "aave": SimpleNamespace(tag="aave"),
    }
    score_by_name = {"doge": 0.005, "aave": -0.01}

    def fake_score(history, lookback, maker_fee, max_leverage, metric):
        return score_by_name[history.tag]

    monkeypatch.setattr(meta, "_run_hypothetical_score", fake_score)

    selected = meta.select_model(
        histories,
        lookback=12,
        max_leverage=2.0,
        metric="sortino",
        selection_mode="winner_cash",
        cash_threshold=0.01,
        current_model="doge",
        switch_margin=0.01,
    )
    assert selected == "doge"


def test_select_model_winner_cash_min_gap_can_stay_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    histories = {
        "doge": SimpleNamespace(tag="doge"),
        "aave": SimpleNamespace(tag="aave"),
    }
    score_by_name = {"doge": 0.200, "aave": 0.195}

    def fake_score(history, lookback, maker_fee, max_leverage, metric):
        return score_by_name[history.tag]

    monkeypatch.setattr(meta, "_run_hypothetical_score", fake_score)

    selected = meta.select_model(
        histories,
        lookback=12,
        max_leverage=2.0,
        metric="sortino",
        selection_mode="winner_cash",
        cash_threshold=0.0,
        current_model="",
        switch_margin=0.0,
        min_score_gap=0.01,
    )
    assert selected == ""


def test_select_model_profit_gate_blocks_unprofitable_model(monkeypatch: pytest.MonkeyPatch) -> None:
    histories = {
        "doge": SimpleNamespace(tag="doge"),
        "aave": SimpleNamespace(tag="aave"),
    }
    score_by_name = {"doge": 0.40, "aave": 0.20}
    recent_return_by_name = {"doge": -0.01, "aave": 0.02}

    def fake_score(history, lookback, maker_fee, max_leverage, metric, **kwargs):
        if metric == "return":
            return recent_return_by_name[history.tag]
        return score_by_name[history.tag]

    monkeypatch.setattr(meta, "_run_hypothetical_score", fake_score)

    selected = meta.select_model(
        histories,
        lookback=12,
        max_leverage=2.0,
        metric="sortino",
        selection_mode="winner_cash",
        cash_threshold=0.0,
        profit_gate_lookback_hours=24,
        profit_gate_min_return=0.0,
    )
    assert selected == "aave"


def test_select_model_profit_gate_can_force_cash(monkeypatch: pytest.MonkeyPatch) -> None:
    histories = {
        "doge": SimpleNamespace(tag="doge"),
        "aave": SimpleNamespace(tag="aave"),
    }

    def fake_score(history, lookback, maker_fee, max_leverage, metric, **kwargs):
        if metric == "return":
            return -0.001
        return 0.40 if history.tag == "doge" else 0.20

    monkeypatch.setattr(meta, "_run_hypothetical_score", fake_score)

    selected = meta.select_model(
        histories,
        lookback=12,
        max_leverage=2.0,
        metric="sortino",
        selection_mode="winner_cash",
        cash_threshold=0.0,
        profit_gate_lookback_hours=24,
        profit_gate_min_return=0.0,
    )
    assert selected == ""


def test_select_model_min_gap_hysteresis_keeps_current_model(monkeypatch: pytest.MonkeyPatch) -> None:
    histories = {
        "doge": SimpleNamespace(tag="doge"),
        "aave": SimpleNamespace(tag="aave"),
    }
    score_by_name = {"doge": 0.200, "aave": 0.195}

    def fake_score(history, lookback, maker_fee, max_leverage, metric):
        return score_by_name[history.tag]

    monkeypatch.setattr(meta, "_run_hypothetical_score", fake_score)

    selected = meta.select_model(
        histories,
        lookback=12,
        max_leverage=2.0,
        metric="sortino",
        selection_mode="winner_cash",
        cash_threshold=0.0,
        current_model="doge",
        switch_margin=0.01,
        min_score_gap=0.01,
    )
    assert selected == "doge"


def test_hypothetical_score_accepts_timestamp_based_lookback() -> None:
    hist = meta.SignalHistory()
    base = meta.datetime(2026, 1, 1, tzinfo=meta.timezone.utc)
    prices = [100.0, 98.0, 96.0, 97.0, 99.0, 101.0, 103.0]
    for i, close in enumerate(prices):
        ts = (base + meta.pd.Timedelta(hours=i)).isoformat()
        hist.add(
            ts,
            buy_p=close * 0.99,
            sell_p=close * 1.01,
            buy_a=100.0 if i % 2 == 0 else 0.0,
            sell_a=100.0 if i % 2 == 1 else 0.0,
            close=close,
            equity=close,
        )

    s_short = meta._run_hypothetical_score(hist, lookback=2, maker_fee=0.001, max_leverage=2.0, metric="return")
    s_long = meta._run_hypothetical_score(hist, lookback=48, maker_fee=0.001, max_leverage=2.0, metric="return")
    assert np.isfinite(s_short)
    assert np.isfinite(s_long)


def test_cap_buy_notional_uses_free_and_borrowable() -> None:
    capped = meta._cap_buy_notional(2500.0, usdt_free=100.0, max_borrowable_usdt=500.0)
    assert capped == pytest.approx(600.0)


def test_cap_buy_notional_leaves_smaller_target_unchanged() -> None:
    capped = meta._cap_buy_notional(80.0, usdt_free=100.0, max_borrowable_usdt=500.0)
    assert capped == pytest.approx(80.0)


def test_cap_position_notional_is_noop_without_cap() -> None:
    capped = meta._cap_position_notional(80.0, current_asset_notional=10.0, max_position_notional=None)
    assert capped == pytest.approx(80.0)


def test_cap_position_notional_limits_fresh_probe_entry() -> None:
    capped = meta._cap_position_notional(600.0, current_asset_notional=0.0, max_position_notional=5.0)
    assert capped == pytest.approx(5.0)


def test_cap_position_notional_blocks_add_when_already_at_probe_cap() -> None:
    capped = meta._cap_position_notional(40.0, current_asset_notional=5.2, max_position_notional=5.0)
    assert capped == pytest.approx(0.0)


def test_remaining_target_entry_notional_sizes_through_opposite_side_stub() -> None:
    remaining = meta._remaining_target_entry_notional(
        side="short",
        equity=1_000.0,
        asset_net=1.0,
        market_price=10.0,
        usdt_free=1_000.0,
        asset_free=1.0,
        max_borrowable_usdt=0.0,
        max_borrowable_asset=100.0,
        long_max_leverage=10.0,
        short_max_leverage=10.0,
        max_position_notional=50.0,
    )

    assert remaining == pytest.approx(60.0)


def test_order_needs_resize_to_target_for_material_qty_gap() -> None:
    order = {"id": 123, "qty": 50.0}
    assert meta._order_needs_resize_to_target(
        order,
        target_qty=100.0,
        reference_price=10.0,
        step_size=1.0,
        min_notional=5.0,
    ) is True
    assert meta._order_needs_resize_to_target(
        order,
        target_qty=50.4,
        reference_price=10.0,
        step_size=1.0,
        min_notional=5.0,
    ) is False


def test_resolve_max_position_notional_preserves_explicit_value() -> None:
    resolved = meta._resolve_max_position_notional(7.5, dry_run=False, paper_env="0")
    assert resolved == pytest.approx(7.5)


def test_resolve_max_position_notional_can_disable_probe_cap() -> None:
    resolved = meta._resolve_max_position_notional(
        None,
        dry_run=False,
        disable_probe_cap=True,
        paper_env="0",
    )
    assert resolved is None


def test_resolve_max_position_notional_enables_live_probe_default() -> None:
    resolved = meta._resolve_max_position_notional(None, dry_run=False, paper_env="0")
    assert resolved == pytest.approx(meta.LIVE_PROBE_MAX_POSITION_NOTIONAL)


def test_resolve_max_position_notional_skips_probe_default_for_dry_run() -> None:
    resolved = meta._resolve_max_position_notional(None, dry_run=True, paper_env="0")
    assert resolved is None


def test_effective_position_notional_threshold_tracks_probe_cap() -> None:
    assert meta._effective_position_notional_threshold(None) == pytest.approx(meta.MIN_POSITION_NOTIONAL)
    assert meta._effective_position_notional_threshold(5.10) == pytest.approx(4.08)


def test_has_effective_position_accepts_probe_sized_holding() -> None:
    assert meta._has_effective_position(
        asset_total=53.0,
        asset_value=4.90,
        step_size=1.0,
        max_position_notional=5.10,
    ) is True


def test_promote_detected_position_logs_balance_fill_for_tracked_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(meta, "_log_event", lambda event_type, **data: events.append((event_type, data)))
    monkeypatch.setattr(meta.MetaState, "save", lambda self, path=meta.STATE_FILE: None)

    state = meta.MetaState(active_model="doge", in_position=False, open_ts=None, open_price=0.0)
    entry_order = {"id": 123, "price": 0.09318, "qty": 53.0, "symbol": "DOGEUSDT"}

    cleared = meta._promote_detected_position(
        state,
        model="doge",
        position_side="long",
        market_price=0.09322,
        asset_total=53.0,
        asset_value=4.94,
        entry_order=entry_order,
    )

    assert state.in_position is True
    assert state.open_ts is not None
    assert state.open_price == pytest.approx(0.09318)
    assert cleared["id"] is None
    assert events == [
        (
            "entry_filled",
            {
                "model": "doge",
                "position_side": "long",
                "price": 0.09318,
                "qty": 53.0,
                "source": "balance_detection",
            },
        )
    ]


def test_reconcile_managed_positions_keeps_current_active_during_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    balances = {
        "DOGE": {
            "asset_net": 53.0,
            "position_value": 4.94,
            "position_side": "long",
            "market_price": 0.09318,
        },
        "AAVE": {
            "asset_net": 0.044,
            "position_value": 5.22,
            "position_side": "long",
            "market_price": 118.63,
        },
    }

    monkeypatch.setattr(meta, "_get_margin_equity_for", lambda symbol, base_asset: dict(balances[base_asset]))
    monkeypatch.setattr(meta.MetaState, "save", lambda self, path=meta.STATE_FILE: None)

    state = meta.MetaState(active_model="doge", in_position=False, open_ts=None, open_price=0.0)
    rules = {
        "doge": SimpleNamespace(step_size=1.0),
        "aave": SimpleNamespace(step_size=0.001),
    }

    reconciled = meta._reconcile_managed_positions(
        state,
        rules,
        max_position_notional=5.10,
    )

    assert reconciled["conflict"] is True
    assert reconciled["chosen_model"] == "doge"
    assert reconciled["details"] == {
        "aave": {"position_value": 5.22, "position_side": "long", "asset_net": 0.044},
        "doge": {"position_value": 4.94, "position_side": "long", "asset_net": 53.0},
    }
    assert state.active_model == "doge"
    assert state.in_position is True
    assert state.position_side == "long"
    assert state.open_ts is not None
    assert state.open_price == pytest.approx(0.09318)


def test_reconcile_managed_positions_prefers_largest_available_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    balances = {
        "DOGE": {
            "asset_net": 53.0,
            "position_value": 4.94,
            "position_side": "long",
            "market_price": 0.09318,
        },
        "AAVE": {
            "asset_net": -0.045,
            "position_value": 5.31,
            "position_side": "short",
            "market_price": 118.1,
        },
    }

    monkeypatch.setattr(meta, "_get_margin_equity_for", lambda symbol, base_asset: dict(balances[base_asset]))
    monkeypatch.setattr(meta.MetaState, "save", lambda self, path=meta.STATE_FILE: None)

    old_open_ts = "2026-03-06T00:00:00+00:00"
    state = meta.MetaState(
        active_model="btc",
        in_position=True,
        position_side="long",
        open_ts=old_open_ts,
        open_price=1.0,
    )
    rules = {
        "doge": SimpleNamespace(step_size=1.0),
        "aave": SimpleNamespace(step_size=0.001),
    }

    reconciled = meta._reconcile_managed_positions(
        state,
        rules,
        max_position_notional=5.10,
    )

    assert reconciled["chosen_model"] == "aave"
    assert state.active_model == "aave"
    assert state.in_position is True
    assert state.position_side == "short"
    assert state.open_ts is not None
    assert state.open_ts != old_open_ts
    assert state.open_price == pytest.approx(118.1)


def test_remaining_order_qty_never_negative() -> None:
    order = {"origQty": "10", "executedQty": "12"}
    assert meta._remaining_order_qty(order) == pytest.approx(0.0)


def test_meta_state_hours_held_clamps_future_timestamp() -> None:
    future = (meta.datetime.now(meta.timezone.utc) + meta.pd.Timedelta(hours=3)).isoformat()
    state = meta.MetaState(open_ts=future)
    assert state.hours_held() == pytest.approx(0.0)


def test_order_age_minutes_parses_millisecond_timestamp() -> None:
    now = meta.datetime.now(meta.timezone.utc)
    ts = now - meta.pd.Timedelta(minutes=95)
    order = {"time": int(ts.timestamp() * 1000)}
    age = meta._order_age_minutes(order, now=now)
    assert age is not None
    assert 94.0 <= age <= 96.0


def test_normalize_open_ts_prefers_open_order_time(monkeypatch: pytest.MonkeyPatch) -> None:
    now = meta.datetime.now(meta.timezone.utc)
    stale_order_time = now - meta.pd.Timedelta(hours=7)
    state = meta.MetaState(
        active_model="aave",
        in_position=True,
        open_ts=(now + meta.pd.Timedelta(hours=3)).isoformat(),
    )

    monkeypatch.setattr(meta, "_list_open_side_orders", lambda symbol, side: [{"time": int(stale_order_time.timestamp() * 1000)}])
    monkeypatch.setattr(meta, "_log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(meta.MetaState, "save", lambda self, path=meta.STATE_FILE: None)

    changed = meta._normalize_state_open_ts_for_position(
        state,
        "AAVEUSDT",
        max_hold_hours=6,
    )
    assert changed
    opened = meta.datetime.fromisoformat(state.open_ts)
    assert opened.tzinfo is not None
    assert abs((opened - stale_order_time).total_seconds()) <= 1.0


def test_normalize_open_ts_fallback_backdates_by_max_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    now = meta.datetime.now(meta.timezone.utc)
    state = meta.MetaState(
        active_model="aave",
        in_position=True,
        open_ts=(now + meta.pd.Timedelta(hours=2)).isoformat(),
    )

    monkeypatch.setattr(meta, "_list_open_side_orders", lambda symbol, side: [])
    monkeypatch.setattr(meta, "_log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(meta.MetaState, "save", lambda self, path=meta.STATE_FILE: None)

    changed = meta._normalize_state_open_ts_for_position(
        state,
        "AAVEUSDT",
        max_hold_hours=6,
    )
    assert changed
    opened = meta.datetime.fromisoformat(state.open_ts)
    delta_hours = (meta.datetime.now(meta.timezone.utc) - opened).total_seconds() / 3600.0
    assert delta_hours >= 5.99


def test_inventory_mostly_locked_detects_locked_position() -> None:
    assert meta._inventory_mostly_locked(asset_free=0.00043, asset_locked=47.936) is True


def test_inventory_mostly_locked_ignores_unlocked_position() -> None:
    assert meta._inventory_mostly_locked(asset_free=30.0, asset_locked=10.0) is False


def test_is_same_signal_hour_true_for_equivalent_hour() -> None:
    sig_hour = meta.pd.Timestamp("2026-03-06 08:00:00+00:00")
    order = {"signal_hour": sig_hour.isoformat()}
    signal = {"signal_hour": sig_hour}
    assert meta._is_same_signal_hour(order, signal) is True


def test_is_same_signal_hour_false_for_different_hours() -> None:
    order = {"signal_hour": "2026-03-06T08:00:00+00:00"}
    signal = {"signal_hour": meta.pd.Timestamp("2026-03-06 09:00:00+00:00")}
    assert meta._is_same_signal_hour(order, signal) is False


def test_upsert_signal_history_coalesces_same_hour_refreshes() -> None:
    history = meta.SignalHistory()
    first = {
        "buy_price": 100.0,
        "sell_price": 101.0,
        "buy_amount": 20.0,
        "sell_amount": 0.0,
        "close": 100.5,
        "signal_hour": meta.pd.Timestamp("2026-03-06 08:00:00+00:00"),
    }
    second = {
        "buy_price": 99.5,
        "sell_price": 101.5,
        "buy_amount": 25.0,
        "sell_amount": 5.0,
        "close": 100.2,
        "signal_hour": meta.pd.Timestamp("2026-03-06 08:00:00+00:00"),
    }

    meta._upsert_signal_history(history, first)
    meta._upsert_signal_history(history, second)

    assert len(history.timestamps) == 1
    assert history.timestamps[0] == "2026-03-06T08:00:00+00:00"
    assert history.buy_prices == [pytest.approx(99.5)]
    assert history.sell_prices == [pytest.approx(101.5)]
    assert history.buy_amounts == [pytest.approx(25.0)]
    assert history.sell_amounts == [pytest.approx(5.0)]
    assert history.closes == [pytest.approx(100.2)]


def test_runtime_snapshot_roundtrip_restores_signals_histories_and_order_hours(tmp_path) -> None:
    state = meta.MetaState(
        active_model="doge",
        in_position=True,
        open_ts="2026-03-06T08:34:34+00:00",
        open_price=0.09318,
    )
    histories = {name: meta.SignalHistory() for name in meta.MODELS}
    doge_signal_hour = meta.pd.Timestamp("2026-03-06 08:00:00+00:00")
    aave_signal_hour = meta.pd.Timestamp("2026-03-06 13:00:00+00:00")
    meta._upsert_signal_history(
        histories["doge"],
        {
            "buy_price": 0.0932,
            "sell_price": 0.0941,
            "buy_amount": 35.0,
            "sell_amount": 0.0,
            "close": 0.0935,
            "signal_hour": doge_signal_hour,
        },
    )
    meta._upsert_signal_history(
        histories["aave"],
        {
            "buy_price": 114.5,
            "sell_price": 116.0,
            "buy_amount": 40.0,
            "sell_amount": 0.0,
            "close": 115.0,
            "signal_hour": aave_signal_hour,
        },
    )
    signals = {
        "doge": {
            "symbol": "DOGEUSDT",
            "buy_price": 0.0932,
            "sell_price": 0.0941,
            "buy_amount": 35.0,
            "sell_amount": 0.0,
            "close": 0.0935,
            "signal_hour": doge_signal_hour,
        },
        "aave": {
            "symbol": "AAVEUSDT",
            "buy_price": 114.5,
            "sell_price": 116.0,
            "buy_amount": 40.0,
            "sell_amount": 0.0,
            "close": 115.0,
            "signal_hour": aave_signal_hour,
        },
    }
    entry_order = {
        "id": 77,
        "price": 114.5,
        "qty": 0.044,
        "symbol": "AAVEUSDT",
        "signal_hour": aave_signal_hour,
    }
    signature = {"mode": "test"}
    path = tmp_path / "margin_meta_runtime.json"

    meta._save_runtime_snapshot(
        state=state,
        histories=histories,
        signals=signals,
        entry_order=entry_order,
        exit_order=meta._empty_exit(),
        signature=signature,
        path=path,
    )

    payload = meta._load_runtime_snapshot(path=path)
    assert payload is not None
    assert meta._runtime_snapshot_is_compatible(payload, signature) is True
    assert meta._snapshot_signals_are_current(
        payload["signals"],
        {"doge": doge_signal_hour, "aave": aave_signal_hour},
    ) is True

    restored_histories = meta._restore_histories_from_snapshot(payload["histories"])
    restored_signals = meta._restore_signals_from_snapshot(payload["signals"])
    restored_entry = meta._deserialize_order(payload["entry_order"])

    assert restored_histories["doge"].timestamps == ["2026-03-06T08:00:00+00:00"]
    assert restored_histories["aave"].timestamps == ["2026-03-06T13:00:00+00:00"]
    assert restored_signals["doge"]["signal_hour"] == doge_signal_hour
    assert restored_signals["aave"]["signal_hour"] == aave_signal_hour
    assert restored_entry["signal_hour"] == aave_signal_hour


def test_merge_state_with_snapshot_fills_missing_restart_fields() -> None:
    state = meta.MetaState(active_model="", in_position=False, open_ts=None, open_price=0.0)

    merged = meta._merge_state_with_snapshot(
        state,
        {
            "active_model": "aave",
            "in_position": True,
            "open_ts": "2026-03-06T13:25:40+00:00",
            "open_price": 114.5,
            "position_side": "short",
        },
    )

    assert merged.active_model == "aave"
    assert merged.in_position is True
    assert merged.open_ts == "2026-03-06T13:25:40+00:00"
    assert merged.open_price == pytest.approx(114.5)
    assert merged.position_side == "short"


def test_place_direct_entry_uses_auto_borrow_repay_for_short_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_create_margin_order(symbol, side, order_type, qty, **kwargs):
        captured.update(
            {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "qty": qty,
                **kwargs,
            }
        )
        return {"orderId": 1234}

    monkeypatch.setattr(meta, "create_margin_order", fake_create_margin_order)

    order = meta._place_direct_entry(
        "DOGEUSDT",
        qty=60.0,
        price=0.09318,
        rules=SimpleNamespace(tick_size=0.00001, step_size=1.0, min_notional=5.0),
        entry_side="sell",
    )

    assert captured["side"] == "SELL"
    assert captured["side_effect_type"] == "AUTO_BORROW_REPAY"
    assert order == {
        "id": 1234,
        "price": pytest.approx(0.09318),
        "qty": pytest.approx(60.0),
        "symbol": "DOGEUSDT",
        "side": "sell",
        "kind": "short_entry",
    }


def test_place_direct_exit_uses_buy_for_short_exit_without_force_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_create_margin_order(symbol, side, order_type, qty, **kwargs):
        captured.update(
            {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "qty": qty,
                **kwargs,
            }
        )
        return {"orderId": 9876}

    monkeypatch.setattr(meta, "create_margin_order", fake_create_margin_order)

    order = meta._place_direct_exit(
        "AAVEUSDT",
        qty=0.0449,
        price=114.50,
        rules=SimpleNamespace(tick_size=0.01, step_size=0.001, min_notional=5.0),
        position_side="short",
    )

    assert captured["side"] == "BUY"
    assert captured["side_effect_type"] == "AUTO_REPAY"
    assert order == {
        "id": 9876,
        "price": pytest.approx(114.50),
        "qty": pytest.approx(0.044),
        "symbol": "AAVEUSDT",
        "side": "buy",
        "kind": "exit",
    }


def test_recover_entry_orders_preserves_exchange_order_hour(monkeypatch: pytest.MonkeyPatch) -> None:
    order_hour = meta.pd.Timestamp("2026-03-06 13:00:00+00:00")
    monkeypatch.setattr(
        meta,
        "_list_open_side_orders",
        lambda symbol, side: [
            {
                "orderId": "901",
                "price": "114.50",
                "origQty": "0.044",
                "executedQty": "0",
                "time": int(order_hour.timestamp() * 1000),
            }
        ],
    )

    recovered = meta._recover_entry_orders("AAVEUSDT")

    assert recovered["id"] == 901
    assert recovered["signal_hour"] == order_hour
