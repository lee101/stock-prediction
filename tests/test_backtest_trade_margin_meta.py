from __future__ import annotations

import os
from types import SimpleNamespace

import pandas as pd
import pytest

from binanceleveragesui.backtest_trade_margin_meta import (
    _resolve_initial_model_name,
    _resolve_checkpoint_forecast_horizons,
    _inventory_blocks_meta_rotation,
    _make_sim_args,
    build_signal_histories_through,
    compute_equity_stats,
    run_meta_backtest,
    resolve_model_initial_state,
    summarize_trades,
)


def test_backtest_trade_margin_meta_disables_live_runtime_logging() -> None:
    assert os.environ.get("MARGIN_META_DISABLE_LOG") == "1"


def test_resolve_checkpoint_forecast_horizons_expands_from_checkpoint_features() -> None:
    resolved = _resolve_checkpoint_forecast_horizons(
        ["chronos_close_delta_h1", "chronos_high_delta_h6", "chronos_low_delta_h6"],
        horizon=1,
    )

    assert resolved == (1, 6)


def test_resolve_initial_model_name_normalizes_empty_and_case() -> None:
    args = SimpleNamespace(initial_model=" AAVE ")

    assert _resolve_initial_model_name(args) == "aave"
    assert _resolve_initial_model_name(SimpleNamespace(initial_model=None)) == ""


def test_make_sim_args_falls_back_to_max_leverage_when_directional_caps_unset() -> None:
    args = SimpleNamespace(
        fee=0.001,
        fill_buffer_pct=0.0,
        initial_cash=10_000.0,
        min_notional=5.0,
        tick_size=0.01,
        step_size=1.0,
        max_hold_hours=6.0,
        max_leverage=2.3,
        long_max_leverage=None,
        short_max_leverage=None,
        margin_hourly_rate=0.0,
        verbose=False,
        use_order_expiry=False,
        reprice_threshold=0.003,
        max_position_notional=5.1,
        allow_short=False,
        expiry_minutes=90,
        max_fill_fraction=0.01,
    )
    rules = SimpleNamespace(min_notional=5.0, tick_size=0.01, step_size=1.0)

    sim_args = _make_sim_args(args, pd.Timestamp("2026-03-01 00:00:00+00:00"), rules=rules)

    assert sim_args.long_max_leverage == pytest.approx(2.3)
    assert sim_args.short_max_leverage == pytest.approx(2.3)


def test_build_signal_histories_through_applies_cutoff_per_model() -> None:
    signals = {
        "doge": {
            pd.Timestamp("2026-03-01 00:00:00+00:00"): {
                "buy_price": 0.09,
                "sell_price": 0.10,
                "buy_amount": 20.0,
                "sell_amount": 0.0,
                "close": 0.095,
                "signal_hour": pd.Timestamp("2026-03-01 00:00:00+00:00"),
            },
            pd.Timestamp("2026-03-01 01:00:00+00:00"): {
                "buy_price": 0.091,
                "sell_price": 0.101,
                "buy_amount": 25.0,
                "sell_amount": 0.0,
                "close": 0.096,
                "signal_hour": pd.Timestamp("2026-03-01 01:00:00+00:00"),
            },
        },
        "aave": {
            pd.Timestamp("2026-03-01 00:00:00+00:00"): {
                "buy_price": 110.0,
                "sell_price": 112.0,
                "buy_amount": 10.0,
                "sell_amount": 0.0,
                "close": 111.0,
                "signal_hour": pd.Timestamp("2026-03-01 00:00:00+00:00"),
            }
        },
    }

    histories = build_signal_histories_through(signals, pd.Timestamp("2026-03-01 00:00:00+00:00"))

    assert histories["doge"].timestamps == ["2026-03-01T00:00:00+00:00"]
    assert histories["aave"].timestamps == ["2026-03-01T00:00:00+00:00"]


def test_compute_equity_stats_reports_return_and_drawdown() -> None:
    trace = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-01 00:00:00+00:00",
                    "2026-03-01 00:05:00+00:00",
                    "2026-03-01 00:10:00+00:00",
                ]
            ),
            "equity": [10_000.0, 10_500.0, 10_250.0],
        }
    )

    stats = compute_equity_stats(trace, 10_000.0)

    assert stats["final_equity"] == 10_250.0
    assert stats["return_pct"] == pytest.approx(2.5)
    assert stats["max_drawdown_pct"] == pytest.approx((10_250.0 - 10_500.0) / 10_500.0 * 100.0)
    assert stats["bars"] == 3


def test_summarize_trades_respects_starting_short_inventory() -> None:
    trades = [
        {"side": "buy", "qty": 3.0},
        {"side": "sell", "qty": 2.0},
        {"side": "buy", "qty": 2.0},
    ]

    summary = summarize_trades(trades, initial_inventory=-3.0)

    assert summary["trade_count"] == 3
    assert summary["short_exit_count"] == 2
    assert summary["short_entry_count"] == 1
    assert summary["long_entry_count"] == 0
    assert summary["long_exit_count"] == 0


def test_resolve_model_initial_state_adjusts_cash_for_starting_short() -> None:
    args = SimpleNamespace(
        initial_model="doge",
        initial_inv=-5.0,
        initial_entry_ts="2026-03-01T00:00:00+00:00",
    )
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-01 00:00:00+00:00",
                    "2026-03-01 00:05:00+00:00",
                ]
            ),
            "close": [100.0, 101.0],
        }
    )

    inv, entry_ts, cash = resolve_model_initial_state(
        args,
        name="doge",
        start_ts=pd.Timestamp("2026-03-01 00:00:00+00:00"),
        bars_5m=bars,
        initial_equity=10_000.0,
    )

    assert inv == pytest.approx(-5.0)
    assert entry_ts == pd.Timestamp("2026-03-01 00:00:00+00:00")
    assert cash == pytest.approx(10_500.0)


def test_resolve_model_initial_state_ignores_other_models() -> None:
    args = SimpleNamespace(
        initial_model="doge",
        initial_inv=5.0,
        initial_entry_ts="2026-03-01T00:00:00+00:00",
    )
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-01 00:00:00+00:00"]),
            "close": [100.0],
        }
    )

    inv, entry_ts, cash = resolve_model_initial_state(
        args,
        name="aave",
        start_ts=pd.Timestamp("2026-03-01 00:00:00+00:00"),
        bars_5m=bars,
        initial_equity=10_000.0,
    )

    assert inv == pytest.approx(0.0)
    assert entry_ts is None
    assert cash == pytest.approx(10_000.0)


def test_resolve_model_initial_state_defaults_entry_ts_to_window_start() -> None:
    args = SimpleNamespace(
        initial_model="doge",
        initial_inv=5.0,
        initial_entry_ts=None,
    )
    start_ts = pd.Timestamp("2026-03-01 00:00:00+00:00")
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-01 00:00:00+00:00"]),
            "close": [100.0],
        }
    )

    inv, entry_ts, cash = resolve_model_initial_state(
        args,
        name="doge",
        start_ts=start_ts,
        bars_5m=bars,
        initial_equity=10_000.0,
    )

    assert inv == pytest.approx(5.0)
    assert entry_ts == start_ts
    assert cash == pytest.approx(9_500.0)


def test_inventory_blocks_meta_rotation_uses_live_effective_position_threshold() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-01 00:00:00+00:00"]),
            "close": [100.0],
        }
    )
    args = SimpleNamespace(max_position_notional=None)
    rules = SimpleNamespace(step_size=0.001)

    assert _inventory_blocks_meta_rotation(
        args,
        inventory=0.04,
        bars_5m=bars,
        ts=pd.Timestamp("2026-03-01 00:00:00+00:00"),
        rules=rules,
    ) is False
    assert _inventory_blocks_meta_rotation(
        args,
        inventory=0.06,
        bars_5m=bars,
        ts=pd.Timestamp("2026-03-01 00:00:00+00:00"),
        rules=rules,
    ) is True


def test_run_meta_backtest_bootstraps_flat_active_model(monkeypatch: pytest.MonkeyPatch) -> None:
    start_ts = pd.Timestamp("2026-03-01 00:00:00+00:00")
    end_ts = pd.Timestamp("2026-03-01 00:10:00+00:00")
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-01 00:00:00+00:00",
                    "2026-03-01 00:05:00+00:00",
                    "2026-03-01 00:10:00+00:00",
                ]
            ),
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [10.0, 10.0, 10.0],
        }
    )
    signal = {
        start_ts.floor("h"): {
            "buy_price": 99.0,
            "sell_price": 101.0,
            "buy_amount": 0.0,
            "sell_amount": 0.0,
            "close": 100.0,
            "signal_hour": start_ts.floor("h"),
        }
    }
    args = SimpleNamespace(
        initial_cash=10_000.0,
        initial_model="aave",
        initial_inv=0.0,
        initial_entry_ts=None,
        fee=0.001,
        fill_buffer_pct=0.0,
        min_notional=5.0,
        tick_size=0.01,
        step_size=1.0,
        max_hold_hours=6.0,
        max_leverage=2.3,
        long_max_leverage=2.3,
        short_max_leverage=0.16,
        margin_hourly_rate=0.0,
        verbose=False,
        use_order_expiry=False,
        reprice_threshold=0.003,
        max_position_notional=None,
        allow_short=True,
        expiry_minutes=90,
        max_fill_fraction=0.01,
        lookback=1,
        selection_mode="winner_cash",
        selection_metric="omega",
        cash_threshold=0.0,
        switch_margin=0.0,
        min_score_gap=0.0,
        profit_gate_lookback_hours=24,
        profit_gate_min_return=0.0,
    )

    def fake_simulate(*_args, **_kwargs):
        trace = pd.DataFrame(
            {
                "timestamp": bars["timestamp"],
                "equity": [10_000.0, 10_000.0, 10_000.0],
            }
        )
        return [], 10_000.0, 10_000.0, 0.0, trace

    monkeypatch.setattr("binanceleveragesui.backtest_trade_margin_meta.simulate_5m_with_trace", fake_simulate)

    result = run_meta_backtest(
        args,
        start_ts,
        end_ts,
        signal_maps={"doge": signal, "aave": signal},
        bars_by_model={"doge": bars, "aave": bars},
        rules_by_model={
            "doge": SimpleNamespace(min_notional=5.0, tick_size=0.01, step_size=1.0),
            "aave": SimpleNamespace(min_notional=5.0, tick_size=0.01, step_size=1.0),
        },
    )

    assert result["summary"]["switch_count"] == 0
    assert result["summary"]["segments"][0]["model"] == "aave"
    assert result["summary"]["segments"][0]["initial_inventory"] == pytest.approx(0.0)


def test_run_meta_backtest_ignores_sub_threshold_inventory_for_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_ts = pd.Timestamp("2026-03-01 00:00:00+00:00")
    end_ts = pd.Timestamp("2026-03-01 00:10:00+00:00")
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-01 00:00:00+00:00",
                    "2026-03-01 00:05:00+00:00",
                    "2026-03-01 00:10:00+00:00",
                ]
            ),
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [10.0, 10.0, 10.0],
        }
    )
    signal = {
        start_ts.floor("h"): {
            "buy_price": 99.0,
            "sell_price": 101.0,
            "buy_amount": 0.0,
            "sell_amount": 0.0,
            "close": 100.0,
            "signal_hour": start_ts.floor("h"),
        }
    }
    args = SimpleNamespace(
        initial_cash=10_000.0,
        initial_model="aave",
        initial_inv=0.04,
        initial_entry_ts=None,
        fee=0.001,
        fill_buffer_pct=0.0,
        min_notional=5.0,
        tick_size=0.01,
        step_size=1.0,
        max_hold_hours=6.0,
        max_leverage=2.3,
        long_max_leverage=2.3,
        short_max_leverage=0.16,
        margin_hourly_rate=0.0,
        verbose=False,
        use_order_expiry=False,
        reprice_threshold=0.003,
        max_position_notional=None,
        allow_short=True,
        expiry_minutes=90,
        max_fill_fraction=0.01,
        lookback=1,
        selection_mode="winner_cash",
        selection_metric="omega",
        cash_threshold=0.0,
        switch_margin=0.0,
        min_score_gap=0.0,
        profit_gate_lookback_hours=24,
        profit_gate_min_return=0.0,
    )
    call_count = {"value": 0}

    def fake_simulate(*_args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            trace = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        [
                            "2026-03-01 00:00:00+00:00",
                            "2026-03-01 00:05:00+00:00",
                        ]
                    ),
                        "equity": [10_000.0, 10_000.0],
                    }
                )
            return [{"side": "sell", "qty": 0.01}], 10_000.0, 9_996.0, 0.04, trace
        trace = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-03-01 00:10:00+00:00",
                    ]
                ),
                "equity": [10_000.0],
            }
        )
        return [], 10_000.0, 10_000.0, 0.0, trace

    monkeypatch.setattr("binanceleveragesui.backtest_trade_margin_meta.simulate_5m_with_trace", fake_simulate)
    monkeypatch.setattr("binanceleveragesui.backtest_trade_margin_meta.live_meta.select_model", lambda *_args, **_kwargs: "doge")
    monkeypatch.setattr(
        "binanceleveragesui.backtest_trade_margin_meta.live_meta._run_hypothetical_score",
        lambda *_args, **_kwargs: 1.0,
    )

    result = run_meta_backtest(
        args,
        start_ts,
        end_ts,
        signal_maps={"doge": signal, "aave": signal},
        bars_by_model={"doge": bars, "aave": bars},
        rules_by_model={
            "doge": SimpleNamespace(min_notional=5.0, tick_size=0.01, step_size=0.001),
            "aave": SimpleNamespace(min_notional=5.0, tick_size=0.01, step_size=0.001),
        },
    )

    assert call_count["value"] == 2
    assert result["summary"]["switch_count"] == 1
    assert [segment["model"] for segment in result["summary"]["segments"]] == ["aave", "doge"]
