from __future__ import annotations

from types import SimpleNamespace

import pytest

import unified_hourly_experiment.trade_unified_hourly as live
import unified_hourly_experiment.trade_unified_hourly_meta as meta_mod
from unified_hourly_experiment.trade_unified_hourly_meta import apply_live_sizing_overrides


def test_apply_live_sizing_overrides_updates_live_globals() -> None:
    missing = object()
    names = (
        "TRADE_AMOUNT_SCALE",
        "MIN_BUY_AMOUNT",
        "ENTRY_INTENSITY_POWER",
        "ENTRY_MIN_INTENSITY_FRACTION",
        "LONG_INTENSITY_MULTIPLIER",
        "SHORT_INTENSITY_MULTIPLIER",
    )
    old = {name: getattr(live, name, missing) for name in names}
    args = SimpleNamespace(
        trade_amount_scale=150.0,
        min_buy_amount=3.0,
        entry_intensity_power=0.7,
        entry_min_intensity_fraction=0.2,
        long_intensity_multiplier=1.1,
        short_intensity_multiplier=1.9,
    )
    try:
        apply_live_sizing_overrides(args)
        assert live.TRADE_AMOUNT_SCALE == 150.0
        assert live.MIN_BUY_AMOUNT == 3.0
        assert live.ENTRY_INTENSITY_POWER == 0.7
        assert live.ENTRY_MIN_INTENSITY_FRACTION == 0.2
        assert live.LONG_INTENSITY_MULTIPLIER == 1.1
        assert live.SHORT_INTENSITY_MULTIPLIER == 1.9
    finally:
        for name, value in old.items():
            if value is missing:
                if hasattr(live, name):
                    delattr(live, name)
            else:
                setattr(live, name, value)


@pytest.mark.parametrize(
    ("symbol", "expected_is_short", "expected_side_multiplier"),
    [
        ("MTCH", True, 1.7),
        ("NVDA", False, 1.2),
    ],
)
def test_build_meta_signals_passes_side_specific_intensity_settings(
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
    expected_is_short: bool,
    expected_side_multiplier: float,
) -> None:
    captured: dict[str, float | bool] = {}

    monkeypatch.setattr(
        meta_mod,
        "load_symbol_latest_action",
        lambda **_: {
            "buy_price": 100.0,
            "sell_price": 101.0,
            "buy_amount": 12.0,
            "sell_amount": 34.0,
            "hold_hours": 5.0,
        },
    )
    monkeypatch.setattr(meta_mod, "compute_symbol_edge", lambda **_: 0.02)

    def _fake_entry_intensity_fraction(action, *, is_short, trade_amount_scale, intensity_power, min_intensity_fraction, side_multiplier):
        captured["is_short"] = bool(is_short)
        captured["trade_amount_scale"] = float(trade_amount_scale)
        captured["intensity_power"] = float(intensity_power)
        captured["min_intensity_fraction"] = float(min_intensity_fraction)
        captured["side_multiplier"] = float(side_multiplier)
        return 10.0, 0.25

    monkeypatch.setattr(meta_mod, "entry_intensity_fraction", _fake_entry_intensity_fraction)

    strategy = SimpleNamespace(name="s1")
    args = SimpleNamespace(
        stock_data_root=None,
        stock_cache_root=None,
        meta_history_days=120,
        fee_rate=0.001,
        min_edge=0.001,
        trade_amount_scale=100.0,
        entry_intensity_power=0.8,
        entry_min_intensity_fraction=0.05,
        long_intensity_multiplier=1.2,
        short_intensity_multiplier=1.7,
        max_hold_hours=6,
    )

    signals = meta_mod.build_meta_signals(
        strategies=[strategy],
        symbols=[symbol],
        winners_by_symbol={symbol: "s1"},
        args=args,
        device=None,  # not used by patched loader
    )

    assert symbol in signals
    assert signals[symbol]["meta_strategy"] == "s1"
    assert captured["is_short"] is expected_is_short
    assert captured["trade_amount_scale"] == 100.0
    assert captured["intensity_power"] == 0.8
    assert captured["min_intensity_fraction"] == 0.05
    assert captured["side_multiplier"] == expected_side_multiplier


def test_simulate_symbol_daily_returns_uses_market_order_entry_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, bool] = {}

    class _DummySim:
        def __init__(self) -> None:
            self.equity_curve = meta_mod.pd.Series(
                [100.0, 101.0],
                index=meta_mod.pd.to_datetime(["2026-03-03T15:00:00Z", "2026-03-03T16:00:00Z"], utc=True),
            )

    def _fake_run_portfolio_simulation(bars, actions, config, horizon=1):
        captured["market_order_entry"] = bool(config.market_order_entry)
        return _DummySim()

    monkeypatch.setattr(meta_mod, "run_portfolio_simulation", _fake_run_portfolio_simulation)

    bars = meta_mod.pd.DataFrame(
        [
            {
                "timestamp": meta_mod.pd.Timestamp("2026-03-03T15:00:00Z"),
                "symbol": "MTCH",
                "open": 30.0,
                "high": 31.0,
                "low": 29.0,
                "close": 30.5,
            }
        ]
    )
    actions = meta_mod.pd.DataFrame(
        [
            {
                "timestamp": meta_mod.pd.Timestamp("2026-03-03T15:00:00Z"),
                "symbol": "MTCH",
                "buy_price": 30.2,
                "sell_price": 29.8,
                "buy_amount": 0.0,
                "sell_amount": 100.0,
            }
        ]
    )
    args = SimpleNamespace(
        meta_sim_initial_cash=10_000.0,
        min_edge=0.001,
        max_hold_hours=6,
        no_close_at_eod=False,
        decision_lag_bars=0,
        market_order_entry=True,
        bar_margin=0.0005,
        entry_order_ttl_hours=0,
        leverage=2.0,
        force_close_slippage=0.003,
        no_int_qty=False,
        fee_rate=0.001,
        margin_rate=0.0625,
    )

    out = meta_mod.simulate_symbol_daily_returns(
        symbol="MTCH",
        bars=bars,
        actions=actions,
        args=args,
    )
    assert captured["market_order_entry"] is True
    assert len(out) == 1


def test_run_cycle_logs_meta_events(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict]] = []

    monkeypatch.setattr(live, "is_market_open_now", lambda: True)
    monkeypatch.setattr(live, "save_state", lambda state: None)
    monkeypatch.setattr(live, "log_event", lambda event_type, **fields: events.append((event_type, fields)))
    monkeypatch.setattr(
        meta_mod,
        "collect_daily_returns",
        lambda **kwargs: {"s1": {"NVDA": meta_mod.pd.Series([0.01], index=meta_mod.pd.to_datetime(["2026-03-03"], utc=True))}},
    )
    monkeypatch.setattr(meta_mod, "select_meta_winners", lambda **kwargs: {"NVDA": "s1"})
    monkeypatch.setattr(
        meta_mod,
        "build_meta_signals",
        lambda **kwargs: {"NVDA": {"buy_price": 100.0, "sell_price": 101.0, "edge": 0.02}},
    )

    args = SimpleNamespace(
        ignore_market_hours=False,
        max_hold_hours=6,
        meta_reselect_frequency="daily",
        dry_run=True,
    )
    selection_cache = meta_mod.MetaSelectionCache()

    meta_mod.run_cycle(
        strategies=[SimpleNamespace(name="s1")],
        symbols=["NVDA"],
        args=args,
        device=None,
        api=None,
        state={"positions": {}},
        selection_cache=selection_cache,
    )

    event_types = [event_type for event_type, _ in events]
    assert "meta_cycle_start" in event_types
    assert "meta_winner_cache_refreshed" in event_types
    assert "meta_cycle_complete" in event_types
