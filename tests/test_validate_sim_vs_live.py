from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from binanceleveragesui.validate_sim_vs_live import (
    effective_position_side_from_qty,
    load_5m_bars,
    match_trades,
    reconstruct_initial_state,
    resolve_initial_replay_state,
    simulate_5m,
    simulate_5m_with_trace,
)


def _base_args(**overrides):
    args = {
        "fee": 0.001,
        "fill_buffer_pct": 0.0,
        "initial_cash": 10_000.0,
        "start": "2026-01-01 01:00:00+00:00",
        "realistic": True,
        "expiry_minutes": 5,
        "max_fill_fraction": 1.0,
        "min_notional": 5.0,
        "tick_size": 0.01,
        "step_size": 1.0,
        "max_hold_hours": 6,
        "max_leverage": 1.0,
        "margin_hourly_rate": 0.0,
        "verbose": False,
        "live_like": True,
        "use_order_expiry": False,
        "reprice_threshold": 0.003,
        "max_position_notional": None,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_match_trades_treats_force_sell_as_sell():
    prod_fills = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 01:00:00+00:00"),
                "side": "sell",
                "avg_price": 100.0,
                "total_qty": 10.0,
            }
        ]
    )
    sim_trades = [
        {
            "ts": pd.Timestamp("2026-01-01 01:05:00+00:00"),
            "side": "force_sell",
            "price": 100.1,
            "qty": 10.0,
        }
    ]
    matches, unmatched = match_trades(prod_fills, sim_trades)
    assert len(matches) == 1
    assert matches[0]["matched"] is True
    assert len(unmatched) == 0


def test_match_trades_treats_force_buy_as_buy():
    prod_fills = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 01:00:00+00:00"),
                "side": "buy",
                "avg_price": 100.0,
                "total_qty": 10.0,
            }
        ]
    )
    sim_trades = [
        {
            "ts": pd.Timestamp("2026-01-01 01:05:00+00:00"),
            "side": "force_buy",
            "price": 100.1,
            "qty": 10.0,
        }
    ]
    matches, unmatched = match_trades(prod_fills, sim_trades)
    assert len(matches) == 1
    assert matches[0]["matched"] is True
    assert len(unmatched) == 0


def test_load_5m_bars_uses_trade_margin_meta_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:00:00+00:00"]),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1_000.0],
        }
    )
    calls: list[tuple[str, pd.Timestamp, pd.Timestamp, object, object]] = []

    def fake_load(symbol, start_ts, end_ts, *, args=None):
        calls.append(
            (
                symbol,
                pd.Timestamp(start_ts),
                pd.Timestamp(end_ts),
                getattr(args, "data_root", None),
                getattr(args, "profit_gate_5m_root", None),
            )
        )
        return expected

    monkeypatch.setattr("binanceleveragesui.validate_sim_vs_live._load_profit_gate_5m_bars", fake_load)

    loaded = load_5m_bars(
        "DOGEUSDT",
        pd.Timestamp("2026-01-01 00:00:00+00:00"),
        pd.Timestamp("2026-01-01 00:00:00+00:00"),
        data_root="/tmp/hourly",
        bars_5m_root="/tmp/5m",
    )

    assert calls == [
        (
            "DOGEUSDT",
            pd.Timestamp("2026-01-01 00:00:00+00:00"),
            pd.Timestamp("2026-01-01 00:00:00+00:00"),
            "/tmp/hourly",
            "/tmp/5m",
        )
    ]
    assert loaded.equals(expected)


def test_match_trades_marks_prod_rows_unmatched_when_sim_is_empty():
    prod_fills = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 01:00:00+00:00"),
                "side": "buy",
                "avg_price": 100.0,
                "total_qty": 10.0,
            }
        ]
    )

    matches, unmatched = match_trades(prod_fills, [])

    assert matches == [
        {
            "prod_ts": pd.Timestamp("2026-01-01 01:00:00+00:00"),
            "side": "buy",
            "prod_price": 100.0,
            "prod_qty": 10.0,
            "matched": False,
        }
    ]
    assert unmatched.empty


def test_effective_position_side_from_qty_ignores_sub_threshold_notional():
    assert effective_position_side_from_qty(
        0.04,
        market_price=100.0,
        step_size=0.001,
        max_position_notional=None,
    ) == ""
    assert effective_position_side_from_qty(
        -0.06,
        market_price=100.0,
        step_size=0.001,
        max_position_notional=None,
    ) == "short"


def test_live_like_ignores_expiry_by_default():
    bars_5m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 01:00:00+00:00",
                    "2026-01-01 01:05:00+00:00",
                    "2026-01-01 01:10:00+00:00",
                ]
            ),
            "open": [101.0, 101.0, 101.0],
            "high": [101.5, 101.5, 101.5],
            "low": [100.5, 100.5, 99.0],
            "close": [101.0, 101.0, 101.0],
            "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0],
        }
    )
    hourly_signals = {
        pd.Timestamp("2026-01-01 00:00:00+00:00"): {
            "buy_price": 100.0,
            "sell_price": 101.0,
            "buy_amount": 100.0,
            "sell_amount": 0.0,
        }
    }

    args = _base_args(use_order_expiry=False)
    trades, *_ = simulate_5m(args, hourly_signals, bars_5m, initial_inv=0.0, initial_entry_ts=None)
    assert any(t["side"] == "buy" for t in trades)


def test_live_like_treats_sub_threshold_initial_long_as_flat_for_short_entry():
    bars_5m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 01:00:00+00:00",
                    "2026-01-01 01:05:00+00:00",
                ]
            ),
            "open": [100.0, 100.0],
            "high": [101.5, 101.5],
            "low": [99.5, 99.5],
            "close": [100.0, 100.0],
            "volume": [1_000_000.0, 1_000_000.0],
        }
    )
    hourly_signals = {
        pd.Timestamp("2026-01-01 00:00:00+00:00"): {
            "buy_price": 0.0,
            "sell_price": 101.0,
            "buy_amount": 0.0,
            "sell_amount": 100.0,
        }
    }

    args = _base_args(
        fee=0.0,
        initial_cash=1_000.0,
        max_leverage=1.0,
        long_max_leverage=1.0,
        short_max_leverage=1.0,
        allow_short=True,
        min_notional=5.0,
        step_size=0.001,
        tick_size=0.01,
        live_like=True,
        realistic=True,
    )
    trades, _final_eq, _cash, inv = simulate_5m(
        args,
        hourly_signals,
        bars_5m,
        initial_inv=0.04,
        initial_entry_ts=pd.Timestamp("2026-01-01 00:55:00+00:00"),
    )

    assert any(trade["side"] == "sell" for trade in trades)
    assert inv < 0.0


def test_live_like_clears_sub_threshold_residual_after_round_trip():
    bars_5m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 01:00:00+00:00",
                    "2026-01-01 01:05:00+00:00",
                    "2026-01-01 02:00:00+00:00",
                    "2026-01-01 02:05:00+00:00",
                ]
            ),
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.5, 101.5, 100.5, 100.5],
            "low": [99.5, 99.5, 98.5, 98.5],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0],
        }
    )
    hourly_signals = {
        pd.Timestamp("2026-01-01 00:00:00+00:00"): {
            "buy_price": 0.0,
            "sell_price": 101.0,
            "buy_amount": 0.0,
            "sell_amount": 100.0,
        },
        pd.Timestamp("2026-01-01 01:00:00+00:00"): {
            "buy_price": 99.0,
            "sell_price": 0.0,
            "buy_amount": 100.0,
            "sell_amount": 0.0,
        },
    }

    args = _base_args(
        fee=0.0,
        initial_cash=1_000.0,
        max_leverage=1.0,
        long_max_leverage=1.0,
        short_max_leverage=1.0,
        allow_short=True,
        min_notional=5.0,
        step_size=0.001,
        tick_size=0.01,
        live_like=True,
        realistic=True,
    )
    trades, _final_eq, _cash, inv = simulate_5m(
        args,
        hourly_signals,
        bars_5m,
        initial_inv=0.043992,
        initial_entry_ts=pd.Timestamp("2026-01-01 00:55:00+00:00"),
    )

    assert [trade["side"] for trade in trades] == ["sell", "buy"]
    assert inv == pytest.approx(0.0)


def test_live_like_optional_expiry_can_block_fill():
    bars_5m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 01:00:00+00:00",
                    "2026-01-01 01:05:00+00:00",
                    "2026-01-01 01:10:00+00:00",
                ]
            ),
            "open": [101.0, 101.0, 101.0],
            "high": [101.5, 101.5, 101.5],
            "low": [100.5, 100.5, 99.0],
            "close": [101.0, 101.0, 101.0],
            "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0],
        }
    )
    hourly_signals = {
        pd.Timestamp("2026-01-01 00:00:00+00:00"): {
            "buy_price": 100.0,
            "sell_price": 101.0,
            "buy_amount": 100.0,
            "sell_amount": 0.0,
        }
    }

    args = _base_args(use_order_expiry=True)
    trades, *_ = simulate_5m(args, hourly_signals, bars_5m, initial_inv=0.0, initial_entry_ts=None)
    assert not any(t["side"] == "buy" for t in trades)


def test_live_like_keeps_working_entry_order_active_without_fresh_signal():
    bars_5m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 01:00:00+00:00",
                    "2026-01-01 02:00:00+00:00",
                ]
            ),
            "open": [101.0, 101.0],
            "high": [101.5, 101.5],
            "low": [100.5, 99.0],
            "close": [101.0, 100.5],
            "volume": [1_000_000.0, 1_000_000.0],
        }
    )
    hourly_signals = {
        pd.Timestamp("2026-01-01 00:00:00+00:00"): {
            "buy_price": 100.0,
            "sell_price": 102.0,
            "buy_amount": 100.0,
            "sell_amount": 0.0,
        }
    }

    args = _base_args(
        fee=0.0,
        max_leverage=1.0,
        min_notional=1.0,
        step_size=1.0,
        tick_size=0.01,
        live_like=True,
        realistic=True,
    )
    trades, *_ = simulate_5m(args, hourly_signals, bars_5m, initial_inv=0.0, initial_entry_ts=None)

    buys = [trade for trade in trades if trade["side"] == "buy"]
    assert len(buys) == 1
    assert buys[0]["ts"] == pd.Timestamp("2026-01-01 02:00:00+00:00")


def test_resolve_initial_replay_state_uses_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "binanceleveragesui.validate_sim_vs_live.reconstruct_initial_state",
        lambda *args, **kwargs: pytest.fail("reconstruct_initial_state should not be called"),
    )
    args = SimpleNamespace(
        symbol="DOGEUSDT",
        initial_inv=53.0,
        initial_entry_ts="2026-01-01 00:30:00+00:00",
        skip_initial_reconstruction=False,
        initial_reconstruction_lookback_hours=48,
    )

    inv, entry_ts = resolve_initial_replay_state(args, start_ms=0)

    assert inv == pytest.approx(53.0)
    assert entry_ts == pd.Timestamp("2026-01-01 00:30:00+00:00")


def test_resolve_initial_replay_state_preserves_short_override(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "binanceleveragesui.validate_sim_vs_live.reconstruct_initial_state",
        lambda *args, **kwargs: pytest.fail("reconstruct_initial_state should not be called"),
    )
    args = SimpleNamespace(
        symbol="DOGEUSDT",
        initial_inv=-53.0,
        initial_entry_ts="2026-01-01 00:30:00+00:00",
        skip_initial_reconstruction=False,
        initial_reconstruction_lookback_hours=48,
    )

    inv, entry_ts = resolve_initial_replay_state(args, start_ms=0)

    assert inv == pytest.approx(-53.0)
    assert entry_ts == pd.Timestamp("2026-01-01 00:30:00+00:00")


def test_resolve_initial_replay_state_can_start_flat(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "binanceleveragesui.validate_sim_vs_live.reconstruct_initial_state",
        lambda *args, **kwargs: pytest.fail("reconstruct_initial_state should not be called"),
    )
    args = SimpleNamespace(
        symbol="AAVEUSDT",
        initial_inv=None,
        initial_entry_ts=None,
        skip_initial_reconstruction=True,
        initial_reconstruction_lookback_hours=48,
    )

    inv, entry_ts = resolve_initial_replay_state(args, start_ms=0)

    assert inv == pytest.approx(0.0)
    assert entry_ts is None


def test_reconstruct_initial_state_preserves_net_short(
    monkeypatch: pytest.MonkeyPatch,
):
    short_ts = pd.Timestamp("2026-01-01 00:45:00+00:00")
    monkeypatch.setattr(
        "binanceleveragesui.validate_sim_vs_live.get_margin_trades",
        lambda *args, **kwargs: [
            {
                "qty": "12",
                "isBuyer": False,
                "time": int(short_ts.timestamp() * 1000),
            }
        ],
    )

    inv, entry_ts = reconstruct_initial_state("DOGEUSDT", start_ms=int(pd.Timestamp("2026-01-01 01:00:00+00:00").timestamp() * 1000))

    assert inv == pytest.approx(-12.0)
    assert entry_ts == short_ts


@pytest.mark.parametrize("live_like", [False, True])
def test_simulate_5m_respects_max_position_notional(live_like: bool):
    bars_5m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 01:00:00+00:00",
                    "2026-01-01 01:05:00+00:00",
                ]
            ),
            "open": [10.2, 10.2],
            "high": [10.5, 10.5],
            "low": [9.5, 9.5],
            "close": [10.2, 10.2],
            "volume": [1_000_000.0, 1_000_000.0],
        }
    )
    hourly_signals = {
        pd.Timestamp("2026-01-01 00:00:00+00:00"): {
            "buy_price": 10.0,
            "sell_price": 11.0,
            "buy_amount": 100.0,
            "sell_amount": 0.0,
        }
    }

    args = _base_args(
        fee=0.0,
        initial_cash=1_000.0,
        max_leverage=10.0,
        min_notional=1.0,
        tick_size=0.01,
        step_size=1.0,
        live_like=live_like,
        max_position_notional=50.0,
    )
    trades, _final_eq, _cash, inv = simulate_5m(
        args,
        hourly_signals,
        bars_5m,
        initial_inv=0.0,
        initial_entry_ts=None,
    )

    buys = [trade for trade in trades if trade["side"] == "buy"]
    assert len(buys) == 1
    assert buys[0]["qty"] == 5.0
    assert inv == 5.0


def test_simulate_5m_supports_short_entry_and_cover_cycle():
    bars_5m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 01:00:00+00:00",
                    "2026-01-01 01:05:00+00:00",
                    "2026-01-01 02:00:00+00:00",
                    "2026-01-01 02:05:00+00:00",
                ]
            ),
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.5, 101.5, 100.2, 100.2],
            "low": [99.8, 99.8, 99.2, 98.8],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0],
        }
    )
    hourly_signals = {
        pd.Timestamp("2026-01-01 00:00:00+00:00"): {
            "buy_price": 0.0,
            "sell_price": 101.0,
            "buy_amount": 0.0,
            "sell_amount": 100.0,
        },
        pd.Timestamp("2026-01-01 01:00:00+00:00"): {
            "buy_price": 99.0,
            "sell_price": 0.0,
            "buy_amount": 100.0,
            "sell_amount": 0.0,
        },
    }

    args = _base_args(
        fee=0.0,
        initial_cash=1_000.0,
        max_leverage=1.0,
        long_max_leverage=1.0,
        short_max_leverage=1.0,
        allow_short=True,
        min_notional=1.0,
        step_size=1.0,
        tick_size=0.01,
        live_like=True,
        realistic=True,
    )
    trades, final_eq, _cash, inv = simulate_5m(
        args,
        hourly_signals,
        bars_5m,
        initial_inv=0.0,
        initial_entry_ts=None,
    )

    assert [trade["side"] for trade in trades] == ["sell", "buy"]
    assert trades[0]["qty"] == pytest.approx(9.0)
    assert trades[1]["qty"] == pytest.approx(9.0)
    assert inv == pytest.approx(0.0)
    assert final_eq == pytest.approx(1_018.0)


def test_simulate_5m_respects_max_position_notional_for_short_entries():
    bars_5m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 01:00:00+00:00",
                    "2026-01-01 01:05:00+00:00",
                ]
            ),
            "open": [9.8, 9.8],
            "high": [9.9, 10.5],
            "low": [9.5, 9.5],
            "close": [9.8, 9.8],
            "volume": [1_000_000.0, 1_000_000.0],
        }
    )
    hourly_signals = {
        pd.Timestamp("2026-01-01 00:00:00+00:00"): {
            "buy_price": 0.0,
            "sell_price": 10.0,
            "buy_amount": 0.0,
            "sell_amount": 100.0,
        }
    }

    args = _base_args(
        fee=0.0,
        initial_cash=1_000.0,
        max_leverage=10.0,
        long_max_leverage=10.0,
        short_max_leverage=10.0,
        allow_short=True,
        min_notional=1.0,
        tick_size=0.01,
        step_size=1.0,
        live_like=True,
        realistic=True,
        max_position_notional=50.0,
    )
    trades, _final_eq, _cash, inv = simulate_5m(
        args,
        hourly_signals,
        bars_5m,
        initial_inv=0.0,
        initial_entry_ts=None,
    )

    sells = [trade for trade in trades if trade["side"] == "sell"]
    assert len(sells) == 1
    assert sells[0]["qty"] == pytest.approx(5.0)
    assert inv == pytest.approx(-5.0)


def test_simulate_5m_short_entry_sizes_through_small_long_stub():
    bars_5m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 01:00:00+00:00",
                    "2026-01-01 01:05:00+00:00",
                    "2026-01-01 01:10:00+00:00",
                ]
            ),
            "open": [9.8, 9.8, 9.8],
            "high": [10.5, 10.5, 10.5],
            "low": [9.5, 9.5, 9.5],
            "close": [9.8, 9.8, 9.8],
            "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0],
        }
    )
    hourly_signals = {
        pd.Timestamp("2026-01-01 00:00:00+00:00"): {
            "buy_price": 0.0,
            "sell_price": 10.0,
            "buy_amount": 0.0,
            "sell_amount": 100.0,
        }
    }

    args = _base_args(
        fee=0.0,
        initial_cash=1_000.0,
        max_leverage=10.0,
        long_max_leverage=10.0,
        short_max_leverage=10.0,
        allow_short=True,
        min_notional=1.0,
        tick_size=0.01,
        step_size=1.0,
        live_like=True,
        realistic=True,
        max_position_notional=50.0,
    )
    trades, _final_eq, _cash, inv = simulate_5m(
        args,
        hourly_signals,
        bars_5m,
        initial_inv=1.0,
        initial_entry_ts=pd.Timestamp("2026-01-01 00:30:00+00:00"),
    )

    sells = [trade for trade in trades if trade["side"] == "sell"]
    assert [trade["qty"] for trade in sells] == [pytest.approx(1.0), pytest.approx(5.0)]
    assert sum(trade["qty"] for trade in sells) == pytest.approx(6.0)
    assert inv == pytest.approx(-5.0)


def test_simulate_5m_long_entry_sizes_through_small_short_stub():
    bars_5m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 01:00:00+00:00",
                    "2026-01-01 01:05:00+00:00",
                    "2026-01-01 01:10:00+00:00",
                ]
            ),
            "open": [10.2, 10.2, 10.2],
            "high": [10.5, 10.5, 10.5],
            "low": [9.5, 9.5, 9.5],
            "close": [10.2, 10.2, 10.2],
            "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0],
        }
    )
    hourly_signals = {
        pd.Timestamp("2026-01-01 00:00:00+00:00"): {
            "buy_price": 10.0,
            "sell_price": 11.0,
            "buy_amount": 100.0,
            "sell_amount": 0.0,
        }
    }

    args = _base_args(
        fee=0.0,
        initial_cash=1_000.0,
        max_leverage=10.0,
        long_max_leverage=10.0,
        short_max_leverage=10.0,
        allow_short=True,
        min_notional=1.0,
        tick_size=0.01,
        step_size=1.0,
        live_like=True,
        realistic=True,
        max_position_notional=50.0,
    )
    trades, _final_eq, _cash, inv = simulate_5m(
        args,
        hourly_signals,
        bars_5m,
        initial_inv=-1.0,
        initial_entry_ts=pd.Timestamp("2026-01-01 00:30:00+00:00"),
    )

    buys = [trade for trade in trades if trade["side"] == "buy"]
    assert [trade["qty"] for trade in buys] == [pytest.approx(1.0), pytest.approx(5.0)]
    assert sum(trade["qty"] for trade in buys) == pytest.approx(6.0)
    assert inv == pytest.approx(5.0)


def test_simulate_5m_charges_margin_interest_on_open_short():
    bars_5m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 01:00:00+00:00",
                    "2026-01-01 02:00:00+00:00",
                ]
            ),
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
            "volume": [1_000_000.0, 1_000_000.0],
        }
    )

    args = _base_args(
        fee=0.0,
        initial_cash=2_000.0,
        margin_hourly_rate=0.01,
        allow_short=True,
        min_notional=1.0,
        tick_size=0.01,
        step_size=1.0,
        live_like=True,
        realistic=True,
    )
    trades, final_eq, cash, inv = simulate_5m(
        args,
        hourly_signals={},
        bars_5m=bars_5m,
        initial_inv=-10.0,
        initial_entry_ts=pd.Timestamp("2026-01-01 00:00:00+00:00"),
    )

    assert trades == []
    assert cash == pytest.approx(1_990.0)
    assert inv == pytest.approx(-10.0)
    assert final_eq == pytest.approx(990.0)


def test_simulate_5m_with_trace_can_stop_after_completed_cycle():
    bars_5m = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 01:00:00+00:00",
                    "2026-01-01 01:05:00+00:00",
                    "2026-01-01 01:10:00+00:00",
                    "2026-01-01 01:15:00+00:00",
                ]
            ),
            "open": [100.5, 100.0, 101.0, 101.0],
            "high": [100.6, 100.8, 101.5, 101.2],
            "low": [99.5, 99.0, 100.4, 100.8],
            "close": [100.5, 100.0, 101.0, 101.0],
            "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0, 1_000_000.0],
        }
    )
    hourly_signals = {
        pd.Timestamp("2026-01-01 00:00:00+00:00"): {
            "buy_price": 100.0,
            "sell_price": 101.0,
            "buy_amount": 100.0,
            "sell_amount": 100.0,
        }
    }

    args = _base_args(
        fee=0.0,
        max_leverage=1.0,
        min_notional=1.0,
        step_size=1.0,
        tick_size=0.01,
        live_like=True,
        realistic=True,
    )
    trades, final_eq, _cash, inv, trace = simulate_5m_with_trace(
        args,
        hourly_signals,
        bars_5m,
        initial_inv=0.0,
        initial_entry_ts=None,
        stop_after_cycle=True,
    )

    assert [trade["side"] for trade in trades] == ["buy", "sell"]
    assert inv == 0.0
    assert final_eq == pytest.approx(10_100.0)
    assert not trace.empty
    assert trace["timestamp"].iloc[-1] == pd.Timestamp("2026-01-01 01:10:00+00:00")
