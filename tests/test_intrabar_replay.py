"""Tests for the intra-bar hourly fill simulator.

These tests build synthetic HourlyOHLC + MktdData by hand so we can pin
exactly which hour a fill should fire on. The point is correctness of the
intra-bar walk, not the realism of the synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pufferlib_market.hourly_replay import MktdData
from pufferlib_market.intrabar_replay import (
    HourlyOHLC,
    IntraBarFill,
    build_hourly_marketsim_trace,
    replay_intrabar,
    simulate_daily_policy_intrabar,
)


def _make_mktd(num_days: int = 2, num_symbols: int = 1) -> MktdData:
    T, S, F = num_days, num_symbols, 16
    features = np.zeros((T, S, F), dtype=np.float32)
    prices = np.zeros((T, S, 5), dtype=np.float32)
    prices[..., 0] = 100.0  # open
    prices[..., 1] = 101.0  # high
    prices[..., 2] = 99.0  # low
    prices[..., 3] = 100.0  # close
    prices[..., 4] = 1.0  # vol
    tradable = np.ones((T, S), dtype=np.uint8)
    return MktdData(
        version=2,
        symbols=[f"SYM{i}" for i in range(S)],
        features=features,
        prices=prices,
        tradable=tradable,
    )


def _make_hourly(
    *,
    num_days: int,
    sym: str,
    bars: list[tuple[float, float, float, float]],  # (o,h,l,c) per hour, length num_days*24
    start: str = "2026-01-01",
) -> HourlyOHLC:
    H = num_days * 24
    assert len(bars) == H
    index = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=H, freq="h")
    o = np.array([b[0] for b in bars], dtype=np.float64)
    h = np.array([b[1] for b in bars], dtype=np.float64)
    l = np.array([b[2] for b in bars], dtype=np.float64)
    c = np.array([b[3] for b in bars], dtype=np.float64)
    tr = np.ones(H, dtype=bool)
    return HourlyOHLC(
        index=index,
        symbols=[sym],
        open={sym: o},
        high={sym: h},
        low={sym: l},
        close={sym: c},
        tradable={sym: tr},
    )


def test_long_entry_at_first_tradable_hour() -> None:
    data = _make_mktd(num_days=1, num_symbols=1)
    bars = [(100.0, 101.0, 99.0, 100.0)] * 24
    hourly = _make_hourly(num_days=1, sym="SYM0", bars=bars)
    actions = np.array([1], dtype=np.int32)  # long sym0

    res = replay_intrabar(
        data=data,
        actions=actions,
        hourly=hourly,
        start_date="2026-01-01",
        max_steps=1,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
    )
    entries = [f for f in res.fills if f.kind == "entry"]
    assert len(entries) == 1
    assert entries[0].hourly_idx == 0
    assert abs(entries[0].price - 100.0) < 1e-6


def test_stop_loss_fires_at_exact_hour() -> None:
    """Long opens at hour 0 @100. Stop=2%. At hour 7 the bar dips to 97 (crossing 98)."""
    data = _make_mktd(num_days=1, num_symbols=1)
    bars: list[tuple[float, float, float, float]] = []
    for hi in range(24):
        if hi == 7:
            bars.append((100.0, 100.5, 97.0, 99.0))  # low crosses the 98 stop
        else:
            bars.append((100.0, 100.5, 99.5, 100.0))
    hourly = _make_hourly(num_days=1, sym="SYM0", bars=bars)
    actions = np.array([1], dtype=np.int32)

    res = replay_intrabar(
        data=data,
        actions=actions,
        hourly=hourly,
        start_date="2026-01-01",
        max_steps=1,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        stop_loss_pct=0.02,
    )
    stops = [f for f in res.fills if f.kind == "stop"]
    assert len(stops) == 1, f"expected 1 stop, got fills={res.fills}"
    assert stops[0].hourly_idx == 7
    assert abs(stops[0].price - 98.0) < 1e-6
    # PnL should be roughly -2% on the position (close uses stop price exactly).
    assert res.total_return < -0.015
    assert res.total_return > -0.025


def test_take_profit_fires_at_exact_hour_short() -> None:
    """Short opens at hour 0 @100. TP=2%. Hour 5 dips to 97 (crossing 98 from above)."""
    data = _make_mktd(num_days=1, num_symbols=1)
    bars: list[tuple[float, float, float, float]] = []
    for hi in range(24):
        if hi == 5:
            bars.append((100.0, 100.0, 97.0, 99.0))
        else:
            bars.append((100.0, 100.5, 99.8, 100.0))
    hourly = _make_hourly(num_days=1, sym="SYM0", bars=bars)
    actions = np.array([2], dtype=np.int32)  # short sym0

    res = replay_intrabar(
        data=data,
        actions=actions,
        hourly=hourly,
        start_date="2026-01-01",
        max_steps=1,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        take_profit_pct=0.02,
    )
    tps = [f for f in res.fills if f.kind == "take_profit"]
    assert len(tps) == 1
    assert tps[0].hourly_idx == 5
    assert abs(tps[0].price - 98.0) < 1e-6
    assert res.total_return > 0.015


def test_max_hold_hours_force_exit() -> None:
    data = _make_mktd(num_days=1, num_symbols=1)
    bars = [(100.0, 100.5, 99.5, 100.0)] * 24
    hourly = _make_hourly(num_days=1, sym="SYM0", bars=bars)
    actions = np.array([1], dtype=np.int32)

    res = replay_intrabar(
        data=data,
        actions=actions,
        hourly=hourly,
        start_date="2026-01-01",
        max_steps=1,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        max_hold_hours=4,
    )
    holds = [f for f in res.fills if f.kind == "max_hold"]
    assert len(holds) == 1
    assert holds[0].hourly_idx == 4


def test_two_day_oscillation_stop_then_reentry() -> None:
    """Day 1 enters long, gets stopped at hour 7. Day 2 enters long again, holds."""
    data = _make_mktd(num_days=2, num_symbols=1)
    bars: list[tuple[float, float, float, float]] = []
    for di in range(2):
        for hi in range(24):
            if di == 0 and hi == 7:
                bars.append((100.0, 100.5, 97.0, 99.0))
            else:
                bars.append((100.0, 100.5, 99.5, 100.0))
    hourly = _make_hourly(num_days=2, sym="SYM0", bars=bars)
    actions = np.array([1, 1], dtype=np.int32)

    res = replay_intrabar(
        data=data,
        actions=actions,
        hourly=hourly,
        start_date="2026-01-01",
        max_steps=2,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        stop_loss_pct=0.02,
    )
    kinds = [f.kind for f in res.fills]
    # Expect day1: entry, stop, day2: entry, final_exit (synthetic close).
    assert kinds.count("entry") == 2
    assert kinds.count("stop") == 1


def test_marketsim_trace_adapter() -> None:
    data = _make_mktd(num_days=1, num_symbols=1)
    bars = [(100.0, 100.5, 99.5, 100.0)] * 24
    hourly = _make_hourly(num_days=1, sym="SYM0", bars=bars)
    actions = np.array([1], dtype=np.int32)
    res = replay_intrabar(
        data=data,
        actions=actions,
        hourly=hourly,
        start_date="2026-01-01",
        max_steps=1,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
    )
    trace = build_hourly_marketsim_trace(
        hourly=hourly,
        fills=res.fills,
        equity_curve=res.equity_curve,
        initial_equity=res.initial_equity,
    )
    assert trace.num_steps() == 24
    assert trace.prices_ohlc is not None
    assert trace.prices_ohlc.shape == (24, 1, 4)
    # At least one frame should carry an OrderTick from the entry.
    assert any(len(f.orders) > 0 for f in trace.frames)


def test_render_mp4_smoke(tmp_path: Path) -> None:
    pytest.importorskip("imageio")
    pytest.importorskip("matplotlib")
    from src.marketsim_video import render_mp4

    data = _make_mktd(num_days=1, num_symbols=1)
    bars: list[tuple[float, float, float, float]] = []
    for hi in range(24):
        if hi == 10:
            bars.append((100.0, 100.5, 97.0, 99.0))
        else:
            bars.append((100.0, 100.5, 99.5, 100.0))
    hourly = _make_hourly(num_days=1, sym="SYM0", bars=bars)
    actions = np.array([1], dtype=np.int32)
    res = replay_intrabar(
        data=data,
        actions=actions,
        hourly=hourly,
        start_date="2026-01-01",
        max_steps=1,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        stop_loss_pct=0.02,
    )
    trace = build_hourly_marketsim_trace(
        hourly=hourly,
        fills=res.fills,
        equity_curve=res.equity_curve,
        initial_equity=res.initial_equity,
    )
    out = tmp_path / "test.mp4"
    render_mp4(trace, out, num_pairs=1, fps=4, frames_per_bar=1, title="intrabar test")
    assert out.exists()
    assert out.stat().st_size > 1024


def test_simulate_daily_policy_intrabar_hold_days_zero_on_day_after_buying() -> None:
    """hold_days in obs[base+3] must be 0 on the first decision after buying.

    Bug: `day_idx - pos_open_day` was used, giving 1 on day-1-after-buying,
    but C env shows hold_hours=0 on that step.  Fix: subtract 1 from elapsed days.
    """
    # 3 daily bars, 1 symbol, flat price 100
    data = _make_mktd(num_days=3, num_symbols=1)

    # 3 days with 2 hourly bars each (10am and 11am), spread across 3 calendar days.
    ts_d0 = pd.Timestamp("2026-01-01T10:00:00Z")
    ts_d1 = pd.Timestamp("2026-01-02T10:00:00Z")
    ts_d2 = pd.Timestamp("2026-01-03T10:00:00Z")
    hourly_index = pd.DatetimeIndex(
        [ts_d0, ts_d0 + pd.Timedelta(hours=1),
         ts_d1, ts_d1 + pd.Timedelta(hours=1),
         ts_d2, ts_d2 + pd.Timedelta(hours=1)],
        tz="UTC",
    )
    N = len(hourly_index)
    hourly = HourlyOHLC(
        index=hourly_index,
        symbols=["SYM0"],
        open={"SYM0": np.full(N, 100.0)},
        high={"SYM0": np.full(N, 101.0)},
        low={"SYM0": np.full(N, 99.0)},
        close={"SYM0": np.full(N, 100.0)},
        tradable={"SYM0": np.ones(N, dtype=bool)},
    )

    # Capture hold_days from obs[base+3] at each policy call.
    # S=1, F=16 → base=16, obs[19]=hold_days/max_steps.
    S, F = 1, 16
    base = S * F
    max_steps = 2
    hold_days_obs: list[float] = []
    call_count = [0]

    def _policy(obs: np.ndarray) -> int:
        hold_days_obs.append(float(obs[base + 3]))
        call_count[0] += 1
        return 1  # always buy SYM0

    simulate_daily_policy_intrabar(
        data=data,
        policy_fn=_policy,
        hourly=hourly,
        start_date="2026-01-01",
        max_steps=max_steps,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        max_leverage=1.0,
        periods_per_year=8760.0,
    )

    assert call_count[0] >= 2, f"expected >=2 policy calls, got {call_count[0]}"

    # Day 0 (buy): hold_days=0 → obs shows 0.0
    assert hold_days_obs[0] == pytest.approx(0.0 / max_steps), (
        f"day 0 buy: expected hold_days=0/max_steps=0.0, got {hold_days_obs[0]:.4f}"
    )

    # Day 1 (first day after buying): C env shows hold_hours=0 → hold_days must be 0.
    # Bug would give day_idx(1) - pos_open_day(0) = 1 → obs shows 1/max_steps = 0.5
    assert hold_days_obs[1] == pytest.approx(0.0 / max_steps), (
        f"day 1 after buying: expected hold_days=0/max_steps=0.0, got {hold_days_obs[1]:.4f} "
        "(off-by-one: should be 0 not 1)"
    )


def test_stop_loss_gap_through_fills_at_bar_open_for_long() -> None:
    """When a long gap-opens below the stop, fill at bar_open (worse than stop)."""
    data = _make_mktd(num_days=2, num_symbols=1)
    # Day 0 hour 0: enter long at 100
    # Hour 24 (day 1 hour 0): HUGE gap down — bar_open=90, stop would be 98
    # Stop should fire but fill at 90, not 98.
    bars = [(100.0, 101.0, 99.0, 100.0)] * 24
    bars += [(90.0, 91.0, 89.0, 90.0)]  # gap-down bar: open below stop
    bars += [(90.0, 90.5, 89.5, 90.0)] * 23
    hourly = _make_hourly(num_days=2, sym="SYM0", bars=bars)
    actions = np.array([1, 0], dtype=np.int32)  # long sym0 day 0, flat day 1

    res = replay_intrabar(
        data=data,
        actions=actions,
        hourly=hourly,
        start_date="2026-01-01",
        max_steps=2,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        max_leverage=1.0,
        stop_loss_pct=0.02,  # 2% stop: entry=100 → stop=98
    )

    stop_fills = [f for f in res.fills if f.kind == "stop"]
    assert len(stop_fills) == 1, f"expected 1 stop, got {[f.kind for f in res.fills]}"
    # Gap-through should fill at bar_open=90, NOT the stop level 98.
    assert stop_fills[0].price == pytest.approx(90.0), (
        f"gap-through-stop should fill at bar_open=90 (worse than stop=98), "
        f"got {stop_fills[0].price}"
    )


def test_stop_loss_intrabar_cross_fills_at_stop_level_for_long() -> None:
    """When the bar opens above the stop but wicks through it, fill at stop (not worse)."""
    data = _make_mktd(num_days=2, num_symbols=1)
    bars = [(100.0, 101.0, 99.0, 100.0)] * 24
    # Day 1 hour 0: open 99, low 97 (wicks to 97 — passes stop=98 from above)
    bars += [(99.0, 99.5, 97.0, 99.0)]
    bars += [(99.0, 99.5, 98.5, 99.0)] * 23
    hourly = _make_hourly(num_days=2, sym="SYM0", bars=bars)
    actions = np.array([1, 0], dtype=np.int32)

    res = replay_intrabar(
        data=data,
        actions=actions,
        hourly=hourly,
        start_date="2026-01-01",
        max_steps=2,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        max_leverage=1.0,
        stop_loss_pct=0.02,
    )

    stop_fills = [f for f in res.fills if f.kind == "stop"]
    assert len(stop_fills) == 1
    # Intrabar crossing: fill at the stop level exactly.
    assert stop_fills[0].price == pytest.approx(98.0), (
        f"intrabar crossing should fill at stop=98, got {stop_fills[0].price}"
    )


def test_replay_intrabar_short_borrow_reduces_cash() -> None:
    """replay_intrabar now applies short-borrow carry each hour a short is open."""
    data = _make_mktd(num_days=2, num_symbols=1)
    # Flat prices the entire run: PnL from price movement is exactly zero.
    bars = [(100.0, 100.01, 99.99, 100.0)] * 48
    hourly = _make_hourly(num_days=2, sym="SYM0", bars=bars)
    actions = np.array([2, 0], dtype=np.int32)  # short day 0, flat day 1

    res_no_borrow = replay_intrabar(
        data=data,
        actions=actions,
        hourly=hourly,
        start_date="2026-01-01",
        max_steps=2,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        max_leverage=1.0,
        short_borrow_apr=0.0,
    )
    res_with_borrow = replay_intrabar(
        data=data,
        actions=actions,
        hourly=hourly,
        start_date="2026-01-01",
        max_steps=2,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        max_leverage=1.0,
        short_borrow_apr=0.30,  # 30% APR (chunky for signal)
    )

    assert res_with_borrow.final_equity < res_no_borrow.final_equity, (
        f"short borrow should reduce final equity: "
        f"with={res_with_borrow.final_equity} vs without={res_no_borrow.final_equity}"
    )
    # Ballpark: 24h of 30% APR on ~$1 notional ≈ 0.00082 ≈ a few bps of $10k.
    delta = res_no_borrow.final_equity - res_with_borrow.final_equity
    assert delta > 0.0, f"expected positive borrow cost, got delta={delta}"


def test_stop_loss_gap_through_fills_at_bar_open_for_short() -> None:
    """When a short gap-opens above the stop, fill at bar_open (worse than stop)."""
    data = _make_mktd(num_days=2, num_symbols=1)
    # Enter short at 100. Stop = 102 (2% up).
    # Day 1 hour 0: gap up to 110 — open > stop.
    bars = [(100.0, 101.0, 99.0, 100.0)] * 24
    bars += [(110.0, 111.0, 109.0, 110.0)]
    bars += [(110.0, 110.5, 109.5, 110.0)] * 23
    hourly = _make_hourly(num_days=2, sym="SYM0", bars=bars)
    # action 2 = short sym0 (S=1 so S+1..2S = action 2)
    actions = np.array([2, 0], dtype=np.int32)

    res = replay_intrabar(
        data=data,
        actions=actions,
        hourly=hourly,
        start_date="2026-01-01",
        max_steps=2,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        max_leverage=1.0,
        stop_loss_pct=0.02,
    )

    stop_fills = [f for f in res.fills if f.kind == "stop"]
    assert len(stop_fills) == 1, f"expected 1 stop, got {[f.kind for f in res.fills]}"
    # Gap-through for short: fill at bar_open=110, not stop=102.
    assert stop_fills[0].price == pytest.approx(110.0), (
        f"short gap-through-stop should fill at bar_open=110 (worse than stop=102), "
        f"got {stop_fills[0].price}"
    )
