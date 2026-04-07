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
