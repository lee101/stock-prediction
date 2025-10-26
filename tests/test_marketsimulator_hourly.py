from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytz

from marketsimulator.state import PriceSeries, SimulationState, SimulatedClock


def _build_state(symbol: str = "TEST") -> SimulationState:
    start = pytz.utc.localize(datetime(2024, 1, 1))
    frame = pd.DataFrame(
        [
            {"timestamp": start, "Open": 100.0, "High": 110.0, "Low": 90.0, "Close": 100.0},
            {"timestamp": start + pd.Timedelta(days=1), "Open": 100.0, "High": 110.0, "Low": 90.0, "Close": 100.0},
        ]
    )
    series = PriceSeries(symbol=symbol, frame=frame)
    clock = SimulatedClock(start)
    return SimulationState(clock=clock, prices={symbol: series})


def _stub_hourly(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def test_maxdiff_hourly_repeats_long(monkeypatch):
    symbol = "TEST"
    state = _build_state(symbol)
    hourly_rows = [
        {"timestamp": "2024-01-01T01:00:00Z", "Open": 100.0, "High": 100.0, "Low": 98.0, "Close": 99.0},
        {"timestamp": "2024-01-01T02:00:00Z", "Open": 98.5, "High": 99.0, "Low": 94.5, "Close": 95.0},
        {"timestamp": "2024-01-01T03:00:00Z", "Open": 95.0, "High": 106.0, "Low": 95.0, "Close": 105.0},
        {"timestamp": "2024-01-01T04:00:00Z", "Open": 100.0, "High": 100.5, "Low": 94.4, "Close": 95.2},
        {"timestamp": "2024-01-01T05:00:00Z", "Open": 95.2, "High": 106.0, "Low": 95.0, "Close": 105.5},
    ]
    hourly_frame = _stub_hourly(hourly_rows)
    monkeypatch.setattr("marketsimulator.state.load_hourly_bars", lambda sym: hourly_frame if sym == symbol else pd.DataFrame())

    state.register_maxdiff_entry(symbol, "buy", limit_price=95.0, target_qty=1.0, tolerance_pct=0.0, expiry_minutes=2880)
    state.register_maxdiff_exit(symbol, "buy", takeprofit_price=105.0, expiry_minutes=2880, tolerance_pct=0.0)

    state.advance_time()

    assert len(state.trade_log) == 4
    sides = [trade.side for trade in state.trade_log]
    assert sides == ["buy", "sell", "buy", "sell"]
    assert symbol not in state.positions

    entry_watcher = next(w for w in state.maxdiff_entries if w.symbol == symbol)
    exit_watcher = next(w for w in state.maxdiff_exits if w.symbol == symbol)
    assert entry_watcher.fills == 2
    assert exit_watcher.fills == 2


def test_maxdiff_hourly_repeats_short(monkeypatch):
    symbol = "SHORT"
    state = _build_state(symbol)
    hourly_rows = [
        {"timestamp": "2024-01-01T01:00:00Z", "Open": 100.0, "High": 104.0, "Low": 100.0, "Close": 103.5},
        {"timestamp": "2024-01-01T02:00:00Z", "Open": 103.5, "High": 106.0, "Low": 103.0, "Close": 105.0},
        {"timestamp": "2024-01-01T03:00:00Z", "Open": 105.0, "High": 105.5, "Low": 94.0, "Close": 95.5},
        {"timestamp": "2024-01-01T04:00:00Z", "Open": 95.5, "High": 106.5, "Low": 95.0, "Close": 95.2},
    ]
    hourly_frame = _stub_hourly(hourly_rows)
    monkeypatch.setattr("marketsimulator.state.load_hourly_bars", lambda sym: hourly_frame if sym == symbol else pd.DataFrame())

    state.register_maxdiff_entry(symbol, "sell", limit_price=105.0, target_qty=2.0, tolerance_pct=0.0, expiry_minutes=2880)
    state.register_maxdiff_exit(symbol, "sell", takeprofit_price=95.0, expiry_minutes=2880, tolerance_pct=0.0)

    state.advance_time()

    assert len(state.trade_log) == 4
    sides = [trade.side for trade in state.trade_log]
    assert sides == ["sell", "buy", "sell", "buy"]
    assert symbol not in state.positions

    entry_watcher = next(w for w in state.maxdiff_entries if w.symbol == symbol)
    exit_watcher = next(w for w in state.maxdiff_exits if w.symbol == symbol)
    assert entry_watcher.fills == 2
    assert exit_watcher.fills == 2


def test_maxdiff_hourly_limits_intraday_reentry(monkeypatch):
    symbol = "LIMIT"
    state = _build_state(symbol)
    hourly_rows = [
        {"timestamp": "2024-01-01T01:05:00Z", "Open": 100.0, "High": 100.5, "Low": 94.8, "Close": 95.2},
        {"timestamp": "2024-01-01T01:20:00Z", "Open": 95.2, "High": 105.4, "Low": 95.0, "Close": 104.9},
        {"timestamp": "2024-01-01T01:35:00Z", "Open": 104.9, "High": 105.1, "Low": 94.7, "Close": 95.1},
        {"timestamp": "2024-01-01T02:10:00Z", "Open": 95.1, "High": 95.3, "Low": 94.6, "Close": 94.8},
        {"timestamp": "2024-01-01T02:30:00Z", "Open": 94.8, "High": 105.2, "Low": 94.7, "Close": 105.0},
    ]
    hourly_frame = _stub_hourly(hourly_rows)
    monkeypatch.setattr("marketsimulator.state.load_hourly_bars", lambda sym: hourly_frame if sym == symbol else pd.DataFrame())

    state.register_maxdiff_entry(symbol, "buy", limit_price=95.0, target_qty=1.0, tolerance_pct=0.0, expiry_minutes=2880)
    state.register_maxdiff_exit(symbol, "buy", takeprofit_price=105.0, expiry_minutes=2880, tolerance_pct=0.0)

    state.advance_time()

    sides = [trade.side for trade in state.trade_log]
    assert sides == ["buy", "sell", "buy", "sell"]
    entry_watcher = next(w for w in state.maxdiff_entries if w.symbol == symbol)
    exit_watcher = next(w for w in state.maxdiff_exits if w.symbol == symbol)
    assert entry_watcher.fills == 2
    assert exit_watcher.fills == 2
    assert entry_watcher.last_fill and exit_watcher.last_fill
    assert state.positions == {}


def test_maxdiff_hourly_fallback_to_daily(monkeypatch):
    symbol = "FALL"
    state = _build_state(symbol)
    monkeypatch.setattr("marketsimulator.state.load_hourly_bars", lambda sym: pd.DataFrame())

    state.register_maxdiff_entry(symbol, "buy", limit_price=95.0, target_qty=1.0, tolerance_pct=0.0, expiry_minutes=1440)
    state.register_maxdiff_exit(symbol, "buy", takeprofit_price=105.0, expiry_minutes=1440, tolerance_pct=0.0)

    state.advance_time()

    sides = [trade.side for trade in state.trade_log]
    assert sides == ["buy", "sell"]
    assert state.positions == {}
