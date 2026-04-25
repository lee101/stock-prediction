from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from xgbnew.backtest import BacktestConfig, simulate


def _drawdown_panel() -> pd.DataFrame:
    rows = []
    start = date(2025, 1, 2)
    closes = [80.0, 120.0, 130.0]
    for i, close in enumerate(closes):
        rows.append({
            "date": start + timedelta(days=i),
            "symbol": "AAA",
            "actual_open": 100.0,
            "actual_close": close,
            "spread_bps": 1.0,
            "dolvol_20d_log": np.log1p(1e9),
        })
    return pd.DataFrame(rows)


def _intraday_drawdown_panel() -> pd.DataFrame:
    rows = []
    start = date(2025, 1, 2)
    lows = [70.0, 100.0, 100.0]
    for i, low in enumerate(lows):
        rows.append({
            "date": start + timedelta(days=i),
            "symbol": "AAA",
            "actual_open": 100.0,
            "actual_high": 105.0,
            "actual_low": low,
            "actual_close": 100.0,
            "spread_bps": 1.0,
            "dolvol_20d_log": np.log1p(1e9),
        })
    return pd.DataFrame(rows)


def test_simulate_stops_after_drawdown_budget_breach():
    df = _drawdown_panel()
    scores = pd.Series([1.0] * len(df), index=df.index)
    cfg = BacktestConfig(
        top_n=1,
        leverage=1.0,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        commission_bps=0.0,
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
        stop_on_drawdown_pct=10.0,
    )

    result = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]

    assert len(result.day_results) == 1
    assert result.day_results[0].daily_return_pct == pytest.approx(-20.0)
    assert result.max_drawdown_pct == pytest.approx(20.0)
    assert result.stopped_early is True
    assert result.stop_reason == "drawdown_pct>=10"


def test_simulate_stops_after_intraday_drawdown_budget_breach():
    df = _intraday_drawdown_panel()
    scores = pd.Series([1.0] * len(df), index=df.index)
    cfg = BacktestConfig(
        top_n=1,
        leverage=1.0,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        commission_bps=0.0,
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
        stop_on_intraday_drawdown_pct=20.0,
    )

    result = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]

    assert len(result.day_results) == 1
    assert result.max_drawdown_pct == pytest.approx(0.0)
    assert result.worst_intraday_dd_pct == pytest.approx(30.0)
    assert result.stopped_early is True
    assert result.stop_reason == "intraday_drawdown_pct>=20"


def test_simulate_default_keeps_full_window_after_drawdown():
    df = _drawdown_panel()
    scores = pd.Series([1.0] * len(df), index=df.index)
    cfg = BacktestConfig(
        top_n=1,
        leverage=1.0,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        commission_bps=0.0,
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
    )

    result = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]

    assert len(result.day_results) == 3
    assert result.stopped_early is False
    assert result.stop_reason == ""


def test_simulate_default_keeps_full_window_after_intraday_drawdown():
    df = _intraday_drawdown_panel()
    scores = pd.Series([1.0] * len(df), index=df.index)
    cfg = BacktestConfig(
        top_n=1,
        leverage=1.0,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        commission_bps=0.0,
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
    )

    result = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]

    assert len(result.day_results) == 3
    assert result.worst_intraday_dd_pct == pytest.approx(30.0)
    assert result.stopped_early is False
    assert result.stop_reason == ""
