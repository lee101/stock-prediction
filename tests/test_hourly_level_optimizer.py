from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.tradinglib.hourly_level_optimizer import (
    HourlyLevelSearchConfig,
    optimize_long_levels_for_window,
    replay_long_levels_for_window,
    walk_forward_hourly_level_search,
)


def _bars(closes: list[float]) -> pd.DataFrame:
    ts0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    prev = closes[0]
    for i, close in enumerate(closes):
        # Every bar dips 1% below previous close and then can take +1% from
        # that entry. This gives the optimiser an obvious grid optimum.
        low = prev * 0.99
        high = low * 1.01
        rows.append(
            {
                "timestamp": ts0 + timedelta(hours=i),
                "open": prev,
                "high": high,
                "low": low,
                "close": close,
            }
        )
        prev = close
    return pd.DataFrame(rows)


def test_optimize_long_levels_finds_profitable_limit_pair():
    frame = _bars([100.0] * 16)
    cfg = HourlyLevelSearchConfig(
        entry_bps_grid=(50.0, 100.0, 150.0),
        take_profit_bps_grid=(50.0, 100.0, 150.0),
        fill_buffer_bps=0.0,
        fee_bps=0.0,
        max_hold_bars=4,
        min_train_trades=1,
    )

    result = optimize_long_levels_for_window(frame, cfg)

    assert result.entry_bps == 100.0
    assert result.take_profit_bps == 100.0
    assert result.train_trades > 0
    assert result.train_return_pct > 0
    assert result.train_win_rate_pct == pytest.approx(100.0)


def test_replay_long_levels_uses_penetration_buffer():
    frame = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "open": [100.0],
            "high": [100.5],
            "low": [99.0],
            "close": [100.0],
        }
    )
    no_buffer_ret, no_buffer_trades, _ = replay_long_levels_for_window(
        frame,
        100.0,
        50.0,
        HourlyLevelSearchConfig(fill_buffer_bps=0.0, fee_bps=0.0),
        prev_close0=100.0,
    )
    buffered_ret, buffered_trades, _ = replay_long_levels_for_window(
        frame,
        100.0,
        50.0,
        HourlyLevelSearchConfig(fill_buffer_bps=5.0, fee_bps=0.0),
        prev_close0=100.0,
    )

    assert no_buffer_trades == 1
    assert no_buffer_ret > 0
    assert buffered_trades == 0
    assert buffered_ret == 0.0


def test_walk_forward_reoptimizes_without_forward_lookahead():
    frame = _bars([100.0] * 32)
    cfg = HourlyLevelSearchConfig(
        lookback_bars=8,
        forward_bars=8,
        entry_bps_grid=(50.0, 100.0),
        take_profit_bps_grid=(50.0, 100.0),
        fill_buffer_bps=0.0,
        fee_bps=0.0,
        max_hold_bars=4,
    )

    result = walk_forward_hourly_level_search(frame, symbol="aapl", config=cfg)

    assert result.symbol == "AAPL"
    assert len(result.windows) == 3
    assert all(window.entry_bps == 100.0 for window in result.windows)
    assert result.total_return_pct > 0
