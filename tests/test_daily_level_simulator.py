from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.tradinglib.daily_level_simulator import (
    DailyLevelSimulationConfig,
    simulate_daily_levels_on_intraday_bars,
)


def test_daily_level_simulator_multiple_cycles_same_day():
    # 4 hourly bars on a single UTC day.
    ts = [
        datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 2, 1, 1, 0, tzinfo=timezone.utc),
        datetime(2026, 2, 1, 2, 0, tzinfo=timezone.utc),
        datetime(2026, 2, 1, 3, 0, tzinfo=timezone.utc),
    ]
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "low": [9.0, 9.8, 9.4, 9.9],
            "high": [10.0, 10.6, 10.2, 10.7],
            "close": [9.8, 10.4, 9.9, 10.6],
        }
    )
    # Levels apply to the UTC day start.
    levels = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)],
            "buy_price": [9.5],
            "sell_price": [10.5],
        }
    )

    res = simulate_daily_levels_on_intraday_bars(
        bars,
        levels,
        config=DailyLevelSimulationConfig(initial_cash=100.0, maker_fee=0.0, close_at_eod=True),
    )

    # Expected: buy@0h, sell@1h, buy@2h, sell@3h.
    assert [t.side for t in res.trades] == ["buy", "sell", "buy", "sell"]
    assert res.trades[0].price == 9.5
    assert res.trades[1].price == 10.5
    assert res.trades[-1].base_after == 0.0
    assert res.equity_curve.iloc[-1] > 100.0


def test_daily_level_simulator_closes_position_at_eod_boundary():
    ts = [
        datetime(2026, 2, 1, 23, 0, tzinfo=timezone.utc),
        datetime(2026, 2, 2, 0, 0, tzinfo=timezone.utc),
    ]
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "low": [9.0, 9.0],
            "high": [9.1, 9.1],
            "close": [9.05, 9.05],
        }
    )
    levels = pd.DataFrame(
        {
            "timestamp": [
                datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2026, 2, 2, 0, 0, tzinfo=timezone.utc),
            ],
            "buy_price": [9.0, 999.0],  # buy only day1
            "sell_price": [999.0, 999.0],
        }
    )

    res = simulate_daily_levels_on_intraday_bars(
        bars,
        levels,
        config=DailyLevelSimulationConfig(initial_cash=100.0, maker_fee=0.0, close_at_eod=True),
    )

    # Buy on day1 (23:00), then close at EOD at the same bar's close when day flips.
    assert [t.side for t in res.trades][:2] == ["buy", "sell"]
    assert res.trades[1].reason == "eod"
    assert res.trades[1].base_after == 0.0


def test_daily_level_simulator_skips_inverted_levels():
    ts = [
        datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 2, 1, 1, 0, tzinfo=timezone.utc),
    ]
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "low": [1.0, 1.0],
            "high": [2.0, 2.0],
            "close": [1.5, 1.5],
        }
    )
    levels = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)],
            "buy_price": [2.0],
            "sell_price": [1.0],
        }
    )
    res = simulate_daily_levels_on_intraday_bars(
        bars,
        levels,
        config=DailyLevelSimulationConfig(initial_cash=100.0, maker_fee=0.0),
    )
    assert res.trades == []
    assert float(res.equity_curve.iloc[-1]) == 100.0

