from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pufferlib_market.hourly_replay import (
    HourlyMarket,
    MktdData,
    replay_hourly_frozen_daily_actions,
    simulate_daily_policy,
)


def test_hourly_replay_matches_daily_when_hourly_close_is_piecewise_constant():
    # 1 symbol, 3 days, 2-step episode (t=0->1, 1->2)
    symbols = ["AAA"]
    T = 3
    S = 1
    F = 16
    P = 5

    features = np.zeros((T, S, F), dtype=np.float32)
    prices = np.zeros((T, S, P), dtype=np.float32)
    prices[:, 0, 3] = np.array([100.0, 110.0, 121.0], dtype=np.float32)  # close
    prices[:, 0, 0] = prices[:, 0, 3]  # open
    prices[:, 0, 1] = prices[:, 0, 3]  # high
    prices[:, 0, 2] = prices[:, 0, 3]  # low
    tradable = np.ones((T, S), dtype=np.uint8)

    data = MktdData(version=2, symbols=symbols, features=features, prices=prices, tradable=tradable)

    # Policy always chooses "go long symbol 0" (action=1).
    daily = simulate_daily_policy(
        data,
        lambda obs: 1,
        max_steps=2,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=365.0,
    )
    assert daily.total_return == pytest.approx(0.21, abs=1e-9)
    assert daily.num_trades == 1

    start_date = "2024-01-01"
    end_date = "2024-01-03"
    idx = pd.date_range(f"{start_date} 00:00", f"{end_date} 23:00", freq="h", tz="UTC")
    close = np.zeros((len(idx),), dtype=np.float64)

    # Piecewise-constant daily closes for each UTC day.
    close[idx.floor("D") == pd.Timestamp("2024-01-01", tz="UTC")] = 100.0
    close[idx.floor("D") == pd.Timestamp("2024-01-02", tz="UTC")] = 110.0
    close[idx.floor("D") == pd.Timestamp("2024-01-03", tz="UTC")] = 121.0

    market = HourlyMarket(index=idx, close={"AAA": close}, tradable={"AAA": np.ones_like(close, dtype=bool)})

    hourly = replay_hourly_frozen_daily_actions(
        data=data,
        actions=daily.actions,
        market=market,
        start_date=start_date,
        end_date=end_date,
        max_steps=2,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=8760.0,
    )

    assert hourly.total_return == pytest.approx(0.21, abs=1e-9)
    assert hourly.num_trades == 1
    assert hourly.num_orders == 2  # open + terminal close
    assert hourly.max_drawdown == pytest.approx(0.0, abs=1e-12)
    assert hourly.equity_curve.shape == (len(idx),)
