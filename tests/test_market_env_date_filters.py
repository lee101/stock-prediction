from __future__ import annotations

import pandas as pd

from fastmarketsim.env import FastMarketEnv
from pufferlibtraining3.envs.market_env import MarketEnv, MarketEnvConfig


def _write_utc_csv(path) -> None:
    pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01T05:00:00+00:00",
                "2024-01-02T05:00:00+00:00",
                "2024-01-03T05:00:00+00:00",
            ],
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.1, 11.1, 12.1],
            "volume": [100, 110, 120],
        }
    ).to_csv(path, index=False)


def test_market_env_read_csv_accepts_naive_date_filters_for_utc_index(tmp_path) -> None:
    csv_path = tmp_path / "AAPL.csv"
    _write_utc_csv(csv_path)

    env = object.__new__(MarketEnv)
    env.cfg = MarketEnvConfig(start_date="2024-01-02", end_date="2024-01-03")

    frame = MarketEnv._read_csv(env, csv_path)

    assert frame.index.tz is not None
    assert [ts.isoformat() for ts in frame.index] == ["2024-01-02T05:00:00+00:00"]


def test_fast_market_env_read_csv_accepts_naive_date_filters_for_utc_index(tmp_path) -> None:
    csv_path = tmp_path / "AAPL.csv"
    _write_utc_csv(csv_path)

    env = object.__new__(FastMarketEnv)
    env.cfg = {"start_date": "2024-01-02", "end_date": "2024-01-03"}

    frame = FastMarketEnv._read_csv(env, csv_path)

    assert frame.index.tz is not None
    assert [ts.isoformat() for ts in frame.index] == ["2024-01-02T05:00:00+00:00"]
