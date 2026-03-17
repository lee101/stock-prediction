from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pufferlib_market.replay_eval as module
from pufferlib_market.hourly_replay import DailySimResult, HourlyMarket, HourlyReplayResult, MktdData


def test_replay_eval_main_forwards_fill_buffer_bps(monkeypatch, tmp_path: Path) -> None:
    features = np.zeros((3, 1, 16), dtype=np.float32)
    prices = np.zeros((3, 1, 5), dtype=np.float32)
    prices[:, 0, :] = np.asarray(
        [
            [100.0, 100.0, 100.0, 100.0, 0.0],
            [110.0, 110.0, 110.0, 110.0, 0.0],
            [121.0, 121.0, 121.0, 121.0, 0.0],
        ],
        dtype=np.float32,
    )
    data = MktdData(
        version=2,
        symbols=["AAA"],
        features=features,
        prices=prices,
        tradable=np.ones((3, 1), dtype=np.uint8),
    )
    market_index = pd.date_range("2026-01-01T00:00:00Z", "2026-01-03T23:00:00Z", freq="h", tz="UTC")
    market = HourlyMarket(
        index=market_index,
        close={"AAA": np.full((len(market_index),), 100.0, dtype=np.float64)},
        tradable={"AAA": np.ones((len(market_index),), dtype=bool)},
    )

    captured: dict[str, float] = {}

    monkeypatch.setattr(module, "read_mktd", lambda path: data)
    monkeypatch.setattr(module, "load_hourly_market", lambda *args, **kwargs: market)
    monkeypatch.setattr(module, "load_policy", lambda *args, **kwargs: (object(), {}, 3))
    monkeypatch.setattr(module, "make_policy_fn", lambda *args, **kwargs: (lambda obs: 1))
    monkeypatch.setattr(module, "annualize_total_return", lambda total_return, periods, periods_per_year: total_return)

    def _fake_simulate_daily_policy(*args, **kwargs) -> DailySimResult:
        captured["fill_buffer_bps"] = float(kwargs["fill_buffer_bps"])
        return DailySimResult(
            actions=np.asarray([1, 1], dtype=np.int32),
            total_return=0.10,
            sortino=1.2,
            max_drawdown=0.05,
            num_trades=1,
            win_rate=1.0,
            avg_hold_steps=2.0,
        )

    monkeypatch.setattr(module, "simulate_daily_policy", _fake_simulate_daily_policy)
    monkeypatch.setattr(
        module,
        "replay_hourly_frozen_daily_actions",
        lambda **kwargs: HourlyReplayResult(
            total_return=0.08,
            sortino=0.9,
            max_drawdown=0.07,
            num_trades=1,
            num_orders=2,
            win_rate=1.0,
            equity_curve=np.full((len(market_index),), 10_000.0, dtype=np.float64),
            orders_by_day={},
        ),
    )

    output_json = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_eval.py",
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--daily-data-path",
            str(tmp_path / "daily.bin"),
            "--hourly-data-root",
            str(tmp_path / "hourly"),
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-01-03",
            "--max-steps",
            "2",
            "--fill-buffer-bps",
            "7.5",
            "--cpu",
            "--output-json",
            str(output_json),
        ],
    )

    module.main()

    payload = json.loads(output_json.read_text())
    assert captured["fill_buffer_bps"] == 7.5
    assert payload["fill_buffer_bps"] == 7.5
