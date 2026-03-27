from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pufferlib_market.replay_eval as module
from pufferlib_market.hourly_replay import DailySimResult, HourlyMarket, HourlyReplayResult, InitialPositionSpec, MktdData


def _build_test_data() -> tuple[MktdData, HourlyMarket]:
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
    return data, market


def test_replay_eval_main_forwards_fill_buffer_bps_and_caps_steps(monkeypatch, tmp_path: Path) -> None:
    data, market = _build_test_data()
    captured: dict[str, float] = {}

    monkeypatch.setattr(module, "read_mktd", lambda path: data)
    monkeypatch.setattr(module, "load_hourly_market", lambda *args, **kwargs: market)
    monkeypatch.setattr(module, "load_policy", lambda *args, **kwargs: (object(), {}, 3))
    monkeypatch.setattr(module, "make_policy_fn", lambda *args, **kwargs: (lambda obs: 1))
    monkeypatch.setattr(module, "annualize_total_return", lambda total_return, periods, periods_per_year: total_return)

    def _fake_simulate_daily_policy(*args, **kwargs) -> DailySimResult:
        captured["fill_buffer_bps"] = float(kwargs["fill_buffer_bps"])
        captured["max_steps"] = float(kwargs["max_steps"])
        return DailySimResult(
            actions=np.asarray([1, 1], dtype=np.int32),
            total_return=0.10,
            sortino=1.2,
            max_drawdown=0.05,
            num_trades=1,
            win_rate=1.0,
            avg_hold_steps=2.0,
            equity_curve=np.asarray([10_000.0, 10_500.0, 11_000.0], dtype=np.float64),
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
            equity_curve=np.full((len(market.index),), 10_000.0, dtype=np.float64),
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
            "90",
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
    assert captured["max_steps"] == 2.0
    assert payload["fill_buffer_bps"] == 7.5


def test_replay_eval_main_emits_robust_start_state_summary(monkeypatch, tmp_path: Path) -> None:
    data, market = _build_test_data()
    scenarios: list[InitialPositionSpec | None] = []

    monkeypatch.setattr(module, "read_mktd", lambda path: data)
    monkeypatch.setattr(module, "load_hourly_market", lambda *args, **kwargs: market)
    monkeypatch.setattr(module, "load_policy", lambda *args, **kwargs: (object(), {}, 3))
    monkeypatch.setattr(module, "make_policy_fn", lambda *args, **kwargs: (lambda obs: 1))
    monkeypatch.setattr(module, "annualize_total_return", lambda total_return, periods, periods_per_year: total_return)

    def _fake_simulate_daily_policy(*args, **kwargs) -> DailySimResult:
        init_pos = kwargs.get("initial_position")
        if init_pos is not None:
            scenarios.append(init_pos)
        total_return = 0.03 if init_pos is None else -0.02
        sortino = 0.9 if init_pos is None else -0.4
        max_drawdown = 0.05 if init_pos is None else 0.15
        return DailySimResult(
            actions=np.asarray([1, 1], dtype=np.int32),
            total_return=total_return,
            sortino=sortino,
            max_drawdown=max_drawdown,
            num_trades=1,
            win_rate=1.0,
            avg_hold_steps=2.0,
            equity_curve=np.asarray([10_000.0, 9_900.0, 9_800.0], dtype=np.float64),
        )

    def _fake_hourly_result(**kwargs) -> HourlyReplayResult:
        init_pos = kwargs.get("initial_position")
        total_return = 0.02 if init_pos is None else -0.05
        sortino = 0.6 if init_pos is None else -0.8
        max_drawdown = 0.07 if init_pos is None else 0.22
        return HourlyReplayResult(
            total_return=total_return,
            sortino=sortino,
            max_drawdown=max_drawdown,
            num_trades=1,
            num_orders=2,
            win_rate=1.0,
            equity_curve=np.asarray([10_000.0, 9_900.0, 9_800.0], dtype=np.float64),
            orders_by_day={},
        )

    monkeypatch.setattr(module, "simulate_daily_policy", _fake_simulate_daily_policy)
    monkeypatch.setattr(module, "replay_hourly_frozen_daily_actions", _fake_hourly_result)
    monkeypatch.setattr(module, "simulate_hourly_policy", _fake_hourly_result)

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
            "--robust-start-states",
            "flat,long:AAA:0.25",
            "--run-hourly-policy",
            "--cpu",
            "--output-json",
            str(output_json),
        ],
    )

    module.main()

    payload = json.loads(output_json.read_text())
    assert scenarios and scenarios[0].symbol == "AAA"
    assert payload["robust_start_summary"]["hourly_replay"]["worst_total_return"] == -0.05
    assert payload["robust_start_summary"]["hourly_policy"]["worst_sortino"] == -0.8
