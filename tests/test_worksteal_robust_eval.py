"""Tests for multi-window work-steal robustness evaluation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from binance_worksteal.robust_eval import (
    build_recent_windows,
    summarize_config_robustness,
)
from binance_worksteal.strategy import WorkStealConfig


def make_bars(prices, start="2026-01-01", symbol="BTCUSD"):
    dates = pd.date_range(start, periods=len(prices), freq="D", tz="UTC")
    rows = []
    for date, price in zip(dates, prices):
        rows.append(
            {
                "timestamp": date,
                "open": float(price),
                "high": float(price) * 1.01,
                "low": float(price) * 0.99,
                "close": float(price),
                "volume": 1000.0,
                "symbol": symbol,
            }
        )
    return pd.DataFrame(rows)


def test_build_recent_windows_returns_non_overlapping_recent_windows():
    windows = build_recent_windows(end_date="2026-03-14", window_days=60, window_count=3)

    assert [window.label for window in windows] == [
        "w1_2026-01-13_2026-03-14",
        "w2_2025-11-14_2026-01-13",
        "w3_2025-09-15_2025-11-14",
    ]


def test_summarize_config_robustness_returns_scenarios_and_summary():
    all_bars = {
        "BTCUSD": make_bars([100 + i for i in range(30)], symbol="BTCUSD"),
        "ETHUSD": make_bars([200 + i * 2 for i in range(30)], symbol="ETHUSD"),
        "ALTUSD": make_bars(
            [100] * 10 + [92, 90, 89, 91, 95, 98, 100, 103, 105, 107] + [107] * 10,
            symbol="ALTUSD",
        ),
    }
    config = WorkStealConfig(
        dip_pct=0.10,
        proximity_pct=0.03,
        lookback_days=5,
        profit_target_pct=0.05,
        stop_loss_pct=0.10,
        max_positions=3,
    )
    windows = build_recent_windows(end_date="2026-01-30", window_days=10, window_count=2)

    scenario_rows, summary = summarize_config_robustness(
        all_bars=all_bars,
        config=config,
        windows=windows,
        start_states=("flat", "BTC"),
    )

    assert len(scenario_rows) == 4
    assert summary["scenario_count"] == pytest.approx(4.0)
    assert summary["trade_count_mean"] >= 0.0
    assert summary["max_drawdown_worst_pct"] >= 0.0
    assert summary["negative_return_rate"] >= 0.0
    assert {row["start_state"] for row in scenario_rows} == {"flat", "btcusd"}
    assert {row["window_label"] for row in scenario_rows} == {
        "w1_2026-01-20_2026-01-30",
        "w2_2026-01-10_2026-01-20",
    }
    assert all(float(row["max_drawdown_pct"]) >= 0.0 for row in scenario_rows)
