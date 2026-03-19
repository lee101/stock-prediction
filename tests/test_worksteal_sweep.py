"""Tests for robust multi-start work-steal sweeps."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from binance_worksteal.strategy import WorkStealConfig
from binance_worksteal.sweep import _build_start_state_config, run_sweep


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


def test_build_start_state_config_seeds_equal_equity():
    all_bars = {
        "BTCUSD": make_bars([100, 102, 104, 106], symbol="BTCUSD"),
    }
    label, seeded = _build_start_state_config(
        base_config=WorkStealConfig(initial_cash=10_000.0),
        all_bars=all_bars,
        start_state="BTCUSDT",
        start_date="2026-01-01",
        end_date="2026-01-04",
        starting_equity=10_000.0,
    )

    assert label == "btcusd"
    assert seeded.initial_cash == pytest.approx(0.0)
    assert seeded.initial_holdings == {"BTCUSD": pytest.approx(100.0)}


def test_run_sweep_multi_start_outputs_robust_columns(tmp_path):
    all_bars = {
        "BTCUSD": make_bars([100, 101, 102, 103, 104, 105, 106, 107], symbol="BTCUSD"),
        "ETHUSD": make_bars([200, 202, 204, 206, 208, 210, 212, 214], symbol="ETHUSD"),
        "ALTUSD": make_bars([100, 100, 102, 101, 99, 98, 100, 103], symbol="ALTUSD"),
    }
    output_path = tmp_path / "robust_sweep.csv"
    sweep_grid = {
        "dip_pct": [0.10],
        "proximity_pct": [0.02],
        "profit_target_pct": [0.05],
        "stop_loss_pct": [0.10],
        "max_positions": [3],
        "max_hold_days": [5],
        "lookback_days": [3],
        "ref_price_method": ["high"],
        "max_leverage": [1.0],
        "enable_shorts": [False],
        "trailing_stop_pct": [0.0],
        "sma_filter_period": [0],
        "market_breadth_filter": [0.0],
    }

    run_sweep(
        all_bars=all_bars,
        start_date="2026-01-02",
        end_date="2026-01-08",
        output_csv=output_path,
        max_trials=1,
        cash=10_000.0,
        entry_proximity_bps=float("inf"),
        start_states=("flat", "BTC", "ETH"),
        sweep_grid=sweep_grid,
    )

    df = pd.read_csv(output_path)
    assert len(df) == 1
    row = df.iloc[0]

    assert row["scenario_count"] == pytest.approx(3)
    assert row["start_states"] == "flat,btcusd,ethusd"
    assert "flat_total_return_pct" in df.columns
    assert "btcusd_total_return_pct" in df.columns
    assert "ethusd_total_return_pct" in df.columns
    assert "mean_total_return_pct" in df.columns
    assert "worst_total_return_pct" in df.columns
    assert "mean_sortino" in df.columns
    assert "worst_sortino" in df.columns
    assert "worst_max_drawdown_pct" in df.columns
    assert row["worst_total_return_pct"] == pytest.approx(
        min(
            row["flat_total_return_pct"],
            row["btcusd_total_return_pct"],
            row["ethusd_total_return_pct"],
        )
    )
