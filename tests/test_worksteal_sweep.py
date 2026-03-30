"""Tests for binance_worksteal sweep module."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from binance_worksteal.strategy import WorkStealConfig, run_worksteal_backtest
from binance_worksteal.sweep import (
    SWEEP_GRID, build_windows, eval_config_single_window,
    eval_config_multi_window, run_sweep,
)


def _make_bars(symbol="BTCUSD", n_days=120, start_price=40000.0, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2025-10-01", periods=n_days, freq="D", tz="UTC")
    prices = [start_price]
    for _ in range(n_days - 1):
        prices.append(prices[-1] * (1 + rng.normal(0.0, 0.02)))
    prices = np.array(prices)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices * (1 + rng.normal(0, 0.005, n_days)),
        "high": prices * (1 + np.abs(rng.normal(0.01, 0.01, n_days))),
        "low": prices * (1 - np.abs(rng.normal(0.01, 0.01, n_days))),
        "close": prices,
        "volume": rng.uniform(100, 1000, n_days),
    })
    return {symbol: df}


def _make_multi_bars(n_days=180):
    bars = {}
    for i, sym in enumerate(["BTCUSD", "ETHUSD", "SOLUSD"]):
        b = _make_bars(sym, n_days=n_days, start_price=40000 / (i + 1), seed=42 + i)
        bars.update(b)
    return bars


class TestBuildWindows:
    def test_default_3_windows(self):
        bars = _make_multi_bars(n_days=200)
        windows = build_windows(bars, window_days=60, n_windows=3)
        assert len(windows) == 3
        for start, end in windows:
            assert start < end

    def test_single_window(self):
        bars = _make_multi_bars(n_days=100)
        windows = build_windows(bars, window_days=60, n_windows=1)
        assert len(windows) == 1

    def test_windows_non_overlapping(self):
        bars = _make_multi_bars(n_days=200)
        windows = build_windows(bars, window_days=60, n_windows=3)
        for i in range(len(windows) - 1):
            assert windows[i][0] >= windows[i + 1][1] or windows[i][1] <= windows[i + 1][0]


class TestEntryProximity:
    def test_tighter_proximity_yields_fewer_entries(self):
        bars = _make_multi_bars(n_days=120)
        cfg_loose = WorkStealConfig(
            dip_pct=0.05, proximity_pct=0.03,
            max_positions=5, lookback_days=10,
        )
        cfg_tight = WorkStealConfig(
            dip_pct=0.05, proximity_pct=0.0,
            max_positions=5, lookback_days=10,
        )
        _, trades_loose, _ = run_worksteal_backtest(
            {k: v.copy() for k, v in bars.items()}, cfg_loose,
        )
        _, trades_tight, _ = run_worksteal_backtest(
            {k: v.copy() for k, v in bars.items()}, cfg_tight,
        )
        assert len(trades_tight) <= len(trades_loose)


class TestEvalConfigSingleWindow:
    def test_returns_metrics(self):
        bars = _make_multi_bars(n_days=120)
        cfg = WorkStealConfig(dip_pct=0.10, proximity_pct=0.03)
        latest = max(df["timestamp"].max() for df in bars.values())
        start = str((latest - pd.Timedelta(days=60)).date())
        end = str(latest.date())
        m = eval_config_single_window(bars, cfg, start, end)
        assert m is not None
        assert "sortino" in m or "total_return_pct" in m


class TestEvalConfigMultiWindow:
    def test_returns_multi_window_metrics(self):
        bars = _make_multi_bars(n_days=200)
        cfg = WorkStealConfig(dip_pct=0.10, proximity_pct=0.03)
        windows = build_windows(bars, window_days=60, n_windows=2)
        m = eval_config_multi_window(bars, cfg, windows)
        if m is not None:
            assert "min_sortino" in m
            assert "mean_sortino" in m
            assert "max_sortino" in m
            assert "n_windows" in m
            assert m["n_windows"] == 2


class TestSweepGrid:
    def test_grid_has_required_keys(self):
        required = ["dip_pct", "proximity_pct", "profit_target_pct", "stop_loss_pct",
                     "sma_filter_period"]
        for k in required:
            assert k in SWEEP_GRID

    def test_proximity_pct_values(self):
        assert 0.01 in SWEEP_GRID["proximity_pct"]
        assert 0.02 in SWEEP_GRID["proximity_pct"]
        assert 0.03 in SWEEP_GRID["proximity_pct"]
        assert 0.05 in SWEEP_GRID["proximity_pct"]

    def test_sma_filter_has_disabled(self):
        assert 0 in SWEEP_GRID["sma_filter_period"]


class TestRunSweep:
    def test_sweep_produces_results(self, tmp_path):
        bars = _make_multi_bars(n_days=200)
        windows = build_windows(bars, window_days=60, n_windows=2)
        out = str(tmp_path / "test_sweep.csv")
        results = run_sweep(bars, windows, out, max_trials=5, cash=10000.0)
        assert isinstance(results, list)
        if results:
            df = pd.read_csv(out)
            assert "min_sortino" in df.columns
            assert "mean_sortino" in df.columns

    def test_sweep_realistic_mode(self, tmp_path):
        bars = _make_multi_bars(n_days=200)
        windows = build_windows(bars, window_days=60, n_windows=2)
        out = str(tmp_path / "test_sweep_realistic.csv")
        results = run_sweep(bars, windows, out, max_trials=5, cash=10000.0, realistic=True)
        assert isinstance(results, list)
