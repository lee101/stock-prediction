"""Tests for expanded work-stealing sweep."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from binance_worksteal.sweep_expanded import (
    SWEEP_GRID,
    generate_grid,
    combo_to_config,
    compute_safety_score,
    build_windows,
    eval_config_multi_window_python,
    _aggregate_window_metrics,
    run_sweep,
)
from binance_worksteal.strategy import WorkStealConfig


def _make_bars(symbols, n_days=120, base_price=100.0, seed=42):
    rng = np.random.RandomState(seed)
    all_bars = {}
    for si, sym in enumerate(symbols):
        price = base_price * (1 + si * 0.5)
        dates = pd.date_range("2025-09-01", periods=n_days, freq="D", tz="UTC")
        prices = [price]
        for _ in range(n_days - 1):
            ret = rng.normal(0.001, 0.03)
            prices.append(prices[-1] * (1 + ret))
        prices = np.array(prices)
        highs = prices * (1 + rng.uniform(0.005, 0.03, n_days))
        lows = prices * (1 - rng.uniform(0.005, 0.03, n_days))
        opens = prices * (1 + rng.uniform(-0.01, 0.01, n_days))
        all_bars[sym] = pd.DataFrame({
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": rng.uniform(1e6, 1e8, n_days),
        })
    return all_bars


class TestGridGeneration:
    def test_full_grid_size(self):
        keys, combos = generate_grid()
        expected = 1
        for v in SWEEP_GRID.values():
            expected *= len(v)
        assert len(combos) == expected
        assert expected == 2160

    def test_grid_keys_match(self):
        keys, combos = generate_grid()
        assert set(keys) == set(SWEEP_GRID.keys())

    def test_max_trials_limits(self):
        keys, combos = generate_grid(max_trials=50)
        assert len(combos) == 50

    def test_max_trials_larger_than_grid(self):
        keys, combos = generate_grid(max_trials=99999)
        assert len(combos) == 2160

    def test_seed_reproducibility(self):
        _, c1 = generate_grid(max_trials=20, seed=42)
        _, c2 = generate_grid(max_trials=20, seed=42)
        assert c1 == c2

    def test_different_seeds_differ(self):
        _, c1 = generate_grid(max_trials=20, seed=42)
        _, c2 = generate_grid(max_trials=20, seed=99)
        assert c1 != c2


class TestComboToConfig:
    def test_creates_valid_config(self):
        keys, combos = generate_grid(max_trials=1)
        config = combo_to_config(keys, combos[0])
        assert isinstance(config, WorkStealConfig)
        assert config.sma_filter_period == 20
        assert config.lookback_days == 20
        assert config.initial_cash == 10000.0

    def test_params_applied(self):
        keys = list(SWEEP_GRID.keys())
        combo = (0.12, 3.0, 10, 0.05, 0.15, 0.10)
        config = combo_to_config(keys, combo, cash=5000.0)
        assert config.dip_pct == 0.12
        assert config.max_leverage == 3.0
        assert config.max_positions == 10
        assert config.trailing_stop_pct == 0.05
        assert config.profit_target_pct == 0.15
        assert config.stop_loss_pct == 0.10
        assert config.initial_cash == 5000.0


class TestSafetyScore:
    def test_positive_sortino_negative_dd(self):
        score = compute_safety_score(5.0, -10.0)
        assert score == pytest.approx(0.5)

    def test_zero_drawdown_uses_floor(self):
        score = compute_safety_score(2.0, 0.0)
        assert score == pytest.approx(2.0 / 0.01)

    def test_negative_sortino(self):
        score = compute_safety_score(-3.0, -5.0)
        assert score == pytest.approx(-3.0 / 5.0)

    def test_tiny_drawdown(self):
        score = compute_safety_score(1.0, -0.001)
        assert score == pytest.approx(1.0 / 0.01)


class TestBuildWindows:
    def test_window_count(self):
        bars = _make_bars(["BTCUSD"], n_days=200)
        windows = build_windows(bars, window_days=60, n_windows=3)
        assert len(windows) == 3

    def test_window_dates_are_strings(self):
        bars = _make_bars(["BTCUSD"], n_days=200)
        windows = build_windows(bars, window_days=60, n_windows=2)
        for start, end in windows:
            assert isinstance(start, str)
            assert isinstance(end, str)

    def test_windows_non_overlapping(self):
        bars = _make_bars(["BTCUSD"], n_days=250)
        windows = build_windows(bars, window_days=60, n_windows=3)
        for i in range(len(windows) - 1):
            assert windows[i][0] >= windows[i + 1][1]


class TestAggregateWindowMetrics:
    def test_basic_aggregation(self):
        wms = [
            {"sortino": 3.0, "total_return_pct": 5.0, "max_drawdown_pct": -2.0, "n_trades": 10, "win_rate": 70.0, "avg_hold_days": 5.0},
            {"sortino": 1.0, "total_return_pct": 2.0, "max_drawdown_pct": -4.0, "n_trades": 8, "win_rate": 60.0, "avg_hold_days": 6.0},
        ]
        r = _aggregate_window_metrics(wms)
        assert r["mean_sortino"] == pytest.approx(2.0)
        assert r["min_sortino"] == pytest.approx(1.0)
        assert r["mean_return_pct"] == pytest.approx(3.5)
        assert r["max_drawdown_pct"] == pytest.approx(-4.0)
        assert r["total_n_trades"] == 18
        assert r["n_windows"] == 2
        assert "safety_score" in r

    def test_safety_score_in_aggregation(self):
        wms = [
            {"sortino": 4.0, "total_return_pct": 8.0, "max_drawdown_pct": -2.0, "n_trades": 5, "win_rate": 80.0, "avg_hold_days": 3.0},
        ]
        r = _aggregate_window_metrics(wms)
        expected = compute_safety_score(4.0, -2.0)
        assert r["safety_score"] == pytest.approx(expected)


class TestEvalMultiWindow:
    def test_returns_metrics_or_none(self):
        bars = _make_bars(["BTCUSD", "ETHUSD", "SOLUSD"], n_days=200)
        config = WorkStealConfig(
            dip_pct=0.10, profit_target_pct=0.15, stop_loss_pct=0.10,
            max_positions=5, trailing_stop_pct=0.03, sma_filter_period=20,
            lookback_days=20, max_hold_days=14, initial_cash=10000.0,
        )
        windows = build_windows(bars, window_days=60, n_windows=2)
        result = eval_config_multi_window_python(bars, config, windows)
        # May return None if no trades, or a dict with metrics
        if result is not None:
            assert "mean_sortino" in result
            assert "safety_score" in result
            assert "max_drawdown_pct" in result


class TestCSVOutput:
    def test_csv_has_required_columns(self):
        bars = _make_bars(["BTCUSD", "ETHUSD"], n_days=150)
        windows = build_windows(bars, window_days=60, n_windows=2)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            out_path = f.name
        try:
            results = run_sweep(bars, windows, out_path, max_trials=5, cash=10000.0)
            if results:
                df = pd.read_csv(out_path)
                for col in ["dip_pct", "max_leverage", "max_positions",
                            "trailing_stop_pct", "profit_target_pct", "stop_loss_pct",
                            "mean_sortino", "mean_return_pct", "max_drawdown_pct",
                            "safety_score", "total_n_trades", "mean_win_rate"]:
                    assert col in df.columns, f"Missing column: {col}"
                assert df["safety_score"].is_monotonic_decreasing or len(df) <= 1
        finally:
            os.unlink(out_path)

    def test_csv_sorted_by_safety(self):
        bars = _make_bars(["BTCUSD", "ETHUSD", "SOLUSD"], n_days=200)
        windows = build_windows(bars, window_days=60, n_windows=2)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            out_path = f.name
        try:
            results = run_sweep(bars, windows, out_path, max_trials=10, cash=10000.0)
            if len(results) > 1:
                df = pd.read_csv(out_path)
                safety = df["safety_score"].values
                assert all(safety[i] >= safety[i + 1] for i in range(len(safety) - 1))
        finally:
            os.unlink(out_path)
