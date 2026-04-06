"""Tests for expanded work-stealing sweep."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

import binance_worksteal.sweep_expanded as sweep_expanded_module
from binance_worksteal.sweep_expanded import (
    SWEEP_GRID,
    generate_grid,
    combo_to_config,
    compute_safety_score,
    build_windows,
    eval_config_single_window_python,
    eval_config_multi_window_python,
    _aggregate_window_metrics,
    run_sweep,
)
from binance_worksteal.strategy import TradeLog, WorkStealConfig


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

    def test_zero_max_trials_preserves_full_grid(self):
        keys, combos = generate_grid(max_trials=0)
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

    def test_base_config_preserves_non_swept_fields_and_grid_overrides(self):
        keys = list(SWEEP_GRID.keys())
        combo = (0.12, 3.0, 10, 0.05, 0.15, 0.10)
        config = combo_to_config(
            keys,
            combo,
            base_config=WorkStealConfig(
                dip_pct=0.25,
                base_asset_symbol="ETHUSD",
                sma_check_method="current",
                max_hold_days=30,
            ),
        )

        assert config.dip_pct == 0.12
        assert config.base_asset_symbol == "ETHUSD"
        assert config.sma_check_method == "current"
        assert config.max_hold_days == 30


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

    def test_short_data_truncates_requested_windows(self):
        bars = _make_bars(["BTCUSD"], n_days=100)
        windows = build_windows(bars, window_days=60, n_windows=3)
        assert len(windows) == 1

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
        if result is not None:
            assert "mean_sortino" in result
            assert "safety_score" in result
            assert "max_drawdown_pct" in result

    def test_single_window_python_uses_shared_trade_metrics_helpers(self, monkeypatch):
        bars = _make_bars(["BTCUSD"], n_days=120)
        config = WorkStealConfig(dip_pct=0.10, profit_target_pct=0.15, stop_loss_pct=0.10)

        trades = [
            TradeLog(
                timestamp=pd.Timestamp("2025-11-01 00:00:00", tz="UTC"),
                symbol="BTCUSD",
                side="buy",
                price=100.0,
                quantity=1.0,
                notional=100.0,
                fee=0.1,
            ),
            TradeLog(
                timestamp=pd.Timestamp("2025-11-01 12:00:00", tz="UTC"),
                symbol="BTCUSD",
                side="sell",
                price=105.0,
                quantity=1.0,
                notional=105.0,
                fee=0.1,
            ),
        ]

        def fake_backtest(*args, **kwargs):
            return pd.DataFrame(), trades, {"sortino": 1.0, "total_return_pct": 2.0, "max_drawdown_pct": -1.0}

        monkeypatch.setattr("binance_worksteal.sweep_expanded.run_worksteal_backtest", fake_backtest)

        result = eval_config_single_window_python(
            bars,
            config,
            "2025-11-01",
            "2025-12-01",
        )

        assert result["n_trades"] == 1
        assert result["avg_hold_days"] == pytest.approx(1.0)

    def test_single_window_python_reuses_prepared_bars(self, monkeypatch):
        bars = _make_bars(["BTCUSD"], n_days=120)
        config = WorkStealConfig(dip_pct=0.10, profit_target_pct=0.15, stop_loss_pct=0.10)
        prepared = sweep_expanded_module.prepare_backtest_bars(bars)
        seen = {}

        def fake_backtest(*args, **kwargs):
            seen["prepared_is_forwarded"] = kwargs["prepared_bars"] is prepared
            return pd.DataFrame(), [], {"sortino": 1.0, "total_return_pct": 2.0, "max_drawdown_pct": -1.0}

        monkeypatch.setattr("binance_worksteal.sweep_expanded.run_worksteal_backtest", fake_backtest)

        result = eval_config_single_window_python(
            bars,
            config,
            "2025-11-01",
            "2025-12-01",
            prepared_bars=prepared,
        )

        assert result["n_trades"] == 0
        assert seen == {"prepared_is_forwarded": True}


class TestCSVOutput:
    def test_csv_has_required_columns(self, tmp_path):
        bars = _make_bars(["BTCUSD", "ETHUSD"], n_days=150)
        windows = build_windows(bars, window_days=60, n_windows=2)
        out_path = tmp_path / "expanded_columns.csv"
        results = run_sweep(bars, windows, str(out_path), max_trials=5, cash=10000.0)
        if results:
            df = pd.read_csv(out_path)
            for col in ["dip_pct", "max_leverage", "max_positions",
                        "trailing_stop_pct", "profit_target_pct", "stop_loss_pct",
                        "mean_sortino", "mean_return_pct", "max_drawdown_pct",
                        "safety_score", "total_n_trades", "mean_win_rate"]:
                assert col in df.columns, f"Missing column: {col}"
            assert df["safety_score"].is_monotonic_decreasing or len(df) <= 1

    def test_csv_sorted_by_safety(self, tmp_path):
        bars = _make_bars(["BTCUSD", "ETHUSD", "SOLUSD"], n_days=200)
        windows = build_windows(bars, window_days=60, n_windows=2)
        out_path = tmp_path / "expanded_sorted.csv"
        results = run_sweep(bars, windows, str(out_path), max_trials=10, cash=10000.0)
        if len(results) > 1:
            df = pd.read_csv(out_path)
            safety = df["safety_score"].values
            assert all(safety[i] >= safety[i + 1] for i in range(len(safety) - 1))


class TestDispatch:
    def test_run_sweep_forwards_prepared_bars_to_python_eval(self, tmp_path, monkeypatch):
        bars = _make_bars(["BTCUSD"], n_days=120)
        windows = [("2025-11-01", "2025-12-01")]
        output_csv = str(tmp_path / "expanded_prepared.csv")
        keys = list(SWEEP_GRID.keys())
        combo = tuple(values[0] for values in SWEEP_GRID.values())

        monkeypatch.setattr(
            "binance_worksteal.sweep_expanded.generate_grid",
            lambda max_trials=None: (keys, [combo]),
        )
        monkeypatch.setattr("binance_worksteal.sweep_expanded._try_load_csim_batch", lambda: None)
        seen = {}

        def fake_eval(all_bars, config, windows, prepared_bars=None, report_backtest_failure=None):
            seen["prepared_matches_bars"] = (
                prepared_bars is not None
                and set(prepared_bars) == set(all_bars)
                and all(prepared_bars[sym].bars is bars[sym] for sym in all_bars)
            )
            return {
                "mean_sortino": 1.0,
                "min_sortino": 1.0,
                "mean_return_pct": 1.0,
                "max_drawdown_pct": -1.0,
                "mean_win_rate": 50.0,
                "total_n_trades": 2,
                "mean_n_trades": 2.0,
                "avg_hold_days": 3.0,
                "safety_score": 1.0,
                "n_windows": 1,
                "w0_sortino": 1.0,
                "w0_return_pct": 1.0,
                "w0_drawdown_pct": -1.0,
                "w0_n_trades": 2,
            }

        monkeypatch.setattr("binance_worksteal.sweep_expanded.eval_config_multi_window_python", fake_eval)

        results = run_sweep(bars, windows, output_csv, max_trials=1, cash=10000.0, n_workers=1)

        assert len(results) == 1
        assert seen == {"prepared_matches_bars": True}

    def test_run_sweep_disables_csim_for_incompatible_configs(self, tmp_path, monkeypatch):
        bars = _make_bars(["BTCUSD"], n_days=120)
        windows = [("2025-11-01", "2025-12-01")]
        output_csv = str(tmp_path / "expanded.csv")
        keys = list(SWEEP_GRID.keys())
        combo = tuple(values[0] for values in SWEEP_GRID.values())

        def fake_batch(*args, **kwargs):
            raise AssertionError("C batch path should not be used for incompatible configs")

        def fake_eval(*args, **kwargs):
            return {
                "mean_sortino": 1.0,
                "min_sortino": 1.0,
                "mean_return_pct": 1.0,
                "max_drawdown_pct": -1.0,
                "mean_win_rate": 50.0,
                "total_n_trades": 2,
                "mean_n_trades": 2.0,
                "avg_hold_days": 3.0,
                "safety_score": 1.0,
                "n_windows": 1,
                "w0_sortino": 1.0,
                "w0_return_pct": 1.0,
                "w0_drawdown_pct": -1.0,
                "w0_n_trades": 2,
            }

        monkeypatch.setattr("binance_worksteal.sweep_expanded._try_load_csim_batch", lambda: fake_batch)
        monkeypatch.setattr(
            "binance_worksteal.sweep_expanded.generate_grid",
            lambda max_trials=None: (keys, [combo]),
        )
        monkeypatch.setattr("binance_worksteal.sweep_expanded.eval_config_multi_window_python", fake_eval)

        results = run_sweep(bars, windows, output_csv, max_trials=1, cash=10000.0, n_workers=1)

        assert len(results) == 1
        assert pd.read_csv(output_csv).shape[0] == 1

    def test_run_sweep_falls_back_to_python_when_c_batch_raises(self, tmp_path, monkeypatch, capsys):
        bars = _make_bars(["BTCUSD"], n_days=120)
        windows = [("2025-11-01", "2025-12-01")]
        output_csv = str(tmp_path / "expanded_fallback.csv")
        keys = list(SWEEP_GRID.keys())
        combo = tuple(values[0] for values in SWEEP_GRID.values())
        seen = {"python": 0}

        def fake_batch(*args, **kwargs):
            raise RuntimeError("batch boom")

        def fake_eval(*args, **kwargs):
            seen["python"] += 1
            return {
                "mean_sortino": 1.0,
                "min_sortino": 1.0,
                "mean_return_pct": 1.0,
                "max_drawdown_pct": -1.0,
                "mean_win_rate": 50.0,
                "total_n_trades": 2,
                "mean_n_trades": 2.0,
                "avg_hold_days": 3.0,
                "safety_score": 1.0,
                "n_windows": 1,
                "w0_sortino": 1.0,
                "w0_return_pct": 1.0,
                "w0_drawdown_pct": -1.0,
                "w0_n_trades": 2,
            }

        monkeypatch.setattr("binance_worksteal.sweep_expanded._try_load_csim_batch", lambda: fake_batch)
        monkeypatch.setattr(
            "binance_worksteal.sweep_expanded.generate_grid",
            lambda max_trials=None: (keys, [combo]),
        )
        monkeypatch.setattr(
            "binance_worksteal.sweep_expanded.assert_csim_compatible_configs",
            lambda configs, context: None,
        )
        monkeypatch.setattr("binance_worksteal.sweep_expanded.eval_config_multi_window_python", fake_eval)

        results = run_sweep(bars, windows, output_csv, max_trials=1, cash=10000.0)

        output = capsys.readouterr().out
        assert len(results) == 1
        assert seen["python"] == 1
        assert "WARN: sweep_expanded C batch evaluation failed" in output
        assert "falling back to Python for remaining configs" in output

    def test_run_sweep_reports_python_failures(self, tmp_path, monkeypatch, capsys):
        bars = _make_bars(["BTCUSD"], n_days=120)
        windows = [("2025-11-01", "2025-12-01")]
        output_csv = str(tmp_path / "expanded_failures.csv")

        monkeypatch.setattr("binance_worksteal.sweep_expanded.generate_grid", lambda max_trials=None: (["dip_pct"], [(0.1,)]))
        monkeypatch.setattr("binance_worksteal.sweep_expanded._try_load_csim_batch", lambda: None)

        def fake_python(*args, **kwargs):
            raise RuntimeError("python boom")

        monkeypatch.setattr("binance_worksteal.sweep_expanded.run_worksteal_backtest", fake_python)

        results = run_sweep(bars, windows, output_csv, max_trials=1, cash=10000.0, n_workers=1)

        output = capsys.readouterr().out
        assert results == []
        assert "WARN: sweep_expanded Python backtest evaluation failed" in output
        assert "python boom" in output
        assert "WARN: skipped 1 failed Python window evaluations during expanded sweep" in output

    def test_run_sweep_return_metadata_reports_failures(self, tmp_path, monkeypatch):
        bars = _make_bars(["BTCUSD"], n_days=120)
        windows = [("2025-11-01", "2025-12-01")]
        output_csv = str(tmp_path / "expanded_failures_meta.csv")

        monkeypatch.setattr("binance_worksteal.sweep_expanded.generate_grid", lambda max_trials=None: (["dip_pct"], [(0.1,)]))
        monkeypatch.setattr("binance_worksteal.sweep_expanded._try_load_csim_batch", lambda: None)

        def fake_python(*args, **kwargs):
            raise RuntimeError("python boom")

        monkeypatch.setattr("binance_worksteal.sweep_expanded.run_worksteal_backtest", fake_python)

        results, metadata = run_sweep(
            bars,
            windows,
            output_csv,
            max_trials=1,
            cash=10000.0,
            n_workers=1,
            return_metadata=True,
        )

        assert results == []
        assert metadata["skipped_backtest_failure_count"] == 1
        assert metadata["backtest_failure_samples"]
        assert "python boom" in metadata["backtest_failure_samples"][0]
        assert metadata["suppressed_backtest_failure_count"] == 0
