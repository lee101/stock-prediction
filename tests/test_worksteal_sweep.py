"""Tests for binance_worksteal sweep module."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import binance_worksteal.sweep as sweep_module
from binance_worksteal.strategy import WorkStealConfig, run_worksteal_backtest
from binance_worksteal.sweep import (
    SWEEP_GRID, build_windows, eval_config_single_window,
    eval_config_multi_window, run_sweep, summarize_execution_mode,
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

    def test_short_data_truncates_requested_windows(self):
        bars = _make_multi_bars(n_days=100)
        windows = build_windows(bars, window_days=60, n_windows=3)
        assert len(windows) == 1

    def test_windows_non_overlapping(self):
        bars = _make_multi_bars(n_days=200)
        windows = build_windows(bars, window_days=60, n_windows=3)
        for i in range(len(windows) - 1):
            assert windows[i][0] >= windows[i + 1][1] or windows[i][1] <= windows[i + 1][0]


class TestRealisticFill:
    def test_realistic_fill_fewer_entries(self):
        bars = _make_multi_bars(n_days=120)
        cfg_normal = WorkStealConfig(
            dip_pct=0.05, proximity_pct=0.03, realistic_fill=False,
            max_positions=5, lookback_days=10,
        )
        cfg_realistic = WorkStealConfig(
            dip_pct=0.05, proximity_pct=0.03, realistic_fill=True,
            max_positions=5, lookback_days=10,
        )
        _, trades_normal, _ = run_worksteal_backtest(
            {k: v.copy() for k, v in bars.items()}, cfg_normal,
        )
        _, trades_realistic, _ = run_worksteal_backtest(
            {k: v.copy() for k, v in bars.items()}, cfg_realistic,
        )
        assert len(trades_realistic) <= len(trades_normal)


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

    def test_incompatible_config_skips_csim(self, monkeypatch):
        bars = _make_multi_bars(n_days=120)
        cfg = WorkStealConfig(
            dip_pct=0.10,
            proximity_pct=0.03,
            realistic_fill=True,
            daily_checkpoint_only=True,
        )
        latest = max(df["timestamp"].max() for df in bars.values())
        start = str((latest - pd.Timedelta(days=60)).date())
        end = str(latest.date())
        seen = {"python": 0, "csim": 0}

        def fake_csim(*args, **kwargs):
            seen["csim"] += 1
            raise AssertionError("C sim should not be used for incompatible configs")

        def fake_python(*args, **kwargs):
            seen["python"] += 1
            return pd.DataFrame(), [], {"sortino": 1.0, "total_return_pct": 2.0}

        monkeypatch.setattr("binance_worksteal.sweep.run_worksteal_backtest", fake_python)
        metrics = eval_config_single_window(bars, cfg, start, end, use_csim_fn=fake_csim)

        assert metrics["sortino"] == 1.0
        assert seen == {"python": 1, "csim": 0}

    def test_csim_runtime_failure_warns_and_falls_back_to_python(self, monkeypatch):
        bars = _make_multi_bars(n_days=120)
        cfg = WorkStealConfig(dip_pct=0.10, proximity_pct=0.03, sma_filter_period=0)
        latest = max(df["timestamp"].max() for df in bars.values())
        start = str((latest - pd.Timedelta(days=60)).date())
        end = str(latest.date())
        seen = {"python": 0}
        warnings = []

        def fake_csim(*args, **kwargs):
            raise RuntimeError("csim boom")

        def fake_python(*args, **kwargs):
            seen["python"] += 1
            return pd.DataFrame(), [], {"sortino": 1.5, "total_return_pct": 4.0}

        monkeypatch.setattr("binance_worksteal.sweep.run_worksteal_backtest", fake_python)
        monkeypatch.setattr("binance_worksteal.sweep.config_supports_csim", lambda config: True)
        metrics = eval_config_single_window(
            bars,
            cfg,
            start,
            end,
            use_csim_fn=fake_csim,
            warn_csim_failure=warnings.append,
        )

        assert metrics["sortino"] == 1.5
        assert seen["python"] == 1
        assert len(warnings) == 1
        assert "sweep C sim evaluation failed" in warnings[0]
        assert "csim boom" in warnings[0]

    def test_python_path_counts_completed_trades_not_order_rows(self, monkeypatch):
        bars = _make_multi_bars(n_days=120)
        cfg = WorkStealConfig(dip_pct=0.10, proximity_pct=0.03)
        latest = max(df["timestamp"].max() for df in bars.values())
        start = str((latest - pd.Timedelta(days=60)).date())
        end = str(latest.date())

        fake_trades = [
            SimpleNamespace(side="buy"),
            SimpleNamespace(side="sell"),
            SimpleNamespace(side="buy"),
        ]

        def fake_python(*args, **kwargs):
            return pd.DataFrame(), fake_trades, {"sortino": 1.0, "total_return_pct": 2.0}

        monkeypatch.setattr("binance_worksteal.sweep.run_worksteal_backtest", fake_python)
        metrics = eval_config_single_window(bars, cfg, start, end)

        assert metrics["n_trades"] == 1

    def test_python_path_reuses_prepared_bars(self, monkeypatch):
        bars = _make_multi_bars(n_days=120)
        cfg = WorkStealConfig(dip_pct=0.10, proximity_pct=0.03)
        latest = max(df["timestamp"].max() for df in bars.values())
        start = str((latest - pd.Timedelta(days=60)).date())
        end = str(latest.date())
        prepared = sweep_module.prepare_backtest_bars(bars)
        seen = {}

        def fake_python(*args, **kwargs):
            seen["prepared_is_forwarded"] = kwargs["prepared_bars"] is prepared
            return pd.DataFrame(), [], {"sortino": 1.0, "total_return_pct": 2.0, "n_trades": 1}

        monkeypatch.setattr("binance_worksteal.sweep.run_worksteal_backtest", fake_python)
        metrics = eval_config_single_window(bars, cfg, start, end, prepared_bars=prepared)

        assert metrics["sortino"] == 1.0
        assert seen == {"prepared_is_forwarded": True}


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

    def test_sweep_realistic_mode_sets_config_flag(self, tmp_path, monkeypatch):
        bars = _make_multi_bars(n_days=120)
        windows = build_windows(bars, window_days=60, n_windows=1)
        out = str(tmp_path / "test_sweep_realistic_flag.csv")
        seen = []

        def fake_eval_config_multi_window(
            all_bars,
            config,
            windows,
            prepared_bars=None,
            use_csim_fn=None,
            warn_csim_failure=None,
            report_backtest_failure=None,
        ):
            seen.append((bool(config.realistic_fill), bool(config.daily_checkpoint_only)))
            return {
                "mean_sortino": 1.0,
                "min_sortino": 1.0,
                "max_sortino": 1.0,
                "mean_return_pct": 1.0,
                "min_return_pct": 1.0,
                "max_return_pct": 1.0,
                "mean_drawdown_pct": -1.0,
                "worst_drawdown_pct": -1.0,
                "mean_n_trades": 1.0,
                "total_n_trades": 1,
                "mean_win_rate": 50.0,
                "n_windows": len(windows),
            }

        monkeypatch.setattr("binance_worksteal.sweep.eval_config_multi_window", fake_eval_config_multi_window)
        run_sweep(bars, windows, out, max_trials=1, cash=10000.0, realistic=True)

        assert seen == [(True, True)]

    def test_sweep_base_config_preserves_non_swept_fields(self, tmp_path, monkeypatch):
        bars = _make_multi_bars(n_days=120)
        windows = build_windows(bars, window_days=60, n_windows=1)
        out = str(tmp_path / "test_sweep_base_config.csv")
        seen = []

        monkeypatch.setattr(
            "binance_worksteal.sweep.SWEEP_GRID",
            {
                "dip_pct": [0.05],
                "proximity_pct": [0.03],
                "profit_target_pct": [0.05],
                "stop_loss_pct": [0.08],
                "max_positions": [5],
                "max_hold_days": [14],
                "lookback_days": [20],
                "ref_price_method": ["high"],
                "max_leverage": [1.0],
                "enable_shorts": [False],
                "trailing_stop_pct": [0.0],
                "sma_filter_period": [0],
                "market_breadth_filter": [0.0],
            },
        )

        def fake_eval_config_multi_window(
            all_bars,
            config,
            windows,
            prepared_bars=None,
            use_csim_fn=None,
            warn_csim_failure=None,
            report_backtest_failure=None,
        ):
            seen.append((config.dip_pct, config.base_asset_symbol, config.sma_check_method))
            return {
                "mean_sortino": 1.0,
                "min_sortino": 1.0,
                "max_sortino": 1.0,
                "mean_return_pct": 1.0,
                "min_return_pct": 1.0,
                "max_return_pct": 1.0,
                "mean_drawdown_pct": -1.0,
                "worst_drawdown_pct": -1.0,
                "mean_n_trades": 1.0,
                "total_n_trades": 1,
                "mean_win_rate": 50.0,
                "n_windows": len(windows),
            }

        monkeypatch.setattr("binance_worksteal.sweep.eval_config_multi_window", fake_eval_config_multi_window)

        run_sweep(
            bars,
            windows,
            out,
            max_trials=1,
            base_config=WorkStealConfig(
                dip_pct=0.18,
                base_asset_symbol="ETHUSD",
                sma_check_method="current",
            ),
        )

        assert seen == [(0.05, "ETHUSD", "current")]

    def test_sweep_reports_skipped_python_failures(self, tmp_path, monkeypatch, capsys):
        bars = _make_multi_bars(n_days=120)
        windows = build_windows(bars, window_days=60, n_windows=1)
        out = str(tmp_path / "test_sweep_failures.csv")

        monkeypatch.setattr("binance_worksteal.sweep.SWEEP_GRID", {"dip_pct": [0.1]})

        def fake_python(*args, **kwargs):
            raise RuntimeError("python boom")

        monkeypatch.setattr("binance_worksteal.sweep.run_worksteal_backtest", fake_python)

        results = run_sweep(bars, windows, out, max_trials=1, cash=10000.0, use_csim=False)

        output = capsys.readouterr().out
        assert results == []
        assert "WARN: sweep Python backtest evaluation failed" in output
        assert "python boom" in output
        assert "WARN: skipped 1 failed Python window evaluations during sweep" in output

    def test_sweep_return_metadata_reports_failures(self, tmp_path, monkeypatch):
        bars = _make_multi_bars(n_days=120)
        windows = build_windows(bars, window_days=60, n_windows=1)
        out = str(tmp_path / "test_sweep_failures_meta.csv")

        monkeypatch.setattr("binance_worksteal.sweep.SWEEP_GRID", {"dip_pct": [0.1]})

        def fake_python(*args, **kwargs):
            raise RuntimeError("python boom")

        monkeypatch.setattr("binance_worksteal.sweep.run_worksteal_backtest", fake_python)

        results, metadata = run_sweep(
            bars,
            windows,
            out,
            max_trials=1,
            cash=10000.0,
            use_csim=False,
            return_metadata=True,
        )

        assert results == []
        assert metadata["skipped_backtest_failure_count"] == 1
        assert metadata["backtest_failure_samples"]
        assert "python boom" in metadata["backtest_failure_samples"][0]
        assert metadata["suppressed_backtest_failure_count"] == 0


class TestExecutionModeSummary:
    def test_default_mode(self):
        summary = summarize_execution_mode(WorkStealConfig())
        assert summary == {
            "label": "DEFAULT",
            "realistic_fill": False,
            "daily_checkpoint_only": False,
            "realistic": False,
        }

    def test_touch_fill_only_mode(self):
        summary = summarize_execution_mode(WorkStealConfig(realistic_fill=True))
        assert summary == {
            "label": "TOUCH_FILL_ONLY",
            "realistic_fill": True,
            "daily_checkpoint_only": False,
            "realistic": False,
        }

    def test_next_bar_only_mode(self):
        summary = summarize_execution_mode(WorkStealConfig(daily_checkpoint_only=True))
        assert summary == {
            "label": "NEXT_BAR_ONLY",
            "realistic_fill": False,
            "daily_checkpoint_only": True,
            "realistic": False,
        }

    def test_full_realistic_mode(self):
        summary = summarize_execution_mode(
            WorkStealConfig(realistic_fill=True, daily_checkpoint_only=True)
        )
        assert summary == {
            "label": "REALISTIC (touch-fill + next-bar execution)",
            "realistic_fill": True,
            "daily_checkpoint_only": True,
            "realistic": True,
        }
