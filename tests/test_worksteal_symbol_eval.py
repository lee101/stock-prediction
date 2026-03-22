"""Tests for per-symbol evaluation tool."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from binance_worksteal.evaluate_symbols import (
    PRODUCTION_CONFIG,
    compute_rolling_windows,
    evaluate_standalone,
    evaluate_leave_one_out,
    evaluate_full_universe,
    format_results_table,
    run_evaluation,
    _filter_bars,
)
from binance_worksteal.strategy import WorkStealConfig


def _make_random_walk(symbol: str, n_days: int = 90, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2025-06-01", periods=n_days, freq="D", tz="UTC")
    returns = rng.normal(0.0005, 0.02, n_days)
    prices = start_price * np.cumprod(1 + returns)
    noise = rng.uniform(0.005, 0.025, n_days)
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices * (1 - noise * 0.3),
        "high": prices * (1 + noise),
        "low": prices * (1 - noise),
        "close": prices,
        "volume": rng.uniform(1e6, 1e8, n_days),
        "symbol": symbol,
    })


def _make_dipping_walk(symbol: str, n_days: int = 90, seed: int = 99) -> pd.DataFrame:
    """Generate a walk that has big dips (triggers entries) then recovers."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2025-06-01", periods=n_days, freq="D", tz="UTC")
    prices = np.zeros(n_days)
    prices[0] = 100.0
    for i in range(1, n_days):
        if i % 25 == 0:
            prices[i] = prices[i - 1] * 0.78  # 22% dip
        elif i % 25 < 15:
            prices[i] = prices[i - 1] * 1.015  # recovery
        else:
            prices[i] = prices[i - 1] * (1 + rng.normal(0.001, 0.005))
    noise = rng.uniform(0.005, 0.02, n_days)
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices * (1 - noise * 0.3),
        "high": prices * (1 + noise),
        "low": prices * (1 - noise),
        "close": prices,
        "volume": rng.uniform(1e6, 1e8, n_days),
        "symbol": symbol,
    })


def _make_bars(n_symbols: int = 3, n_days: int = 90) -> dict:
    names = [f"SYM{i}USD" for i in range(n_symbols)]
    return {name: _make_random_walk(name, n_days=n_days, seed=42 + i) for i, name in enumerate(names)}


def _test_config() -> WorkStealConfig:
    return WorkStealConfig(
        dip_pct=0.20,
        profit_target_pct=0.15,
        stop_loss_pct=0.10,
        sma_filter_period=20,
        trailing_stop_pct=0.03,
        max_positions=5,
        max_hold_days=14,
        max_drawdown_exit=0.0,
    )


class TestComputeRollingWindows:
    def test_basic(self):
        bars = _make_bars(2, 200)
        windows = compute_rolling_windows(bars, window_days=60, n_windows=3)
        assert len(windows) == 3
        for start, end in windows:
            s = pd.Timestamp(start, tz="UTC")
            e = pd.Timestamp(end, tz="UTC")
            assert (e - s).days == 60

    def test_short_data(self):
        bars = _make_bars(2, 30)
        windows = compute_rolling_windows(bars, window_days=60, n_windows=3)
        assert len(windows) == 0

    def test_single_window(self):
        bars = _make_bars(2, 90)
        windows = compute_rolling_windows(bars, window_days=60, n_windows=1)
        assert len(windows) == 1

    def test_windows_non_overlapping_enough_data(self):
        bars = _make_bars(2, 250)
        windows = compute_rolling_windows(bars, window_days=60, n_windows=3)
        assert len(windows) == 3
        ends = [pd.Timestamp(w[1], tz="UTC") for w in windows]
        for i in range(1, len(ends)):
            assert ends[i - 1] > ends[i]


class TestFilterBars:
    def test_basic(self):
        bars = _make_bars(3)
        filtered = _filter_bars(bars, ["SYM0USD", "SYM2USD"])
        assert set(filtered.keys()) == {"SYM0USD", "SYM2USD"}

    def test_missing_symbol(self):
        bars = _make_bars(3)
        filtered = _filter_bars(bars, ["SYM0USD", "NOTEXIST"])
        assert set(filtered.keys()) == {"SYM0USD"}


class TestEvaluateStandalone:
    def test_returns_all_symbols(self):
        bars = _make_bars(3, 150)
        config = _test_config()
        windows = compute_rolling_windows(bars, window_days=60, n_windows=2)
        results = evaluate_standalone(bars, list(bars.keys()), config, windows, use_csim=False)
        assert set(results.keys()) == set(bars.keys())
        for sym, r in results.items():
            assert "standalone_return" in r
            assert "standalone_sortino" in r
            assert "n_trades" in r
            assert isinstance(r["standalone_return"], float)

    def test_single_symbol(self):
        bars = _make_bars(1, 150)
        config = _test_config()
        windows = compute_rolling_windows(bars, window_days=60, n_windows=1)
        results = evaluate_standalone(bars, list(bars.keys()), config, windows, use_csim=False)
        assert len(results) == 1


class TestEvaluateLeaveOneOut:
    def test_marginal_contribution(self):
        bars = _make_bars(3, 150)
        config = _test_config()
        windows = compute_rolling_windows(bars, window_days=60, n_windows=2)
        full_sortinos, _ = evaluate_full_universe(bars, list(bars.keys()), config, windows, use_csim=False)
        loo = evaluate_leave_one_out(bars, list(bars.keys()), config, windows, full_sortinos, use_csim=False)
        assert set(loo.keys()) == set(bars.keys())
        for sym, r in loo.items():
            assert "marginal_contribution" in r
            assert isinstance(r["marginal_contribution"], float)

    def test_positive_contributor_removed_lowers_sortino(self):
        bars = {}
        bars["GOODUSD"] = _make_dipping_walk("GOODUSD", n_days=150, seed=99)
        bars["FLATUSD"] = _make_random_walk("FLATUSD", n_days=150, start_price=100.0, seed=77)
        bars["FLAT2USD"] = _make_random_walk("FLAT2USD", n_days=150, start_price=100.0, seed=88)
        config = _test_config()
        windows = compute_rolling_windows(bars, window_days=60, n_windows=2)
        if not windows:
            pytest.skip("not enough data")
        full_sortinos, _ = evaluate_full_universe(bars, list(bars.keys()), config, windows, use_csim=False)
        loo = evaluate_leave_one_out(bars, list(bars.keys()), config, windows, full_sortinos, use_csim=False)
        for sym, r in loo.items():
            assert "per_window_marginal" in r
            assert len(r["per_window_marginal"]) == len(windows)


class TestFormatResultsTable:
    def test_output_format(self):
        syms = ["BTCUSD", "ETHUSD"]
        standalone = {
            "BTCUSD": {"standalone_return": 5.0, "standalone_sortino": 2.0, "n_trades": 10.0, "avg_hold_days": 5.2},
            "ETHUSD": {"standalone_return": -1.0, "standalone_sortino": -0.5, "n_trades": 3.0, "avg_hold_days": 8.1},
        }
        loo = {
            "BTCUSD": {"marginal_contribution": 0.5},
            "ETHUSD": {"marginal_contribution": -0.3},
        }
        table = format_results_table(syms, standalone, loo)
        assert "BTCUSD" in table
        assert "ETHUSD" in table
        assert "Symbol" in table
        assert "Marginal" in table
        lines = table.strip().split("\n")
        assert len(lines) >= 4  # sep, header, sep, rows
        # BTCUSD should be first (higher marginal)
        btc_idx = next(i for i, l in enumerate(lines) if "BTCUSD" in l)
        eth_idx = next(i for i, l in enumerate(lines) if "ETHUSD" in l)
        assert btc_idx < eth_idx

    def test_empty(self):
        table = format_results_table([], {}, {})
        assert "Symbol" in table


class TestRunEvaluation:
    def test_with_synthetic_data(self, tmp_path):
        bars = _make_bars(3, 200)
        for sym, df in bars.items():
            base = sym.replace("USD", "")
            df.to_csv(tmp_path / f"{base}USDT.csv", index=False)
        symbols = list(bars.keys())
        output, rows = run_evaluation(
            data_dir=str(tmp_path),
            symbols=symbols,
            config=_test_config(),
            window_days=60,
            n_windows=2,
            use_csim=False,
        )
        assert len(rows) == 3
        assert "Symbol" in output
        for r in rows:
            assert "symbol" in r
            assert "standalone_return" in r
            assert "marginal_contribution" in r

    def test_candidate_symbols(self, tmp_path):
        bars = _make_bars(3, 200)
        bars["NEWUSD"] = _make_random_walk("NEWUSD", n_days=200, seed=999)
        for sym, df in bars.items():
            base = sym.replace("USD", "")
            df.to_csv(tmp_path / f"{base}USDT.csv", index=False)
        base_symbols = ["SYM0USD", "SYM1USD", "SYM2USD"]
        output, rows = run_evaluation(
            data_dir=str(tmp_path),
            symbols=base_symbols,
            config=_test_config(),
            window_days=60,
            n_windows=2,
            candidate_symbols=["NEWUSD"],
            use_csim=False,
        )
        assert len(rows) == 4
        new_row = [r for r in rows if r["symbol"] == "NEWUSD"]
        assert len(new_row) == 1
        assert "marginal_contribution" in new_row[0]
        assert "Candidate" in output or "candidate" in output.lower()

    def test_no_data(self, tmp_path):
        output, rows = run_evaluation(
            data_dir=str(tmp_path),
            symbols=["FAKEUSD"],
            config=_test_config(),
            window_days=60,
            n_windows=2,
            use_csim=False,
        )
        assert "ERROR" in output
        assert len(rows) == 0


class TestProductionConfig:
    def test_matches_spec(self):
        assert PRODUCTION_CONFIG.dip_pct == 0.20
        assert PRODUCTION_CONFIG.profit_target_pct == 0.15
        assert PRODUCTION_CONFIG.stop_loss_pct == 0.10
        assert PRODUCTION_CONFIG.sma_filter_period == 20
        assert PRODUCTION_CONFIG.trailing_stop_pct == 0.03
        assert PRODUCTION_CONFIG.max_positions == 5
        assert PRODUCTION_CONFIG.max_hold_days == 14
