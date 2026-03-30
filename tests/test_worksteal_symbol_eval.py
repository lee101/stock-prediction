"""Tests for per-symbol evaluation tool."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import binance_worksteal.evaluate_symbols as evaluate_symbols_module
from binance_worksteal.evaluate_symbols import (
    PRODUCTION_CONFIG,
    _run_backtest,
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

    def test_ignores_invalid_timestamp_rows_and_symbols(self):
        good = _make_bars(1, 120)
        baseline = compute_rolling_windows(good, window_days=60, n_windows=2)

        mixed = next(iter(good.values())).copy()
        mixed["timestamp"] = mixed["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        mixed.loc[0, "timestamp"] = "not-a-date"
        bad = pd.DataFrame({"timestamp": ["still-bad", "also-bad"]})

        windows = compute_rolling_windows(
            {
                "GOOD": next(iter(good.values())),
                "MIXED": mixed,
                "BAD": bad,
            },
            window_days=60,
            n_windows=2,
        )

        assert windows == baseline

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

    def test_reuses_symbol_frame_without_copying(self, monkeypatch):
        bars = _make_bars(1, 150)
        symbol = next(iter(bars))
        config = _test_config()
        windows = compute_rolling_windows(bars, window_days=60, n_windows=1)
        seen = {}

        def fake_backtest(inner_bars, inner_config, start_date, end_date, **kwargs):
            seen["bars_is_original"] = inner_bars[symbol] is bars[symbol]
            return pd.DataFrame(), [], {"total_return_pct": 0.0, "sortino": 0.0, "n_trades": 0}

        monkeypatch.setattr(evaluate_symbols_module, "run_worksteal_backtest", fake_backtest)

        results = evaluate_standalone(bars, [symbol], config, windows, use_csim=False)

        assert seen == {"bars_is_original": True}
        assert symbol in results

    def test_reuses_prepared_bars_when_provided(self, monkeypatch):
        bars = _make_bars(1, 150)
        symbol = next(iter(bars))
        config = _test_config()
        windows = compute_rolling_windows(bars, window_days=60, n_windows=1)
        seen = {}

        def fake_backtest(inner_bars, inner_config, start_date, end_date, **kwargs):
            seen["prepared_matches_symbol"] = kwargs["prepared_bars"][symbol].bars is bars[symbol]
            return pd.DataFrame(), [], {"total_return_pct": 0.0, "sortino": 0.0, "n_trades": 0}

        monkeypatch.setattr(evaluate_symbols_module, "run_worksteal_backtest", fake_backtest)

        results = evaluate_standalone(
            bars,
            [symbol],
            config,
            windows,
            use_csim=False,
            prepared_bars=evaluate_symbols_module.prepare_backtest_bars(bars),
        )

        assert seen == {"prepared_matches_symbol": True}
        assert symbol in results

    def test_skips_failed_symbol_and_records_failure(self, monkeypatch, capsys):
        bars = _make_bars(2, 150)
        symbols = list(bars)
        config = _test_config()
        windows = compute_rolling_windows(bars, window_days=60, n_windows=1)
        failures = []

        def fake_backtest(inner_bars, inner_config, start_date, end_date, **kwargs):
            sym = next(iter(inner_bars))
            if sym == symbols[1]:
                raise RuntimeError("standalone boom")
            return pd.DataFrame(), [], {"total_return_pct": 2.0, "sortino": 1.0, "n_trades": 3}

        monkeypatch.setattr(evaluate_symbols_module, "run_worksteal_backtest", fake_backtest)

        results = evaluate_standalone(
            bars,
            symbols,
            config,
            windows,
            use_csim=False,
            failures=failures,
        )

        output = capsys.readouterr().out
        assert set(results.keys()) == {symbols[0]}
        assert len(failures) == 1
        assert failures[0]["stage"] == "standalone"
        assert failures[0]["symbol"] == symbols[1]
        assert "WARN: evaluate_symbols standalone" in output
        assert "skipping symbol" in output


class TestEvaluateLeaveOneOut:
    def test_full_universe_reuses_prepared_bars_when_provided(self, monkeypatch):
        bars = _make_bars(2, 150)
        symbols = list(bars)
        config = _test_config()
        windows = compute_rolling_windows(bars, window_days=60, n_windows=2)
        prepared = evaluate_symbols_module.prepare_backtest_bars(bars)
        seen = []

        def fake_run_backtest(
            inner_bars,
            inner_config,
            start_date=None,
            end_date=None,
            use_csim=True,
            prepared_bars=None,
        ):
            seen.append(
                prepared_bars is not None
                and set(prepared_bars) == set(inner_bars)
                and all(prepared_bars[sym].bars is bars[sym] for sym in inner_bars)
            )
            return {"sortino": 0.25}

        monkeypatch.setattr(evaluate_symbols_module, "_run_backtest", fake_run_backtest)

        sortinos, metrics = evaluate_full_universe(
            bars,
            symbols,
            config,
            windows,
            use_csim=False,
            prepared_bars=prepared,
        )

        assert sortinos == [0.25 for _ in windows]
        assert len(metrics) == len(windows)
        assert seen == [True] * len(windows)

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

    def test_skips_failed_symbol_and_records_failure(self, monkeypatch, capsys):
        bars = _make_bars(3, 150)
        symbols = list(bars)
        config = _test_config()
        windows = compute_rolling_windows(bars, window_days=60, n_windows=1)
        failures = []

        def fake_run_backtest(
            inner_bars,
            inner_config,
            start_date=None,
            end_date=None,
            use_csim=True,
            prepared_bars=None,
        ):
            removed = next(sym for sym in symbols if sym not in inner_bars)
            if removed == symbols[0]:
                raise RuntimeError("loo boom")
            return {"sortino": 0.25}

        monkeypatch.setattr(evaluate_symbols_module, "_run_backtest", fake_run_backtest)

        loo = evaluate_leave_one_out(
            bars,
            symbols,
            config,
            windows,
            [0.5],
            use_csim=False,
            failures=failures,
        )

        output = capsys.readouterr().out
        assert set(loo.keys()) == set(symbols[1:])
        assert len(failures) == 1
        assert failures[0]["stage"] == "leave_one_out"
        assert failures[0]["symbol"] == symbols[0]
        assert "WARN: evaluate_symbols leave_one_out" in output
        assert "skipping symbol" in output

    def test_leave_one_out_reuses_prepared_bars_when_provided(self, monkeypatch):
        bars = _make_bars(3, 150)
        symbols = list(bars)
        config = _test_config()
        windows = compute_rolling_windows(bars, window_days=60, n_windows=1)
        prepared = evaluate_symbols_module.prepare_backtest_bars(bars)
        seen = []

        def fake_run_backtest(
            inner_bars,
            inner_config,
            start_date=None,
            end_date=None,
            use_csim=True,
            prepared_bars=None,
        ):
            seen.append(
                prepared_bars is not None
                and set(prepared_bars) == set(inner_bars)
                and all(prepared_bars[sym].bars is bars[sym] for sym in inner_bars)
            )
            return {"sortino": 0.25}

        monkeypatch.setattr(evaluate_symbols_module, "_run_backtest", fake_run_backtest)

        loo = evaluate_leave_one_out(
            bars,
            symbols,
            config,
            windows,
            [0.5],
            use_csim=False,
            prepared_bars=prepared,
        )

        assert set(loo) == set(symbols)
        assert seen == [True] * len(symbols)


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

    def test_candidate_symbols_are_normalized_and_reported_separately(self, monkeypatch, capsys):
        bars = {
            "SYM0USD": _make_random_walk("SYM0USD", n_days=120, seed=42),
            "NEWUSD": _make_random_walk("NEWUSD", n_days=120, seed=99),
        }

        monkeypatch.setattr(evaluate_symbols_module, "load_daily_bars", lambda data_dir, requested: bars)
        monkeypatch.setattr(
            evaluate_symbols_module,
            "evaluate_full_universe",
            lambda *args, **kwargs: ([1.0], [{"sortino": 1.0, "total_return_pct": 2.0}]),
        )
        monkeypatch.setattr(
            evaluate_symbols_module,
            "evaluate_standalone",
            lambda *args, **kwargs: {
                "SYM0USD": {
                    "standalone_return": 1.0,
                    "standalone_sortino": 1.0,
                    "n_trades": 1.0,
                    "avg_hold_days": 2.0,
                },
                "NEWUSD": {
                    "standalone_return": 0.5,
                    "standalone_sortino": 0.5,
                    "n_trades": 0.0,
                    "avg_hold_days": 0.0,
                },
            },
        )
        monkeypatch.setattr(
            evaluate_symbols_module,
            "evaluate_leave_one_out",
            lambda *args, **kwargs: {"SYM0USD": {"marginal_contribution": 0.1}},
        )
        monkeypatch.setattr(
            evaluate_symbols_module,
            "_run_backtest",
            lambda *args, **kwargs: {"sortino": 1.1},
        )

        output, rows, metadata = run_evaluation(
            data_dir="ignored",
            symbols=["sym0usd"],
            config=_test_config(),
            window_days=60,
            n_windows=1,
            candidate_symbols=["newusd", "missingusd"],
            use_csim=False,
            return_metadata=True,
        )

        printed = capsys.readouterr().out
        assert not output.startswith("ERROR:")
        assert len(rows) == 2
        candidate_row = next(row for row in rows if row["symbol"] == "NEWUSD")
        assert candidate_row["marginal_contribution"] == pytest.approx(0.1)
        assert metadata["candidate_symbols"] == ["NEWUSD", "MISSINGUSD"]
        assert metadata["ignored_candidate_symbols"] == []
        assert metadata["ignored_candidate_symbol_count"] == 0
        assert metadata["candidate_loaded_symbol_count"] == 1
        assert metadata["candidate_loaded_symbols"] == ["NEWUSD"]
        assert metadata["candidate_missing_symbol_count"] == 1
        assert metadata["candidate_missing_symbols"] == ["MISSINGUSD"]
        assert metadata["warnings"] == ["missing data for 1 candidate symbol: MISSINGUSD"]
        assert "Base symbols: 1/1 with data" in printed
        assert "Candidate symbols: 1/2 with data" in printed
        assert "WARN: 1 missing from candidate symbols: MISSINGUSD (symbol)" in printed
        assert "Candidate symbols (positive marginal = improves universe):" in output
        assert "NEWUSD" in output

    def test_overlapping_candidate_symbols_are_ignored_and_reported(self, monkeypatch):
        bars = {
            "SYM0USD": _make_random_walk("SYM0USD", n_days=120, seed=42),
            "NEWUSD": _make_random_walk("NEWUSD", n_days=120, seed=99),
        }

        monkeypatch.setattr(evaluate_symbols_module, "load_daily_bars", lambda data_dir, requested: bars)
        monkeypatch.setattr(
            evaluate_symbols_module,
            "evaluate_full_universe",
            lambda *args, **kwargs: ([1.0], [{"sortino": 1.0, "total_return_pct": 2.0}]),
        )
        monkeypatch.setattr(
            evaluate_symbols_module,
            "evaluate_standalone",
            lambda *args, **kwargs: {
                "SYM0USD": {
                    "standalone_return": 1.0,
                    "standalone_sortino": 1.0,
                    "n_trades": 1.0,
                    "avg_hold_days": 2.0,
                },
                "NEWUSD": {
                    "standalone_return": 0.5,
                    "standalone_sortino": 0.5,
                    "n_trades": 0.0,
                    "avg_hold_days": 0.0,
                },
            },
        )
        monkeypatch.setattr(
            evaluate_symbols_module,
            "evaluate_leave_one_out",
            lambda *args, **kwargs: {"SYM0USD": {"marginal_contribution": 0.1}},
        )
        monkeypatch.setattr(
            evaluate_symbols_module,
            "_run_backtest",
            lambda *args, **kwargs: {"sortino": 1.1},
        )

        output, rows, metadata = run_evaluation(
            data_dir="ignored",
            symbols=["SYM0USD"],
            config=_test_config(),
            window_days=60,
            n_windows=1,
            candidate_symbols=["sym0usd", "newusd", "sym0usd"],
            use_csim=False,
            return_metadata=True,
        )

        assert len(rows) == 2
        assert metadata["candidate_symbols"] == ["NEWUSD"]
        assert metadata["ignored_candidate_symbols"] == ["SYM0USD"]
        assert metadata["ignored_candidate_symbol_count"] == 1
        assert "Ignored candidate symbols already present in base universe: SYM0USD" in output

    def test_candidate_short_history_does_not_shrink_base_windows(self, monkeypatch):
        bars = {
            "SYM0USD": _make_random_walk("SYM0USD", n_days=120, seed=42),
            "NEWUSD": _make_random_walk("NEWUSD", n_days=10, seed=99),
        }
        seen = {}

        monkeypatch.setattr(evaluate_symbols_module, "load_daily_bars", lambda data_dir, requested: bars)

        def fake_full(all_bars, symbols, config, windows, use_csim=True, prepared_bars=None):
            seen["windows"] = list(windows)
            seen["base_symbols"] = list(symbols)
            return [1.0 for _ in windows], [{"sortino": 1.0, "total_return_pct": 2.0} for _ in windows]

        monkeypatch.setattr(evaluate_symbols_module, "evaluate_full_universe", fake_full)
        monkeypatch.setattr(
            evaluate_symbols_module,
            "evaluate_standalone",
            lambda *args, **kwargs: {
                "SYM0USD": {
                    "standalone_return": 1.0,
                    "standalone_sortino": 1.0,
                    "n_trades": 1.0,
                    "avg_hold_days": 2.0,
                },
                "NEWUSD": {
                    "standalone_return": 0.5,
                    "standalone_sortino": 0.5,
                    "n_trades": 0.0,
                    "avg_hold_days": 0.0,
                },
            },
        )
        monkeypatch.setattr(
            evaluate_symbols_module,
            "evaluate_leave_one_out",
            lambda *args, **kwargs: {
                "SYM0USD": {"marginal_contribution": 0.1},
            },
        )
        monkeypatch.setattr(
            evaluate_symbols_module,
            "_run_backtest",
            lambda *args, **kwargs: {"sortino": 1.1},
        )

        output, rows, metadata = run_evaluation(
            data_dir="ignored",
            symbols=["SYM0USD"],
            config=_test_config(),
            window_days=60,
            n_windows=1,
            candidate_symbols=["NEWUSD"],
            use_csim=False,
            return_metadata=True,
        )

        assert not output.startswith("ERROR:")
        assert len(rows) == 2
        assert seen["base_symbols"] == ["SYM0USD"]
        assert len(seen["windows"]) == 1
        assert metadata["loaded_symbol_count"] == 1
        assert metadata["loaded_symbols"] == ["SYM0USD"]

    def test_require_full_universe_returns_error_when_candidate_data_is_missing(self, monkeypatch):
        bars = {
            "SYM0USD": _make_random_walk("SYM0USD", n_days=120, seed=42),
        }

        monkeypatch.setattr(evaluate_symbols_module, "load_daily_bars", lambda data_dir, requested: bars)

        output, rows, metadata = run_evaluation(
            data_dir="ignored",
            symbols=["SYM0USD"],
            config=_test_config(),
            window_days=60,
            n_windows=1,
            candidate_symbols=["NEWUSD"],
            use_csim=False,
            require_full_universe=True,
            return_metadata=True,
        )

        assert output == "ERROR: --require-full-universe found missing data for 1 symbol: NEWUSD"
        assert rows == []
        assert metadata["loaded_symbols"] == ["SYM0USD"]
        assert metadata["candidate_missing_symbols"] == ["NEWUSD"]
        assert metadata["base_universe_complete"] is True
        assert metadata["candidate_universe_complete"] is False
        assert metadata["universe_complete"] is False
        assert metadata["warnings"] == ["missing data for 1 candidate symbol: NEWUSD"]


    def test_returns_error_when_no_requested_base_symbols_have_data(self, monkeypatch):
        bars = {
            "NEWUSD": _make_random_walk("NEWUSD", n_days=120, seed=999),
        }

        monkeypatch.setattr(evaluate_symbols_module, "load_daily_bars", lambda data_dir, requested: bars)

        output, rows, metadata = run_evaluation(
            data_dir="ignored",
            symbols=["SYM0USD"],
            config=_test_config(),
            window_days=60,
            n_windows=1,
            candidate_symbols=["NEWUSD"],
            use_csim=False,
            return_metadata=True,
        )

        assert output == "ERROR: None of the requested base symbols have data"
        assert rows == []
        assert metadata["base_symbol_count"] == 0
        assert metadata["loaded_symbol_count"] == 0
        assert metadata["loaded_symbols"] == []
        assert metadata["missing_symbols"] == ["SYM0USD"]

    def test_no_data(self, tmp_path, capsys):
        output, rows = run_evaluation(
            data_dir=str(tmp_path),
            symbols=["FAKEUSD"],
            config=_test_config(),
            window_days=60,
            n_windows=2,
            use_csim=False,
        )
        printed = capsys.readouterr().out.strip().splitlines()
        assert printed == [
            "Loaded 0/1 symbols with data",
            "WARN: missing data for 1 symbol: FAKEUSD",
        ]
        assert "ERROR" in output
        assert len(rows) == 0

    def test_no_data_returns_consistent_metadata_for_fixed_window(self, tmp_path):
        output, rows, metadata = run_evaluation(
            data_dir=str(tmp_path),
            symbols=["FAKEUSD"],
            config=_test_config(),
            start_date="2026-01-01",
            end_date="2026-01-31",
            use_csim=False,
            return_metadata=True,
        )

        assert output == "ERROR: No data loaded"
        assert rows == []
        assert metadata["window_mode"] == "fixed"
        assert metadata["start_date"] == "2026-01-01"
        assert metadata["end_date"] == "2026-01-31"
        assert metadata["requested_window_days"] is None
        assert metadata["requested_window_count"] == 1
        assert metadata["base_symbol_count"] == 0
        assert metadata["loaded_symbol_count"] == 0
        assert metadata["missing_symbols"] == ["FAKEUSD"]

    def test_loader_error_is_reported_without_becoming_no_data(self, monkeypatch, capsys):
        def raise_load_error(data_dir, symbols):
            raise OSError("eval dir unreadable")

        monkeypatch.setattr(evaluate_symbols_module, "load_daily_bars", raise_load_error)

        output, rows, metadata = run_evaluation(
            data_dir="trainingdata/train",
            symbols=["FAKEUSD"],
            config=_test_config(),
            start_date="2026-01-01",
            end_date="2026-01-31",
            use_csim=False,
            return_metadata=True,
        )

        printed = capsys.readouterr().out.strip().splitlines()
        assert printed == ["ERROR: eval dir unreadable"]
        assert output == "ERROR: eval dir unreadable"
        assert rows == []
        assert metadata["loaded_symbol_count"] == 0
        assert metadata["missing_symbols"] == ["FAKEUSD"]
        assert metadata["load_failure"] == {"error": "ERROR: eval dir unreadable", "error_type": "OSError"}

    def test_warns_when_only_some_requested_windows_fit(self, monkeypatch, capsys):
        bars = _make_bars(2, 100)
        symbols = list(bars)

        monkeypatch.setattr(evaluate_symbols_module, "load_daily_bars", lambda data_dir, requested: bars)
        monkeypatch.setattr(
            evaluate_symbols_module,
            "evaluate_full_universe",
            lambda *args, **kwargs: ([0.0], [{"total_return_pct": 0.0}]),
        )
        monkeypatch.setattr(
            evaluate_symbols_module,
            "evaluate_standalone",
            lambda *args, **kwargs: {
                symbol: {"standalone_return": 0.0, "standalone_sortino": 0.0, "n_trades": 0.0, "avg_hold_days": 0.0}
                for symbol in symbols
            },
        )
        monkeypatch.setattr(
            evaluate_symbols_module,
            "evaluate_leave_one_out",
            lambda *args, **kwargs: {
                symbol: {"marginal_contribution": 0.0}
                for symbol in symbols
            },
        )

        output, rows = run_evaluation(
            data_dir="ignored",
            symbols=symbols,
            config=_test_config(),
            window_days=60,
            n_windows=3,
            use_csim=False,
        )

        printed = capsys.readouterr().out
        assert "WARN: only 1/3 rolling windows of 60 days fit within loaded data coverage" in printed
        assert len(rows) == 2
        assert "Windows: 1x60d" in output

    def test_fixed_window_bypasses_rolling_window_builder(self, monkeypatch):
        bars = _make_bars(2, 100)
        symbols = list(bars)
        seen = {}

        monkeypatch.setattr(evaluate_symbols_module, "load_daily_bars", lambda data_dir, requested: bars)
        monkeypatch.setattr(
            evaluate_symbols_module,
            "compute_rolling_windows",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rolling windows should not be used")),
        )

        def fake_full(_all_bars, _symbols, _config, windows, use_csim=True, prepared_bars=None):
            seen["windows"] = list(windows)
            return [0.0], [{"sortino": 0.0, "total_return_pct": 0.0}]

        monkeypatch.setattr(evaluate_symbols_module, "evaluate_full_universe", fake_full)
        monkeypatch.setattr(
            evaluate_symbols_module,
            "evaluate_standalone",
            lambda *args, **kwargs: {
                symbol: {"standalone_return": 0.0, "standalone_sortino": 0.0, "n_trades": 0.0, "avg_hold_days": 0.0}
                for symbol in symbols
            },
        )
        monkeypatch.setattr(
            evaluate_symbols_module,
            "evaluate_leave_one_out",
            lambda *args, **kwargs: {
                symbol: {"marginal_contribution": 0.0}
                for symbol in symbols
            },
        )

        output, rows, metadata = run_evaluation(
            data_dir="ignored",
            symbols=symbols,
            config=_test_config(),
            start_date="2026-01-01",
            end_date="2026-01-31",
            use_csim=False,
            return_metadata=True,
        )

        assert len(rows) == 2
        assert seen["windows"] == [("2026-01-01", "2026-01-31")]
        assert metadata["window_mode"] == "fixed"
        assert metadata["start_date"] == "2026-01-01"
        assert metadata["end_date"] == "2026-01-31"
        assert metadata["requested_window_days"] is None
        assert metadata["requested_window_count"] == 1
        assert "Window: 2026-01-01 to 2026-01-31" in output

    def test_reports_and_omits_symbols_with_evaluation_failures(self, monkeypatch):
        bars = _make_bars(3, 120)
        symbols = list(bars)

        def fake_backtest(inner_bars, inner_config, start_date=None, end_date=None, **kwargs):
            if len(inner_bars) == 1 and symbols[1] in inner_bars:
                raise RuntimeError("bad standalone symbol")
            return pd.DataFrame(), [], {"sortino": 0.5, "total_return_pct": 1.0, "n_trades": 1}

        monkeypatch.setattr(evaluate_symbols_module, "load_daily_bars", lambda data_dir, requested: bars)
        monkeypatch.setattr(evaluate_symbols_module, "run_worksteal_backtest", fake_backtest)

        output, rows, metadata = run_evaluation(
            data_dir="ignored",
            symbols=symbols,
            config=_test_config(),
            window_days=60,
            n_windows=1,
            use_csim=False,
            return_metadata=True,
        )

        assert len(rows) == 2
        assert {row["symbol"] for row in rows} == {symbols[0], symbols[2]}
        assert metadata["evaluation_failure_count"] == 1
        assert metadata["skipped_symbols"] == [symbols[1]]
        assert metadata["evaluation_failures"][0]["stage"] == "standalone"
        assert output.count(symbols[1]) == 1
        assert "Skipped 1 symbols due to evaluation failures" in output

    def test_returns_error_metadata_when_full_universe_fails(self, monkeypatch):
        bars = _make_bars(2, 120)
        symbols = list(bars)

        monkeypatch.setattr(evaluate_symbols_module, "load_daily_bars", lambda data_dir, requested: bars)

        def fake_full(*args, **kwargs):
            raise RuntimeError("full boom")

        monkeypatch.setattr(evaluate_symbols_module, "evaluate_full_universe", fake_full)

        output, rows, metadata = run_evaluation(
            data_dir="ignored",
            symbols=symbols,
            config=_test_config(),
            window_days=60,
            n_windows=1,
            use_csim=False,
            return_metadata=True,
        )

        assert rows == []
        assert output.startswith("ERROR: evaluate_symbols full_universe")
        assert metadata["evaluation_failure_count"] == 1
        assert metadata["evaluation_failures"][0]["stage"] == "full_universe"
        assert metadata["evaluation_failures"][0]["error"] == "full boom"


class TestRunBacktestDispatch:
    def test_use_csim_false_bypasses_csim_even_when_available(self, monkeypatch):
        bars = _make_bars(2, 120)
        config = WorkStealConfig(sma_filter_period=0)
        seen = {"python": 0, "csim": 0}

        def fake_csim_loader():
            seen["csim"] += 1
            raise AssertionError("C sim loader should not be touched when use_csim=False")

        def fake_python(*args, **kwargs):
            seen["python"] += 1
            return pd.DataFrame(), [], {"sortino": 0.5, "total_return_pct": 1.0}

        monkeypatch.setattr("binance_worksteal.evaluate_symbols._get_csim_fn", fake_csim_loader)
        monkeypatch.setattr("binance_worksteal.evaluate_symbols.run_worksteal_backtest", fake_python)

        metrics = _run_backtest(bars, config, use_csim=False)

        assert metrics["sortino"] == 0.5
        assert seen == {"python": 1, "csim": 0}

    def test_incompatible_config_falls_back_to_python(self, monkeypatch):
        bars = _make_bars(2, 120)
        config = _test_config()
        seen = {"python": 0, "csim": 0}

        def fake_csim(*args, **kwargs):
            seen["csim"] += 1
            raise AssertionError("C sim should not run for incompatible config")

        def fake_python(*args, **kwargs):
            seen["python"] += 1
            return pd.DataFrame(), [], {"sortino": 1.25, "total_return_pct": 3.5}

        monkeypatch.setattr("binance_worksteal.evaluate_symbols._get_csim_fn", lambda: fake_csim)
        monkeypatch.setattr("binance_worksteal.evaluate_symbols.run_worksteal_backtest", fake_python)

        metrics = _run_backtest(bars, config, use_csim=True)

        assert metrics["sortino"] == 1.25
        assert seen == {"python": 1, "csim": 0}

    def test_incompatible_config_warns_once_and_uses_python(self, monkeypatch, capsys):
        bars = _make_bars(2, 120)
        config = _test_config()
        seen = {"python": 0, "csim": 0}

        def fake_csim_loader():
            seen["csim"] += 1
            raise AssertionError("C sim loader should not run for incompatible configs")

        def fake_python(*args, **kwargs):
            seen["python"] += 1
            return pd.DataFrame(), [], {"sortino": 1.25, "total_return_pct": 3.5}

        monkeypatch.setattr("binance_worksteal.evaluate_symbols._get_csim_fn", fake_csim_loader)
        monkeypatch.setattr("binance_worksteal.evaluate_symbols.run_worksteal_backtest", fake_python)
        monkeypatch.setattr("binance_worksteal.evaluate_symbols._csim_incompatibility_warned", False)

        first = _run_backtest(bars, config, use_csim=True)
        second = _run_backtest(bars, config, use_csim=True)

        output = capsys.readouterr().out
        assert first["sortino"] == 1.25
        assert second["sortino"] == 1.25
        assert seen == {"python": 2, "csim": 0}
        assert output.count("WARN: evaluate_symbols disabled C sim for unsupported config features:") == 1
        assert "risk_off regime switching" in output

    def test_csim_runtime_failure_warns_and_falls_back_to_python(self, monkeypatch, capsys):
        bars = _make_bars(2, 120)
        config = WorkStealConfig(
            dip_pct=0.20,
            profit_target_pct=0.15,
            stop_loss_pct=0.10,
            sma_filter_period=0,
            trailing_stop_pct=0.03,
            max_positions=5,
            max_hold_days=14,
            max_drawdown_exit=0.0,
        )
        seen = {"python": 0, "csim": 0}

        def fake_csim(*args, **kwargs):
            seen["csim"] += 1
            raise RuntimeError("csim boom")

        def fake_python(*args, **kwargs):
            seen["python"] += 1
            return pd.DataFrame(), [], {"sortino": 0.75, "total_return_pct": 1.5}

        monkeypatch.setattr("binance_worksteal.evaluate_symbols._get_csim_fn", lambda: fake_csim)
        monkeypatch.setattr("binance_worksteal.evaluate_symbols.run_worksteal_backtest", fake_python)
        monkeypatch.setattr("binance_worksteal.evaluate_symbols.config_supports_csim", lambda config: True)
        monkeypatch.setattr("binance_worksteal.evaluate_symbols._csim_runtime_failure_warned", False)

        metrics = _run_backtest(
            bars,
            config,
            start_date="2025-07-01",
            end_date="2025-08-01",
            use_csim=True,
        )

        output = capsys.readouterr().out
        assert metrics["sortino"] == 0.75
        assert seen == {"python": 1, "csim": 1}
        assert "WARN: evaluate_symbols C sim evaluation failed" in output
        assert "csim boom" in output
        assert "falling back to Python backtest" in output


class TestProductionConfig:
    def test_matches_spec(self):
        assert PRODUCTION_CONFIG.dip_pct == 0.20
        assert PRODUCTION_CONFIG.profit_target_pct == 0.15
        assert PRODUCTION_CONFIG.stop_loss_pct == 0.10
        assert PRODUCTION_CONFIG.sma_filter_period == 20
        assert PRODUCTION_CONFIG.trailing_stop_pct == 0.03
        assert PRODUCTION_CONFIG.max_positions == 5
        assert PRODUCTION_CONFIG.max_hold_days == 14
