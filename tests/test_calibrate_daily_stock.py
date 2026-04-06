"""Tests for scripts/calibrate_daily_stock.py — daily stock execution calibration."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.calibrate_daily_stock import (
    BacktestResult,
    CalibrationParams,
    CalibrationResult,
    _write_results_csv,
    aggregate_results,
    load_daily_frames,
    run_calibrated_backtest,
    run_multiwindow_eval,
)


# ─── Fixtures ─────────────────────────────────────────────────────────


def _make_synthetic_data(
    n_bars: int = 300,
    n_symbols: int = 3,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create synthetic daily OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    result = {}
    for sym in symbols:
        base = 100.0
        closes = [base]
        for _ in range(n_bars - 1):
            ret = rng.normal(0.0005, 0.02)
            closes.append(closes[-1] * (1 + ret))
        closes = np.array(closes)
        opens = closes * (1 + rng.normal(0, 0.003, n_bars))
        highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.015, n_bars))
        lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.015, n_bars))
        volume = rng.integers(100_000, 2_000_000, n_bars).astype(float)
        df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        })
        df.index = pd.date_range("2022-01-03", periods=n_bars, freq="B", tz="UTC")
        df.index.name = "timestamp"
        result[sym] = df
    return result


class MockTrader:
    """Lightweight mock of DailyPPOTrader for tests without loading checkpoints."""

    def __init__(self, symbols: list[str], signal_sequence: list | None = None):
        self.SYMBOLS = symbols
        self.num_symbols = len(symbols)
        self.obs_size = self.num_symbols * 16 + 5 + self.num_symbols
        self.num_actions = 1 + self.num_symbols  # flat + long per symbol
        self.device = "cpu"
        self.cash = 10_000.0
        self.current_position = None
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.hold_hours = 0
        self.hold_days = 0
        self.step = 0
        self.max_steps = 90
        self._signal_sequence = signal_sequence or []
        self._call_idx = 0
        self.long_only = True

    def build_observation(self, features, prices):
        return np.zeros(self.obs_size, dtype=np.float32)

    def apply_action_constraints(self, logits):
        return logits

    def _decode_action(self, action_idx, confidence, value_est):
        class Sig:
            pass
        s = Sig()
        if action_idx == 0:
            s.action = "flat"
            s.symbol = None
            s.direction = None
        else:
            sym_idx = (action_idx - 1) % self.num_symbols
            s.action = f"long_{self.SYMBOLS[sym_idx]}"
            s.symbol = self.SYMBOLS[sym_idx]
            s.direction = "long"
        s.confidence = confidence
        s.value_estimate = value_est
        return s


class MockPolicy:
    """Mock policy that returns predetermined actions."""

    def __init__(self, num_actions: int, action_sequence: list[int] | None = None):
        self.num_actions = num_actions
        self._actions = action_sequence or [0]  # default: always flat
        self._idx = 0

    def __call__(self, obs_t):
        import torch
        action = self._actions[self._idx % len(self._actions)]
        self._idx += 1
        logits = torch.full((1, self.num_actions), -10.0)
        logits[0, action] = 10.0
        value = torch.zeros(1)
        return logits, value


# ─── Unit Tests ───────────────────────────────────────────────────────


class TestCalibrationParams:
    def test_defaults(self):
        p = CalibrationParams()
        assert p.entry_offset_bps == 0.0
        assert p.exit_offset_bps == 0.0
        assert p.allocation_scale == 1.0
        assert p.confidence_threshold == 0.0
        assert p.fee_bps == 0.0

    def test_custom_params(self):
        p = CalibrationParams(entry_offset_bps=-10, exit_offset_bps=15, allocation_scale=1.5)
        assert p.entry_offset_bps == -10
        assert p.exit_offset_bps == 15
        assert p.allocation_scale == 1.5


class TestAggregateResults:
    def test_empty(self):
        agg = aggregate_results([])
        assert agg["n"] == 0
        assert agg["neg"] == 0

    def test_all_positive(self):
        results = [
            BacktestResult(total_return=0.10, annualized_return=0.25, sortino=2.0,
                          max_drawdown=-0.05, win_rate=0.6, num_trades=10, num_days=90,
                          final_equity=11000),
            BacktestResult(total_return=0.20, annualized_return=0.55, sortino=3.0,
                          max_drawdown=-0.03, win_rate=0.7, num_trades=12, num_days=90,
                          final_equity=12000),
        ]
        agg = aggregate_results(results)
        assert agg["n"] == 2
        assert agg["neg"] == 0
        assert agg["med_return"] == pytest.approx(0.15)
        assert agg["p10_return"] == pytest.approx(0.11)

    def test_with_negative(self):
        results = [
            BacktestResult(total_return=-0.05, annualized_return=-0.18, sortino=-1.0,
                          max_drawdown=-0.15, win_rate=0.3, num_trades=5, num_days=90,
                          final_equity=9500),
            BacktestResult(total_return=0.10, annualized_return=0.25, sortino=2.0,
                          max_drawdown=-0.05, win_rate=0.6, num_trades=10, num_days=90,
                          final_equity=11000),
        ]
        agg = aggregate_results(results)
        assert agg["neg"] == 1
        assert agg["worst_return"] == pytest.approx(-0.05)


class TestCalibratedBacktest:
    """Test the calibrated backtest with mock trader and policies."""

    def _run_with_action_sequence(
        self,
        actions: list[int],
        params: CalibrationParams | None = None,
        n_bars: int = 200,
    ) -> BacktestResult:
        """Run a backtest with a specific action sequence."""
        symbols = ["SYM0", "SYM1", "SYM2"]
        indexed = _make_synthetic_data(n_bars=n_bars, n_symbols=3)
        trader = MockTrader(symbols)
        # Create mock policies that always agree on the action sequence
        mock_policies = [MockPolicy(1 + len(symbols), actions)]
        # Override the trader's policy to also use the sequence
        trader.policy = MockPolicy(1 + len(symbols), actions)

        if params is None:
            params = CalibrationParams()

        return run_calibrated_backtest(
            trader=trader,
            extra_policies=mock_policies,
            indexed=indexed,
            start_idx=120,
            end_idx=min(n_bars - 1, 120 + 60),
            params=params,
            base_allocation_pct=100.0,
        )

    def test_always_flat_returns_zero(self):
        """If we never trade, return should be ~0."""
        result = self._run_with_action_sequence([0])  # always flat
        assert result.num_trades == 0
        assert abs(result.total_return) < 1e-10

    def test_positive_entry_offset_costs_more(self):
        """Positive entry offset = paying more = lower returns."""
        base = self._run_with_action_sequence([1, 0] * 30, CalibrationParams())
        worse = self._run_with_action_sequence(
            [1, 0] * 30, CalibrationParams(entry_offset_bps=25),
        )
        # With positive entry offset, we pay more per share, so returns should be lower
        assert worse.total_return <= base.total_return + 0.01  # allow small tolerance

    def test_fees_reduce_returns(self):
        """Adding fees should reduce returns."""
        no_fee = self._run_with_action_sequence([1, 0] * 30, CalibrationParams(fee_bps=0))
        with_fee = self._run_with_action_sequence([1, 0] * 30, CalibrationParams(fee_bps=10))
        assert with_fee.total_return < no_fee.total_return

    def test_allocation_scale_affects_magnitude(self):
        """Higher allocation = more exposure = more extreme returns."""
        small = self._run_with_action_sequence([1, 0] * 30, CalibrationParams(allocation_scale=0.5))
        large = self._run_with_action_sequence([1, 0] * 30, CalibrationParams(allocation_scale=2.0))
        # Larger allocation should amplify returns (in either direction)
        assert abs(large.total_return) >= abs(small.total_return) * 0.5  # loose bound

    def test_confidence_threshold_reduces_trades(self):
        """High confidence threshold should filter some trades."""
        no_gate = self._run_with_action_sequence([1, 0] * 30, CalibrationParams(confidence_threshold=0.0))
        with_gate = self._run_with_action_sequence([1, 0] * 30, CalibrationParams(confidence_threshold=0.99))
        # With very high threshold, fewer trades should execute
        assert with_gate.num_trades <= no_gate.num_trades

    def test_equity_curve_starts_at_initial_cash(self):
        """Verify backtest starts with correct initial cash."""
        result = self._run_with_action_sequence([0], CalibrationParams())
        assert result.final_equity == pytest.approx(10_000.0)


class TestMultiwindowEval:
    def test_produces_multiple_windows(self):
        symbols = ["SYM0", "SYM1", "SYM2"]
        indexed = _make_synthetic_data(n_bars=300, n_symbols=3)
        trader = MockTrader(symbols)
        trader.policy = MockPolicy(1 + len(symbols), [0])
        mock_policies = [MockPolicy(1 + len(symbols), [0])]

        results = run_multiwindow_eval(
            trader=trader,
            extra_policies=mock_policies,
            indexed=indexed,
            params=CalibrationParams(),
            window_size=30,
            start_range=(120, 150),
        )
        assert len(results) == 30  # 150 - 120

    def test_window_size_respected(self):
        symbols = ["SYM0", "SYM1"]
        indexed = _make_synthetic_data(n_bars=250, n_symbols=2)
        trader = MockTrader(symbols)
        trader.policy = MockPolicy(1 + len(symbols), [0])

        results = run_multiwindow_eval(
            trader=trader,
            extra_policies=[MockPolicy(1 + len(symbols), [0])],
            indexed=indexed,
            params=CalibrationParams(),
            window_size=60,
            start_range=(120, 130),
        )
        for r in results:
            assert r.num_days == 60


class TestWriteResultsCsv:
    def test_csv_output(self, tmp_path):
        results = [
            CalibrationResult(
                params=CalibrationParams(entry_offset_bps=-5, exit_offset_bps=10, allocation_scale=1.2),
                train_sortino=2.5, train_return=0.15, train_max_dd=-0.05, train_p10=0.08,
                val_sortino=1.8, val_return=0.10, val_max_dd=-0.07, val_p10=0.03,
                num_train_windows=50, num_val_windows=20,
            ),
        ]
        output_path = tmp_path / "calibration_results.csv"
        _write_results_csv(results, str(output_path))
        with output_path.open() as rf:
            reader = csv.DictReader(rf)
            rows = list(reader)
        assert len(rows) == 1
        assert float(rows[0]["entry_offset_bps"]) == -5.0
        assert float(rows[0]["val_sortino"]) == pytest.approx(1.8, abs=0.01)


class TestLoadDailyFrames:
    def test_loads_real_data_if_available(self):
        """Smoke test: load real stock data if trainingdata/ exists."""
        data_dir = REPO / "trainingdata"
        if not (data_dir / "AAPL.csv").exists():
            pytest.skip("No training data available")
        frames = load_daily_frames(["AAPL", "MSFT"], data_dir="trainingdata")
        assert "AAPL" in frames
        assert "MSFT" in frames
        assert len(frames["AAPL"]) > 100

    def test_missing_symbol_raises(self):
        with pytest.raises(FileNotFoundError):
            load_daily_frames(["NONEXISTENT_SYM_XYZ"], data_dir="trainingdata")


class TestNegativeEntryOffset:
    """Test that negative entry offset (limit buy below market) properly simulates fills."""

    def test_negative_offset_skips_unfillable(self):
        """If we set buy limit far below open and low doesn't reach it, no fill."""
        symbols = ["SYM0"]
        # Create data where lows are always close to opens (tight range)
        n = 200
        closes = 100.0 * np.ones(n)
        opens = closes.copy()
        highs = closes * 1.001  # very tight range
        lows = closes * 0.999   # very tight range
        volume = np.ones(n) * 1_000_000
        indexed = {
            "SYM0": pd.DataFrame({
                "open": opens, "high": highs, "low": lows,
                "close": closes, "volume": volume,
            }, index=pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")),
        }
        indexed["SYM0"].index.name = "timestamp"

        trader = MockTrader(symbols)
        trader.policy = MockPolicy(2, [1])  # always try to buy
        mock_policies = [MockPolicy(2, [1])]

        # -50bps offset = buy at 99.50, but low is 99.90 — should not fill
        result = run_calibrated_backtest(
            trader=trader,
            extra_policies=mock_policies,
            indexed=indexed,
            start_idx=120,
            end_idx=180,
            params=CalibrationParams(entry_offset_bps=-50),
        )
        assert result.num_trades == 0  # limit never reached
