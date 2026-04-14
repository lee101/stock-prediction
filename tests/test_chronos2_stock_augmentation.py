"""
Tests for chronos2_stock_augmentation.py

Covers:
  - load_ohlc_csv: valid file, missing columns, too-short, non-existent
  - load_all_series: directory scanning, ignores non-CSV, handles subdirs
  - create_sliding_daily_from_hourly: correct OHLC aggregation, offset logic, min-length filter
  - to_return_series: correct return computation, edge cases
  - prepare_all_training_series: combined data loading, return variant counts
  - split_series_list: train/val/test split, short-series edge case
  - AugmentedChronos2Dataset: augmentation applied in TRAIN mode, skipped in VAL mode
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chronos2_stock_augmentation import (
    AugConfig,
    OHLC_COLS,
    create_sliding_daily_from_hourly,
    load_all_series,
    load_ohlc_csv,
    prepare_all_training_series,
    split_series_list,
    to_return_series,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_csv(path: Path, n_rows: int = 100, cols=OHLC_COLS, seed: int = 0) -> None:
    """Write a minimal OHLC CSV to path."""
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n_rows))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="D"),
        "open":  prices,
        "high":  prices + rng.uniform(0, 1, n_rows),
        "low":   prices - rng.uniform(0, 1, n_rows),
        "close": prices + rng.normal(0, 0.2, n_rows),
        "volume": rng.uniform(1e5, 1e6, n_rows),
    })
    for col in cols:
        if col not in df.columns:
            df[col] = prices
    df.to_csv(path, index=False)


def _make_hourly_array(n_hours: int = 700, seed: int = 1) -> np.ndarray:
    """Return synthetic hourly OHLC array shape (4, n_hours)."""
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.1, n_hours))
    open_  = prices
    high_  = prices + rng.uniform(0, 0.5, n_hours)
    low_   = prices - rng.uniform(0, 0.5, n_hours)
    close_ = prices + rng.normal(0, 0.1, n_hours)
    return np.stack([open_, high_, low_, close_]).astype(np.float32)


# ---------------------------------------------------------------------------
# load_ohlc_csv
# ---------------------------------------------------------------------------

class TestLoadOhlcCsv:
    def test_valid_csv(self, tmp_path):
        csv = tmp_path / "TEST.csv"
        _make_csv(csv, 100)
        arr = load_ohlc_csv(csv)
        assert arr is not None
        assert arr.shape == (4, 100)
        assert arr.dtype == np.float32

    def test_missing_column(self, tmp_path):
        csv = tmp_path / "BAD.csv"
        df = pd.DataFrame({"open": [1.0, 2.0], "high": [1.5, 2.5], "low": [0.5, 1.5]})
        df.to_csv(csv, index=False)
        assert load_ohlc_csv(csv) is None

    def test_too_short(self, tmp_path):
        csv = tmp_path / "SHORT.csv"
        _make_csv(csv, 20)  # below default min_length=50
        assert load_ohlc_csv(csv, min_length=50) is None

    def test_nonexistent_file(self, tmp_path):
        assert load_ohlc_csv(tmp_path / "MISSING.csv") is None

    def test_all_nan_column(self, tmp_path):
        csv = tmp_path / "NAN.csv"
        df = pd.DataFrame({
            "open":  [float("nan")] * 60,
            "high":  [float("nan")] * 60,
            "low":   [float("nan")] * 60,
            "close": [float("nan")] * 60,
        })
        df.to_csv(csv, index=False)
        assert load_ohlc_csv(csv) is None

    def test_strips_nan_edges(self, tmp_path):
        """Leading/trailing NaNs should be dropped."""
        csv = tmp_path / "EDGE.csv"
        n = 80
        opens  = [float("nan")] * 5 + [100.0] * n + [float("nan")] * 3
        df = pd.DataFrame({"open": opens, "high": opens, "low": opens, "close": opens})
        df.to_csv(csv, index=False)
        arr = load_ohlc_csv(csv, min_length=50)
        # Only the middle n rows should be valid
        assert arr is not None
        assert arr.shape[1] == n

    def test_custom_min_length(self, tmp_path):
        csv = tmp_path / "CUSTOM.csv"
        _make_csv(csv, 30)
        assert load_ohlc_csv(csv, min_length=25) is not None
        assert load_ohlc_csv(csv, min_length=35) is None


# ---------------------------------------------------------------------------
# load_all_series
# ---------------------------------------------------------------------------

class TestLoadAllSeries:
    def test_loads_multiple_csvs(self, tmp_path):
        for sym in ("AAPL", "GOOG", "TSLA"):
            _make_csv(tmp_path / f"{sym}.csv", 100)
        series = load_all_series(tmp_path)
        assert len(series) == 3
        symbols = {s["symbol"] for s in series}
        assert symbols == {"AAPL", "GOOG", "TSLA"}

    def test_ignores_non_csv(self, tmp_path):
        _make_csv(tmp_path / "AAPL.csv", 100)
        (tmp_path / "README.txt").write_text("ignore me")
        assert len(load_all_series(tmp_path)) == 1

    def test_skips_subdirectories(self, tmp_path):
        _make_csv(tmp_path / "AAPL.csv", 100)
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        _make_csv(subdir / "MSFT.csv", 100)  # should not be loaded
        assert len(load_all_series(tmp_path)) == 1

    def test_empty_dir(self, tmp_path):
        assert load_all_series(tmp_path) == []

    def test_skips_short_series(self, tmp_path):
        _make_csv(tmp_path / "SHORT.csv", 20)
        _make_csv(tmp_path / "LONG.csv", 100)
        series = load_all_series(tmp_path, min_length=50)
        assert len(series) == 1
        assert series[0]["symbol"] == "LONG"


# ---------------------------------------------------------------------------
# create_sliding_daily_from_hourly
# ---------------------------------------------------------------------------

class TestSlidingDailyFromHourly:
    def test_basic_offset_0(self):
        arr = _make_hourly_array(700)
        results = create_sliding_daily_from_hourly(arr, offsets=[0], hours_per_day=7)
        assert len(results) == 1
        expected_bars = 700 // 7  # 100
        assert results[0].shape == (4, expected_bars)

    def test_multiple_offsets(self):
        arr = _make_hourly_array(700)
        results = create_sliding_daily_from_hourly(arr, offsets=[0, 1, 2, 3, 4, 5, 6], hours_per_day=7)
        assert len(results) == 7  # one per offset
        # All results should have the same length (different offsets lose different amounts)
        for r in results:
            assert r.shape[0] == 4
            assert r.shape[1] >= 30

    def test_ohlc_aggregation_correctness(self):
        # Build a simple synthetic hourly array with known values
        # 2 days × 7 hours = 14 bars
        hours = 14
        open_  = np.arange(hours, dtype=np.float32) * 10 + 100
        high_  = open_ + 5
        low_   = open_ - 5
        close_ = open_ + 2
        arr = np.stack([open_, high_, low_, close_])

        results = create_sliding_daily_from_hourly(arr, offsets=[0], hours_per_day=7)
        assert len(results) == 0  # only 2 days → below min 30; test with smaller min

    def test_ohlc_aggregation_values(self):
        # Use a small example with predictable values
        hours = 7
        open_  = np.array([100.0] * hours, dtype=np.float32)
        high_  = np.array([100.0, 102.0, 103.0, 101.0, 99.0, 105.0, 100.0], dtype=np.float32)
        low_   = np.array([98.0, 99.0, 97.0, 100.0, 96.0, 100.0, 99.0], dtype=np.float32)
        close_ = np.array([99.0, 101.0, 102.0, 100.0, 98.0, 104.0, 101.0], dtype=np.float32)
        arr = np.stack([open_, high_, low_, close_])

        # Manually replicate to get enough bars
        arr_big = np.concatenate([arr] * 40, axis=1)  # 280 hours → 40 daily bars
        results = create_sliding_daily_from_hourly(arr_big, offsets=[0], hours_per_day=7)
        assert len(results) == 1
        daily = results[0]
        # Check first aggregated bar
        assert daily[0, 0] == pytest.approx(100.0)  # open = first hour open
        assert daily[1, 0] == pytest.approx(105.0)  # high = max of highs
        assert daily[2, 0] == pytest.approx(96.0)   # low = min of lows
        assert daily[3, 0] == pytest.approx(101.0)  # close = last hour close

    def test_wrong_shape_raises(self):
        arr = np.zeros((3, 100))  # wrong: not 4 channels
        with pytest.raises((AssertionError, ValueError)):
            create_sliding_daily_from_hourly(arr, offsets=[0])

    def test_offset_excludes_short(self):
        # With offset=6, hours_per_day=7, T=40: only 4 full bars starting at pos 6
        # (pos 6, 13, 20, 27 → next would be 34 which needs 34+7=41>40, excluded)
        arr = _make_hourly_array(40)
        results = create_sliding_daily_from_hourly(arr, offsets=[6], hours_per_day=7)
        # 4 bars < 30 min bars → should be excluded
        assert len(results) == 0


# ---------------------------------------------------------------------------
# to_return_series
# ---------------------------------------------------------------------------

class TestToReturnSeries:
    def test_basic_return(self):
        # Need at least 31 bars so ret has >= 30 timesteps after dropping first
        n = 32
        prices = np.ones(n, dtype=np.float32) * 100.0
        prices[1] = 110.0  # single step up at index 1
        arr_4 = np.tile(prices, (4, 1))  # (4, n)
        ret = to_return_series(arr_4)
        assert ret is not None
        # close channel (index 3): at t=0 → (110 - 100) / (100 + eps) ≈ 0.1
        expected = (110.0 - 100.0) / (abs(100.0) + 1e-8)
        assert ret[3, 0] == pytest.approx(expected, rel=1e-4)

    def test_shape_output(self):
        arr = _make_hourly_array(200)
        ret = to_return_series(arr)
        assert ret is not None
        assert ret.shape == (4, 199)  # T-1

    def test_too_short(self):
        arr = np.ones((4, 5), dtype=np.float32)
        assert to_return_series(arr) is None  # 4 bars after drop → too short

    def test_float32_output(self):
        arr = _make_hourly_array(100)
        ret = to_return_series(arr)
        assert ret is not None
        assert ret.dtype == np.float32


# ---------------------------------------------------------------------------
# prepare_all_training_series
# ---------------------------------------------------------------------------

class TestPrepareAllTrainingSeries:
    def test_daily_only(self, tmp_path):
        for sym in ("A", "B", "C"):
            _make_csv(tmp_path / f"{sym}.csv", 100)
        series = prepare_all_training_series(
            daily_data_dir=tmp_path,
            aug_config=AugConfig(add_return_variants=False, sliding_daily_offsets=[]),
        )
        assert len(series) == 3

    def test_return_variants_added(self, tmp_path):
        _make_csv(tmp_path / "A.csv", 100)
        series = prepare_all_training_series(
            daily_data_dir=tmp_path,
            aug_config=AugConfig(add_return_variants=True, sliding_daily_offsets=[]),
        )
        # original + return variant = 2
        assert len(series) == 2

    def test_no_data_returns_empty(self, tmp_path):
        series = prepare_all_training_series(
            daily_data_dir=tmp_path,  # empty dir
            aug_config=AugConfig(add_return_variants=False, sliding_daily_offsets=[]),
        )
        assert series == []

    def test_sliding_creates_more_series(self, tmp_path):
        hourly_dir = tmp_path / "hourly"
        hourly_dir.mkdir()
        # 700 hours → 100 daily bars per offset → valid for min 30
        _make_csv(hourly_dir / "BTC.csv", 700)
        offsets = [0, 1, 2, 3, 4, 5, 6]
        series = prepare_all_training_series(
            hourly_data_dirs=[hourly_dir],
            aug_config=AugConfig(
                add_return_variants=False,
                sliding_daily_offsets=offsets,
                hours_per_day=7,
                min_length=50,
            ),
        )
        # raw hourly + 7 sliding aggs = 8 minimum (exact count depends on hourly length)
        assert len(series) >= 8


# ---------------------------------------------------------------------------
# split_series_list
# ---------------------------------------------------------------------------

class TestSplitSeriesList:
    def test_basic_split(self):
        arr = np.ones((4, 200), dtype=np.float32)
        series = [{"target": arr}]
        train, val, test = split_series_list(series, val_bars=30, test_bars=30, min_train_bars=60)
        assert len(train) == 1
        assert len(val)   == 1
        assert len(test)  == 1
        assert train[0]["target"].shape[1] == 140
        assert val[0]["target"].shape[1]   == 30
        assert test[0]["target"].shape[1]  == 30

    def test_short_series_goes_to_train_only(self):
        arr = np.ones((4, 50), dtype=np.float32)  # too short for val/test
        series = [{"target": arr}]
        train, val, test = split_series_list(series, val_bars=30, test_bars=30, min_train_bars=60)
        assert len(train) == 1
        assert len(val) == 0
        assert len(test) == 0

    def test_no_data_leakage(self):
        """Val must come strictly after train."""
        arr = np.arange(200 * 4, dtype=np.float32).reshape(4, 200)
        series = [{"target": arr}]
        train, val, _test = split_series_list(series, val_bars=40, test_bars=40, min_train_bars=60)
        t_end = train[0]["target"][0, -1]
        v_start = val[0]["target"][0, 0]
        assert v_start > t_end  # time is monotonically increasing here

    def test_multiple_series(self):
        series = [{"target": np.ones((4, 200))} for _ in range(10)]
        train, val, test = split_series_list(series, val_bars=30, test_bars=30)
        assert len(train) == 10
        assert len(val) == 10
        assert len(test) == 10


# ---------------------------------------------------------------------------
# AugmentedChronos2Dataset (only runs when chronos is installed)
# ---------------------------------------------------------------------------

try:
    from chronos.chronos2.dataset import DatasetMode
    _CHRONOS_OK = True
except ImportError:
    _CHRONOS_OK = False


@pytest.mark.skipif(not _CHRONOS_OK, reason="chronos-forecasting not installed")
class TestAugmentedChronos2Dataset:
    def _make_inputs(self, n_series: int = 5, T: int = 200):
        rng = np.random.default_rng(0)
        return [{"target": (100 + rng.standard_normal((4, T))).astype(np.float32)}
                for _ in range(n_series)]

    def test_aug_not_applied_in_val_mode(self):
        """In VAL mode, no augmentation should change the slices."""
        from chronos2_stock_augmentation import AugmentedChronos2Dataset
        inputs = self._make_inputs(5, 200)
        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=64,
            prediction_length=1,
            batch_size=4,
            output_patch_size=16,
            mode=DatasetMode.VALIDATION,
            aug_config=AugConfig(amplitude_log_std=2.0, noise_std_frac=0.5, time_dropout_rate=0.5),
        )
        # Get two batches and check they are deterministic (no randomness)
        batch1 = next(iter(ds))
        batch2 = next(iter(ds))
        # Same val slice → same context
        import torch
        assert torch.allclose(batch1["context"], batch2["context"])

    def test_aug_applied_in_train_mode(self):
        """In TRAIN mode, amplitude jitter should produce different scale across batches."""
        from chronos2_stock_augmentation import AugmentedChronos2Dataset
        import torch
        rng = np.random.default_rng(7)
        arr = (100 + np.cumsum(rng.standard_normal((4, 500)), axis=1)).astype(np.float32)
        inputs = [{"target": arr}]
        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=64,
            prediction_length=1,
            batch_size=4,
            output_patch_size=16,
            mode=DatasetMode.TRAIN,
            aug_config=AugConfig(amplitude_log_std=1.0, noise_std_frac=0.0, time_dropout_rate=0.0),
        )
        # Collect max absolute values across many batches; they should vary due to amplitude jitter
        max_vals = []
        it = iter(ds)
        for _ in range(20):
            batch = next(it)
            ctx = batch["context"]
            # Use nanmax to handle any NaN padding
            val = float(torch.nan_to_num(ctx.abs(), nan=0.0).max())
            max_vals.append(val)
        # With log_std=1.0, scales vary by > factor of 2; the spread should be large
        assert max(max_vals) / max(min(max_vals), 1e-6) > 1.5, (
            "Expected large variation in context max values due to amplitude jitter"
        )

    def test_time_dropout_injects_nans(self):
        """High dropout rate should produce NaNs in the context."""
        from chronos2_stock_augmentation import AugmentedChronos2Dataset
        inputs = self._make_inputs(10, 200)
        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=128,
            prediction_length=1,
            batch_size=4,
            output_patch_size=16,
            mode=DatasetMode.TRAIN,
            aug_config=AugConfig(amplitude_log_std=0.0, noise_std_frac=0.0, time_dropout_rate=0.5),
        )
        import torch
        found_nan = False
        for batch in ds:
            if torch.isnan(batch["context"]).any():
                found_nan = True
                break
        assert found_nan, "Expected NaNs in context from time dropout"

    def test_batch_shape(self):
        """Batch shapes should be consistent with configuration."""
        from chronos2_stock_augmentation import AugmentedChronos2Dataset
        inputs = self._make_inputs(20, 500)  # long enough that slices always reach ctx_len
        batch_size = 8
        ctx_len = 64
        pred_len = 1
        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=ctx_len,
            prediction_length=pred_len,
            batch_size=batch_size,
            output_patch_size=16,
            mode=DatasetMode.TRAIN,
            aug_config=AugConfig(amplitude_log_std=0.0, noise_std_frac=0.0, time_dropout_rate=0.0),
        )
        # Collect several batches; at least one should have full context_length
        it = iter(ds)
        max_ctx_len = 0
        for _ in range(20):
            batch = next(it)
            max_ctx_len = max(max_ctx_len, batch["context"].shape[-1])
            assert batch["future_target"].shape[-1] == pred_len
            # Context must not exceed context_length
            assert batch["context"].shape[-1] <= ctx_len

        # After 20 batches from 500-bar series, should have seen full-length contexts
        assert max_ctx_len == ctx_len, f"Expected at least one batch with context={ctx_len}, got {max_ctx_len}"

    def test_freq_subsample_reduces_context_length(self):
        """freq_subsample_prob=1.0 should halve context length via stride-2 averaging."""
        try:
            from chronos.chronos2.dataset import DatasetMode
        except ImportError:
            pytest.skip("chronos not installed")
        from chronos2_stock_augmentation import AugmentedChronos2Dataset
        import torch
        inputs = self._make_inputs(5, 500)
        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=128,
            prediction_length=1,
            batch_size=4,
            output_patch_size=16,
            mode=DatasetMode.TRAIN,
            aug_config=AugConfig(
                amplitude_log_std=0.0, noise_std_frac=0.0,
                time_dropout_rate=0.0, freq_subsample_prob=1.0,
            ),
        )
        it = iter(ds)
        batch = next(it)
        # With stride-2, context should be halved (~64 instead of 128)
        ctx_len = batch["context"].shape[-1]
        assert ctx_len <= 64, f"Expected stride-2 subsampled context <= 64, got {ctx_len}"

    def test_detrend_context_removes_trend(self):
        """detrend_context=True should produce roughly zero-mean context for linear trend inputs."""
        try:
            from chronos.chronos2.dataset import DatasetMode
        except ImportError:
            pytest.skip("chronos not installed")
        from chronos2_stock_augmentation import AugmentedChronos2Dataset
        import torch
        # Create strongly trending series
        rng = np.random.default_rng(42)
        T = 300
        inputs = []
        for _ in range(5):
            trend = np.linspace(100, 200, T).astype(np.float32)
            noise = rng.standard_normal((4, T)).astype(np.float32) * 0.01
            inputs.append({"target": trend[np.newaxis] + noise})

        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=64,
            prediction_length=1,
            batch_size=4,
            output_patch_size=16,
            mode=DatasetMode.TRAIN,
            aug_config=AugConfig(
                amplitude_log_std=0.0, noise_std_frac=0.0,
                time_dropout_rate=0.0, detrend_context=True,
            ),
        )
        it = iter(ds)
        # Check that detrended context is approximately zero-mean
        batch = next(it)
        ctx = batch["context"]
        ctx_finite = ctx[~torch.isnan(ctx)]
        if len(ctx_finite) > 10:
            mean_abs = float(ctx_finite.abs().mean())
            # After detrending a 100→200 trend, values should be much smaller
            assert mean_abs < 10.0, f"Expected detrended context to be close to 0, mean_abs={mean_abs:.2f}"

    def test_gap_inject_creates_level_shift(self):
        """Gap injection should produce a persistent level shift mid-context."""
        try:
            from chronos.chronos2.dataset import DatasetMode
            from chronos2_stock_augmentation import AugmentedChronos2Dataset
        except ImportError:
            pytest.skip("chronos not installed")
        import torch

        rng = np.random.default_rng(42)
        T = 200
        # Flat price series (constant ~100) so any shift is clearly visible
        inputs = []
        for _ in range(10):
            prices = np.full((4, T), 100.0, dtype=np.float32)
            inputs.append({"target": prices})

        # Use gap_inject_prob=1.0 so every sample gets a gap
        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=64,
            prediction_length=1,
            batch_size=4,
            output_patch_size=16,
            mode=DatasetMode.TRAIN,
            aug_config=AugConfig(
                amplitude_log_std=0.0, noise_std_frac=0.0,
                time_dropout_rate=0.0,
                gap_inject_prob=1.0, gap_magnitude_frac=0.10,
            ),
        )
        batch = next(iter(ds))
        ctx = batch["context"]  # (batch, channels, T) or similar
        # At least one sample should have non-constant context (gap changed values)
        if ctx.numel() > 0:
            flat_ctx = ctx.float().reshape(-1)
            finite = flat_ctx[~torch.isnan(flat_ctx)]
            # If gap was injected on a constant series, std should be > 0
            assert float(finite.std()) > 0.0, "Gap inject should produce non-constant context"

    def test_gap_inject_disabled_when_zero_prob(self):
        """Gap injection disabled (prob=0) should not change constant series."""
        try:
            from chronos.chronos2.dataset import DatasetMode
            from chronos2_stock_augmentation import AugmentedChronos2Dataset
        except ImportError:
            pytest.skip("chronos not installed")
        import torch

        T = 100
        inputs = [{"target": np.full((4, T), 100.0, dtype=np.float32)} for _ in range(5)]

        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=32,
            prediction_length=1,
            batch_size=4,
            output_patch_size=16,
            mode=DatasetMode.TRAIN,
            aug_config=AugConfig(
                amplitude_log_std=0.0, noise_std_frac=0.0,
                time_dropout_rate=0.0, gap_inject_prob=0.0,
            ),
        )
        batch = next(iter(ds))
        ctx = batch["context"].float()
        finite = ctx[~torch.isnan(ctx)]
        # No augmentation → all values should be ~100.0
        assert float(finite.std()) < 1e-3, "No gap inject should leave constant series unchanged"

    def test_gap_inject_not_applied_in_val_mode(self):
        """Gap injection should be skipped in validation mode."""
        try:
            from chronos.chronos2.dataset import DatasetMode
            from chronos2_stock_augmentation import AugmentedChronos2Dataset
        except ImportError:
            pytest.skip("chronos not installed")
        import torch

        T = 100
        inputs = [{"target": np.full((4, T), 100.0, dtype=np.float32)} for _ in range(5)]

        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=32,
            prediction_length=1,
            batch_size=4,
            output_patch_size=16,
            mode=DatasetMode.VALIDATION,
            aug_config=AugConfig(gap_inject_prob=1.0, gap_magnitude_frac=0.10),
        )
        batch = next(iter(ds))
        ctx = batch["context"].float()
        finite = ctx[~torch.isnan(ctx)]
        # Val mode: no augmentation → constant series
        assert float(finite.std()) < 1e-3, "Val mode should not apply gap inject"


    def test_trend_inject_creates_linear_drift(self):
        """Trend injection should add a monotone drift to the context."""
        try:
            from chronos.chronos2.dataset import DatasetMode
            from chronos2_stock_augmentation import AugmentedChronos2Dataset
        except ImportError:
            pytest.skip("chronos not installed")
        import torch

        T = 64
        inputs = [{"target": np.full((4, T + 1), 100.0, dtype=np.float32)} for _ in range(20)]

        # Prob=1.0 ensures trend injection fires every time
        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=T,
            prediction_length=1,
            batch_size=4,
            output_patch_size=16,
            mode=DatasetMode.TRAIN,
            aug_config=AugConfig(
                amplitude_log_std=0.0, noise_std_frac=0.0,
                time_dropout_rate=0.0, gap_inject_prob=0.0,
                trend_inject_prob=1.0, trend_magnitude_frac=0.10,
            ),
        )
        batch = next(iter(ds))
        ctx = batch["context"].float()
        # With trend injected onto a constant=100 series, first and last bar should differ
        finite = ctx[~torch.isnan(ctx)]
        # Standard deviation > 0 (not a flat line anymore)
        assert float(finite.std()) > 0.1, "Trend inject should add drift to constant series"

    def test_trend_inject_disabled_when_zero_prob(self):
        """No trend injection when prob=0."""
        try:
            from chronos.chronos2.dataset import DatasetMode
            from chronos2_stock_augmentation import AugmentedChronos2Dataset
        except ImportError:
            pytest.skip("chronos not installed")
        import torch

        T = 64
        inputs = [{"target": np.full((4, T + 1), 100.0, dtype=np.float32)} for _ in range(10)]

        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=T,
            prediction_length=1,
            batch_size=4,
            output_patch_size=16,
            mode=DatasetMode.TRAIN,
            aug_config=AugConfig(
                amplitude_log_std=0.0, noise_std_frac=0.0,
                time_dropout_rate=0.0, gap_inject_prob=0.0,
                trend_inject_prob=0.0,
            ),
        )
        batch = next(iter(ds))
        ctx = batch["context"].float()
        finite = ctx[~torch.isnan(ctx)]
        assert float(finite.std()) < 1e-3, "Zero prob should leave series unchanged"

    def test_vol_regime_changes_volatility(self):
        """Vol regime aug should produce different volatility in second half vs first half."""
        try:
            from chronos.chronos2.dataset import DatasetMode
            from chronos2_stock_augmentation import AugmentedChronos2Dataset
        except ImportError:
            pytest.skip("chronos not installed")
        import torch

        T = 64
        rng = np.random.default_rng(42)
        # Use a long series so context is always T bars
        T_series = T * 4  # 4 full non-overlapping windows
        inputs = [{"target": (100.0 + rng.normal(0, 1.0, (4, T_series))).astype(np.float32)}
                  for _ in range(8)]

        aug = AugConfig(
            amplitude_log_std=0.0, noise_std_frac=0.0, time_dropout_rate=0.0,
            gap_inject_prob=0.0, trend_inject_prob=0.0,
            vol_regime_prob=1.0, vol_regime_max_mult=4.0, mean_reversion_prob=0.0,
        )
        ds = AugmentedChronos2Dataset(
            inputs=inputs, context_length=T, prediction_length=1,
            batch_size=4, output_patch_size=16,
            mode=DatasetMode.TRAIN, aug_config=aug,
        )
        # Collect several long-enough batches
        ratios = []
        for i, batch in enumerate(ds):
            if i >= 10:
                break
            ctx = batch["context"].float()
            if ctx.shape[-1] < T:
                continue   # skip short contexts
            split = T // 2
            first_std = float(ctx[..., :split].reshape(-1).std())
            second_std = float(ctx[..., split:].reshape(-1).std())
            if first_std > 0 and not np.isnan(first_std) and not np.isnan(second_std):
                ratios.append(second_std / first_std)

        assert len(ratios) >= 1, "Need at least one full-length batch"
        # With vol_regime_max_mult=4, the ratio should deviate from 1 in at least some samples
        # (could be close to 1 if multiplier=1.0 was drawn, but mean should vary)
        assert not all(abs(r - 1.0) < 0.01 for r in ratios), \
            f"Vol regime should change relative volatility; got ratios={ratios}"

    def test_vol_regime_disabled_when_zero_prob(self):
        """No vol regime when prob=0 and constant input stays constant."""
        try:
            from chronos.chronos2.dataset import DatasetMode
            from chronos2_stock_augmentation import AugmentedChronos2Dataset
        except ImportError:
            pytest.skip("chronos not installed")
        import torch

        T = 64
        inputs = [{"target": np.full((4, T + 1), 50.0, dtype=np.float32)} for _ in range(10)]
        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=T,
            prediction_length=1,
            batch_size=4,
            output_patch_size=16,
            mode=DatasetMode.TRAIN,
            aug_config=AugConfig(
                amplitude_log_std=0.0, noise_std_frac=0.0, time_dropout_rate=0.0,
                gap_inject_prob=0.0, trend_inject_prob=0.0,
                vol_regime_prob=0.0, mean_reversion_prob=0.0,
            ),
        )
        batch = next(iter(ds))
        ctx = batch["context"].float()
        finite = ctx[~torch.isnan(ctx)]
        assert float(finite.std()) < 1e-3, "Zero vol_regime_prob should leave constant series unchanged"

    def test_mean_reversion_adds_oscillation(self):
        """Mean-reversion aug should produce oscillating pattern on constant series."""
        try:
            from chronos.chronos2.dataset import DatasetMode
            from chronos2_stock_augmentation import AugmentedChronos2Dataset
        except ImportError:
            pytest.skip("chronos not installed")
        import torch

        T = 64
        # Constant series — only mean_reversion can add oscillation
        inputs = [{"target": np.full((4, T + 1), 100.0, dtype=np.float32)} for _ in range(20)]

        aug = AugConfig(
            amplitude_log_std=0.0, noise_std_frac=0.0, time_dropout_rate=0.0,
            gap_inject_prob=0.0, trend_inject_prob=0.0, vol_regime_prob=0.0,
            mean_reversion_prob=1.0,   # always apply
            mean_reversion_amplitude=0.05,
        )
        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=T,
            prediction_length=1,
            batch_size=8,
            output_patch_size=16,
            mode=DatasetMode.TRAIN,
            aug_config=aug,
        )
        batch = next(iter(ds))
        ctx = batch["context"].float()  # (..., T)
        finite = ctx[~torch.isnan(ctx)]
        # With amplitude=0.05 and mean=100, std across time should be > 0.5
        assert float(finite.std()) > 0.5, \
            "Mean-reversion aug should add visible oscillation (std > 0.5)"

    def test_mean_reversion_not_in_val_mode(self):
        """Mean-reversion aug should NOT be applied in validation mode."""
        try:
            from chronos.chronos2.dataset import DatasetMode
            from chronos2_stock_augmentation import AugmentedChronos2Dataset
        except ImportError:
            pytest.skip("chronos not installed")
        import torch

        T = 64
        inputs = [{"target": np.full((4, T + 1), 100.0, dtype=np.float32)} for _ in range(10)]
        aug = AugConfig(
            amplitude_log_std=0.0, noise_std_frac=0.0, time_dropout_rate=0.0,
            mean_reversion_prob=1.0, mean_reversion_amplitude=0.05,
        )
        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=T,
            prediction_length=1,
            batch_size=4,
            output_patch_size=16,
            mode=DatasetMode.VALIDATION,   # val mode
            aug_config=aug,
        )
        batch = next(iter(ds))
        ctx = batch["context"].float()
        finite = ctx[~torch.isnan(ctx)]
        assert float(finite.std()) < 1e-3, "Val mode should not apply mean_reversion_aug"

    def test_earnings_shock_changes_context(self):
        """Earnings shock injection should change context values."""
        try:
            from chronos.chronos2.dataset import DatasetMode
            from chronos2_stock_augmentation import AugmentedChronos2Dataset
        except ImportError:
            pytest.skip("chronos not installed")
        import torch

        rng = np.random.default_rng(42)
        T = 128
        T_series = T * 4
        # Noisy series so shock effects are visible
        base = 100.0 + np.cumsum(rng.normal(0, 1.0, (4, T_series)), axis=1)
        inputs = [{"target": base.astype(np.float32)} for _ in range(20)]

        aug = AugConfig(
            amplitude_log_std=0.0, noise_std_frac=0.0, time_dropout_rate=0.0,
            earnings_shock_prob=1.0, earnings_shock_magnitude=0.15,
        )
        ds = AugmentedChronos2Dataset(
            inputs=inputs,
            context_length=T,
            prediction_length=1,
            batch_size=4,
            output_patch_size=16,
            mode=DatasetMode.TRAIN,
            aug_config=aug,
        )

        found_change = False
        baseline_aug = AugConfig(amplitude_log_std=0.0, noise_std_frac=0.0, time_dropout_rate=0.0)
        ds_noshock = AugmentedChronos2Dataset(
            inputs=inputs, context_length=T, prediction_length=1, batch_size=4,
            output_patch_size=16, mode=DatasetMode.TRAIN, aug_config=baseline_aug,
        )
        for _ in range(30):
            batch_shocked = next(iter(ds))
            batch_plain  = next(iter(ds_noshock))
            ctx_s = batch_shocked["context"].float()
            ctx_p = batch_plain["context"].float()
            if ctx_s.shape[-1] < T:
                continue
            # Check that the shocked context differs from the plain one
            if float((ctx_s - ctx_p).abs().max()) > 0.1:
                found_change = True
                break
        assert found_change, "earnings_shock_prob=1.0 should alter the context"

    def test_earnings_shock_disabled_when_zero_prob(self):
        """Earnings shock should not activate when prob=0."""
        try:
            from chronos.chronos2.dataset import DatasetMode
            from chronos2_stock_augmentation import AugmentedChronos2Dataset
        except ImportError:
            pytest.skip("chronos not installed")
        import torch

        T = 64
        # Constant series — any change would be visible
        inputs = [{"target": np.full((4, T + 1), 100.0, dtype=np.float32)} for _ in range(10)]
        aug = AugConfig(
            amplitude_log_std=0.0, noise_std_frac=0.0, time_dropout_rate=0.0,
            earnings_shock_prob=0.0,
        )
        ds = AugmentedChronos2Dataset(
            inputs=inputs, context_length=T, prediction_length=1, batch_size=4,
            output_patch_size=16, mode=DatasetMode.TRAIN, aug_config=aug,
        )
        for _ in range(10):
            batch = next(iter(ds))
            ctx = batch["context"].float()
            finite = ctx[~torch.isnan(ctx)]
            assert float(finite.std()) < 1e-3, "earnings_shock_prob=0 should not alter constant series"

    def test_earnings_shock_not_in_val_mode(self):
        """Earnings shock should NOT be applied in validation mode."""
        try:
            from chronos.chronos2.dataset import DatasetMode
            from chronos2_stock_augmentation import AugmentedChronos2Dataset
        except ImportError:
            pytest.skip("chronos not installed")
        import torch

        T = 64
        inputs = [{"target": np.full((4, T + 1), 100.0, dtype=np.float32)} for _ in range(10)]
        aug = AugConfig(
            amplitude_log_std=0.0, noise_std_frac=0.0, time_dropout_rate=0.0,
            earnings_shock_prob=1.0, earnings_shock_magnitude=0.15,
        )
        ds = AugmentedChronos2Dataset(
            inputs=inputs, context_length=T, prediction_length=1, batch_size=4,
            output_patch_size=16, mode=DatasetMode.VALIDATION, aug_config=aug,
        )
        batch = next(iter(ds))
        ctx = batch["context"].float()
        finite = ctx[~torch.isnan(ctx)]
        assert float(finite.std()) < 1e-3, "Val mode should not apply earnings_shock"
