"""Tests for LogDiffAugmentation and DiffNormAugmentation."""
import numpy as np
import pandas as pd
import pytest

from preaug_sweeps.augmentations.strategies import (
    LogDiffAugmentation,
    DiffNormAugmentation,
    DifferencingAugmentation,
    AUGMENTATION_REGISTRY,
)
from preaug.strategies import (
    LogDiffAugmentation as CoreLogDiff,
    DiffNormAugmentation as CoreDiffNorm,
    AUGMENTATION_REGISTRY as CORE_REGISTRY,
)


PRICE_COLS = ["open", "high", "low", "close"]


def make_ohlcv(n=120, seed=42, vol=0.015):
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0, vol, n))
    df = pd.DataFrame({
        "open":   prices * (1 + rng.uniform(-0.003, 0.003, n)),
        "high":   prices * (1 + rng.uniform(0,     0.008, n)),
        "low":    prices * (1 - rng.uniform(0,     0.008, n)),
        "close":  prices,
        "volume": rng.uniform(1e6, 1e7, n),
        "amount": rng.uniform(1e8, 1e9, n),
    })
    return df, prices


class TestLogDiffAugmentation:
    def test_round_trip_exact(self):
        """log_diff inverse should reconstruct exactly from ground-truth log-diffs."""
        df, prices = make_ohlcv(120)
        context = df.iloc[:80]
        future = prices[80:]

        aug = LogDiffAugmentation()
        aug.transform_dataframe(context)

        # Ground-truth log-diffs for close
        log_diffs = np.log(prices[80:] / prices[79:-1])
        pred_input = np.column_stack([log_diffs] * 4)
        recovered = aug.inverse_transform_predictions(pred_input, context, columns=PRICE_COLS)
        close_idx = PRICE_COLS.index("close")
        error = np.abs(recovered[:, close_idx] - future) / future
        assert error.mean() < 1e-10, f"LogDiff round-trip error too large: {error.mean():.2e}"

    def test_transform_shape(self):
        df, _ = make_ohlcv(60)
        aug = LogDiffAugmentation()
        out = aug.transform_dataframe(df)
        assert out.shape == df.shape
        assert list(out.columns) == list(df.columns)

    def test_first_row_zero(self):
        """First row should be 0 (no previous bar)."""
        df, _ = make_ohlcv(60)
        aug = LogDiffAugmentation()
        out = aug.transform_dataframe(df)
        for col in PRICE_COLS:
            assert out[col].iloc[0] == pytest.approx(0.0, abs=1e-10), f"{col} first row not zero"

    def test_registered_in_registry(self):
        assert "log_diff" in AUGMENTATION_REGISTRY
        assert "log_diff" in CORE_REGISTRY

    def test_name(self):
        assert LogDiffAugmentation().name() == "log_diff"

    def test_high_vol_round_trip(self):
        """Works for high-volatility (±5% daily) asset."""
        df, prices = make_ohlcv(120, vol=0.05)
        context = df.iloc[:80]
        future = prices[80:]
        aug = LogDiffAugmentation()
        aug.transform_dataframe(context)
        log_diffs = np.log(prices[80:] / prices[79:-1])
        pred_input = np.column_stack([log_diffs] * 4)
        recovered = aug.inverse_transform_predictions(pred_input, context, columns=PRICE_COLS)
        error = np.abs(recovered[:, 3] - future) / future
        assert error.mean() < 1e-9


class TestDiffNormAugmentation:
    def test_round_trip_close(self):
        """inverse should reconstruct prices within a small error due to scale approximation."""
        df, prices = make_ohlcv(120)
        context = df.iloc[:80]
        future = prices[80:]

        aug = DiffNormAugmentation(window=20)
        aug.transform_dataframe(context)

        # Use last scale from training to reconstruct
        scale = aug.metadata["close_last_scale"]
        raw_diffs = prices[80:] - np.concatenate([[prices[79]], prices[80:-1]])
        normed = raw_diffs / scale
        pred_input = np.column_stack([normed] * 4)
        recovered = aug.inverse_transform_predictions(pred_input, context, columns=PRICE_COLS)
        close_idx = PRICE_COLS.index("close")
        error = np.abs(recovered[:, close_idx] - future) / future
        # Acceptable error since scale is approximated with training window std
        assert error.mean() < 0.05, f"DiffNorm error too large: {error.mean():.4f}"

    def test_transform_shape(self):
        df, _ = make_ohlcv(60)
        aug = DiffNormAugmentation(window=20)
        out = aug.transform_dataframe(df)
        assert out.shape == df.shape

    def test_normalized_std_approx_one(self):
        """Normalized diffs should have std close to 1 for stationary series."""
        df, _ = make_ohlcv(500, vol=0.01)
        aug = DiffNormAugmentation(window=20)
        out = aug.transform_dataframe(df)
        # After warmup (first 20 bars), std of normalized diffs should be near 1
        std = out["close"].iloc[20:].std()
        assert 0.8 < std < 1.3, f"Expected std ~1 but got {std:.3f}"

    def test_registered(self):
        assert "diff_norm" in AUGMENTATION_REGISTRY
        assert "diff_norm" in CORE_REGISTRY

    def test_name(self):
        assert DiffNormAugmentation().name() == "diff_norm_w20"
        assert DiffNormAugmentation(window=10).name() == "diff_norm_w10"

    def test_window_parameter(self):
        df, _ = make_ohlcv(60)
        aug10 = DiffNormAugmentation(window=10)
        aug30 = DiffNormAugmentation(window=30)
        out10 = aug10.transform_dataframe(df)
        out30 = aug30.transform_dataframe(df)
        # Different windows should produce different results
        assert not np.allclose(out10["close"].values, out30["close"].values)

    def test_constant_series_no_nan(self):
        """Constant price series should not produce NaN (zero std handled)."""
        df = pd.DataFrame({
            "open": [100.0] * 50,
            "high": [101.0] * 50,
            "low":  [99.0]  * 50,
            "close":[100.0] * 50,
            "volume": [1e6]  * 50,
            "amount": [1e8]  * 50,
        })
        aug = DiffNormAugmentation(window=20)
        out = aug.transform_dataframe(df)
        assert not out.isnull().any().any(), "NaN found in constant-series output"


class TestCoreRegistry:
    """Verify core preaug/strategies.py exports match sweep registry."""

    def test_log_diff_core_identical_to_sweep(self):
        df, prices = make_ohlcv(100)
        from preaug_sweeps.augmentations.strategies import LogDiffAugmentation as SweepLD
        from preaug.strategies import LogDiffAugmentation as CoreLD
        s_aug = SweepLD()
        c_aug = CoreLD()
        s_out = s_aug.transform_dataframe(df.copy())
        c_out = c_aug.transform_dataframe(df.copy())
        for col in PRICE_COLS:
            np.testing.assert_allclose(s_out[col].values, c_out[col].values, rtol=1e-8,
                                       err_msg=f"{col} mismatch between sweep and core LogDiff")

    def test_diff_norm_core_identical_to_sweep(self):
        df, _ = make_ohlcv(100)
        from preaug_sweeps.augmentations.strategies import DiffNormAugmentation as SweepDN
        from preaug.strategies import DiffNormAugmentation as CoreDN
        s_aug = SweepDN(window=20)
        c_aug = CoreDN(window=20)
        s_out = s_aug.transform_dataframe(df.copy())
        c_out = c_aug.transform_dataframe(df.copy())
        for col in PRICE_COLS:
            np.testing.assert_allclose(s_out[col].values, c_out[col].values, rtol=1e-8,
                                       err_msg=f"{col} mismatch between sweep and core DiffNorm")
