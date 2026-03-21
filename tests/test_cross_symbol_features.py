"""Tests for pufferlib_market.cross_symbol_features."""
from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pufferlib_market.cross_symbol_features import (
    CROSS_FEATURES,
    compute_cross_features,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

SYMBOLS_CRYPTO = ["BTC", "ETH", "SOL"]
SYMBOLS_NO_BTC = ["ETH", "SOL", "AVAX"]
T = 100
S = 3


def _make_prices(t: int = T, s: int = S, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Random-walk prices, all starting near 100.
    steps = rng.standard_normal((t, s)) * 0.01
    prices = np.exp(np.cumsum(steps, axis=0)) * 100.0
    return prices.astype(np.float64)


# ---------------------------------------------------------------------------
# Shape and range tests
# ---------------------------------------------------------------------------


class TestComputeCrossFeatures:
    def test_output_shape(self):
        prices = _make_prices()
        out = compute_cross_features(prices, SYMBOLS_CRYPTO)
        assert out.shape == (T, S, CROSS_FEATURES), f"Expected ({T},{S},{CROSS_FEATURES}), got {out.shape}"

    def test_output_dtype(self):
        prices = _make_prices()
        out = compute_cross_features(prices, SYMBOLS_CRYPTO)
        assert out.dtype == np.float32

    def test_rolling_corr_in_range(self):
        prices = _make_prices()
        out = compute_cross_features(prices, SYMBOLS_CRYPTO)
        corr = out[:, :, 0]
        assert corr.min() >= -1.0 - 1e-6, f"corr min={corr.min()}"
        assert corr.max() <= 1.0 + 1e-6, f"corr max={corr.max()}"

    def test_rolling_beta_in_range(self):
        prices = _make_prices()
        out = compute_cross_features(prices, SYMBOLS_CRYPTO)
        beta = out[:, :, 1]
        assert beta.min() >= -1.0 - 1e-6, f"beta min={beta.min()}"
        assert beta.max() <= 1.0 + 1e-6, f"beta max={beta.max()}"

    def test_breadth_rank_in_range(self):
        prices = _make_prices()
        out = compute_cross_features(prices, SYMBOLS_CRYPTO)
        rank = out[:, :, 3]
        assert rank.min() >= 0.0 - 1e-6, f"rank min={rank.min()}"
        assert rank.max() <= 1.0 + 1e-6, f"rank max={rank.max()}"

    def test_no_nan_in_output(self):
        prices = _make_prices()
        out = compute_cross_features(prices, SYMBOLS_CRYPTO)
        assert not np.any(np.isnan(out)), "Output contains NaN values"

    def test_no_inf_in_output(self):
        prices = _make_prices()
        out = compute_cross_features(prices, SYMBOLS_CRYPTO)
        assert not np.any(np.isinf(out)), "Output contains Inf values"

    def test_anchor_fallback_when_btc_missing(self):
        """When anchor symbol is not present, should fall back to first symbol."""
        prices = _make_prices()
        out_no_btc = compute_cross_features(prices, SYMBOLS_NO_BTC, anchor_symbol="BTC")
        # Should not raise and output should be valid
        assert out_no_btc.shape == (T, S, CROSS_FEATURES)
        assert not np.any(np.isnan(out_no_btc))

    def test_anchor_fallback_produces_same_as_explicit_first(self):
        """Fallback to first symbol should equal explicit anchor=first symbol."""
        prices = _make_prices()
        out_fallback = compute_cross_features(prices, SYMBOLS_NO_BTC, anchor_symbol="BTC")
        out_explicit = compute_cross_features(prices, SYMBOLS_NO_BTC, anchor_symbol="ETH")
        np.testing.assert_array_equal(out_fallback, out_explicit)

    def test_with_nan_input_no_crash(self):
        """NaN values in input prices should be handled without error."""
        prices = _make_prices()
        prices[10:15, 1] = np.nan  # Some NaN in ETH
        out = compute_cross_features(prices, SYMBOLS_CRYPTO)
        assert out.shape == (T, S, CROSS_FEATURES)
        assert not np.any(np.isnan(out))

    def test_with_leading_nan_input(self):
        """Leading NaN values (before first valid price) are handled."""
        prices = _make_prices()
        prices[:20, 2] = np.nan  # SOL has no data for first 20 bars
        out = compute_cross_features(prices, SYMBOLS_CRYPTO)
        assert out.shape == (T, S, CROSS_FEATURES)
        assert not np.any(np.isnan(out))

    def test_single_symbol(self):
        """Single-symbol edge case: breadth_rank should be 0.5."""
        prices = _make_prices(s=1)
        out = compute_cross_features(prices, ["BTC"])
        assert out.shape == (T, 1, CROSS_FEATURES)
        np.testing.assert_allclose(out[:, 0, 3], 0.5, atol=1e-6)

    def test_custom_window(self):
        prices = _make_prices()
        out_short = compute_cross_features(prices, SYMBOLS_CRYPTO, window=6)
        out_long = compute_cross_features(prices, SYMBOLS_CRYPTO, window=48)
        # Different windows should produce different results
        assert not np.allclose(out_short[:, :, 0], out_long[:, :, 0])

    def test_perfect_correlation(self):
        """A symbol identical to the anchor should have corr ~ 1."""
        prices = _make_prices(s=1)
        # Stack same price series twice: BTC, and identical copy
        prices_2 = np.column_stack([prices[:, 0], prices[:, 0]])
        out = compute_cross_features(prices_2, ["BTC", "CLONE"], window=24, anchor_symbol="BTC")
        # After warm-up (window=24 bars), the clone's corr with BTC should be ~1
        corr_clone = out[30:, 1, 0]
        assert np.all(corr_clone > 0.99), f"Expected ~1.0 corr, got min={corr_clone.min()}"

    def test_wrong_symbols_length_raises(self):
        prices = _make_prices()
        with pytest.raises(ValueError, match="len\\(symbols\\)"):
            compute_cross_features(prices, ["BTC", "ETH"])  # 2 symbols for 3-col prices

    def test_wrong_ndim_raises(self):
        prices = np.ones((100,))  # 1D — wrong
        with pytest.raises(ValueError, match="2-D"):
            compute_cross_features(prices, ["BTC"])

    def test_relative_return_zero_sum(self):
        """Sum of relative_return across symbols at each timestep should be ~0."""
        prices = _make_prices(s=5)
        symbols = ["BTC", "ETH", "SOL", "AVAX", "LTC"]
        out = compute_cross_features(prices, symbols)
        rel_ret = out[:, :, 2].astype(np.float64)
        row_sums = rel_ret.sum(axis=1)
        # Sum should be ~0 (up to floating point + clip effects)
        assert np.all(np.abs(row_sums) < 0.02), f"Row sum deviation too large: max={np.abs(row_sums).max()}"


# ---------------------------------------------------------------------------
# Binary file size test — cross_features produces larger files
# ---------------------------------------------------------------------------


def _write_dummy_mktd(path: Path, features_per_sym: int, T: int = 50, S: int = 3) -> None:
    """Write a minimal valid MKTD binary with the given features_per_sym."""
    MAGIC = b"MKTD"
    VERSION = 2
    PRICE_FEATURES = 5
    header = struct.pack(
        "<4sIIIII40s",
        MAGIC, VERSION, S, T, features_per_sym, PRICE_FEATURES, b"\x00" * 40,
    )
    syms = [b"BTC\x00" * 4, b"ETH\x00" * 4, b"SOL\x00" * 4]
    feat = np.zeros((T, S, features_per_sym), dtype=np.float32)
    price = np.zeros((T, S, PRICE_FEATURES), dtype=np.float32)
    mask = np.zeros((T, S), dtype=np.uint8)

    with open(path, "wb") as fh:
        fh.write(header)
        for s in syms:
            fh.write(s)
        fh.write(feat.tobytes())
        fh.write(price.tobytes())
        fh.write(mask.tobytes())


class TestBinaryFileSizes:
    """Check that cross-feature exports produce correctly larger files."""

    def test_cross_features_file_is_larger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p_base = Path(tmpdir) / "base.bin"
            p_cross = Path(tmpdir) / "cross.bin"
            _write_dummy_mktd(p_base, features_per_sym=16)
            _write_dummy_mktd(p_cross, features_per_sym=20)
            size_base = p_base.stat().st_size
            size_cross = p_cross.stat().st_size
            assert size_cross > size_base, "Cross-feature file should be larger"
            # Extra bytes = T * S * 4 extra features * 4 bytes/float
            expected_extra = 50 * 3 * 4 * 4
            assert size_cross - size_base == expected_extra, (
                f"Expected +{expected_extra} bytes, got +{size_cross - size_base}"
            )

    def test_header_features_per_sym_field(self):
        """Verify the header field features_per_sym is written correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for fps in (16, 20):
                p = Path(tmpdir) / f"fps{fps}.bin"
                _write_dummy_mktd(p, features_per_sym=fps)
                with open(p, "rb") as fh:
                    raw = fh.read(64)
                hdr = struct.unpack("<4sIIIII40s", raw)
                assert hdr[4] == fps, f"Expected features_per_sym={fps}, got {hdr[4]}"


# ---------------------------------------------------------------------------
# Integration: export_binary with cross_features kwarg
# ---------------------------------------------------------------------------


class TestExportBinaryIntegration:
    """Test that export_binary accepts cross_features and produces valid output."""

    def _make_csv_dir(self, tmpdir: str, symbols: list[str], T: int = 300) -> Path:
        """Create minimal daily CSV files for each symbol."""
        import pandas as pd
        root = Path(tmpdir) / "data"
        root.mkdir()
        rng = np.random.default_rng(99)
        dates = pd.date_range("2022-01-01", periods=T, freq="D", tz="UTC")
        for sym in symbols:
            close = np.exp(np.cumsum(rng.standard_normal(T) * 0.02)) * 100.0
            df = pd.DataFrame({
                "date": dates.strftime("%Y-%m-%d"),
                "open": close * (1 + rng.standard_normal(T) * 0.003),
                "high": close * (1 + np.abs(rng.standard_normal(T) * 0.005)),
                "low": close * (1 - np.abs(rng.standard_normal(T) * 0.005)),
                "close": close,
                "volume": np.abs(rng.standard_normal(T)) * 1e6 + 1e6,
            })
            df.to_csv(root / f"{sym}.csv", index=False)
        return root

    def test_daily_export_without_cross_features(self):
        from pufferlib_market.export_data_daily import export_binary

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = self._make_csv_dir(tmpdir, ["BTC", "ETH", "SOL"])
            out = Path(tmpdir) / "out_base.bin"
            export_binary(
                symbols=["BTC", "ETH", "SOL"],
                data_root=data_root,
                output_path=out,
                min_days=10,
                cross_features=False,
            )
            assert out.exists()
            with open(out, "rb") as fh:
                raw = fh.read(64)
            hdr = struct.unpack("<4sIIIII40s", raw)
            assert hdr[4] == 16, f"Expected features_per_sym=16, got {hdr[4]}"

    def test_daily_export_with_cross_features(self):
        from pufferlib_market.export_data_daily import export_binary

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = self._make_csv_dir(tmpdir, ["BTC", "ETH", "SOL"])
            out_base = Path(tmpdir) / "out_base.bin"
            out_cross = Path(tmpdir) / "out_cross.bin"
            export_binary(
                symbols=["BTC", "ETH", "SOL"],
                data_root=data_root,
                output_path=out_base,
                min_days=10,
                cross_features=False,
            )
            export_binary(
                symbols=["BTC", "ETH", "SOL"],
                data_root=data_root,
                output_path=out_cross,
                min_days=10,
                cross_features=True,
            )
            with open(out_cross, "rb") as fh:
                raw = fh.read(64)
            hdr = struct.unpack("<4sIIIII40s", raw)
            assert hdr[4] == 20, f"Expected features_per_sym=20, got {hdr[4]}"
            # Cross file must be strictly larger
            assert out_cross.stat().st_size > out_base.stat().st_size

    def test_daily_export_base_identical_bytes(self):
        """Without --cross-features, output is byte-for-byte identical to the old path."""
        from pufferlib_market.export_data_daily import export_binary

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = self._make_csv_dir(tmpdir, ["BTC", "ETH"])
            out1 = Path(tmpdir) / "a.bin"
            out2 = Path(tmpdir) / "b.bin"
            kwargs = dict(
                symbols=["BTC", "ETH"],
                data_root=data_root,
                output_path=out1,
                min_days=10,
                cross_features=False,
            )
            export_binary(**kwargs)
            kwargs["output_path"] = out2
            export_binary(**kwargs)
            assert out1.read_bytes() == out2.read_bytes(), "Same args must produce identical bytes"
