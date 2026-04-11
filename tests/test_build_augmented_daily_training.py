"""Tests for build_augmented_daily_training._concat_binaries."""
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.build_augmented_daily_training import _concat_binaries


MAGIC = b"MKTD"
VERSION = 2
PRICE_FEATURES = 5
SYM_NAME_LEN = 16


def _make_mktd(path: Path, *, num_symbols: int, num_timesteps: int, features_per_sym: int, seed: int = 0):
    """Write a minimal valid MKTD binary for testing."""
    rng = np.random.default_rng(seed)
    header = struct.pack(
        "<4sIIIII40s",
        MAGIC, VERSION, num_symbols, num_timesteps, features_per_sym, PRICE_FEATURES, b"\x00" * 40,
    )
    sym_table = b"".join(
        f"SYM{i:02d}".encode().ljust(SYM_NAME_LEN, b"\x00") for i in range(num_symbols)
    )
    features = rng.uniform(-1, 1, (num_timesteps, num_symbols, features_per_sym)).astype(np.float32)
    prices = rng.uniform(10, 200, (num_timesteps, num_symbols, PRICE_FEATURES)).astype(np.float32)
    mask = np.ones((num_timesteps, num_symbols), dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(header)
        f.write(sym_table)
        f.write(features.tobytes(order="C"))
        f.write(prices.tobytes(order="C"))
        f.write(mask.tobytes(order="C"))
    return features, prices, mask


def _parse_mktd(path: Path):
    """Parse a MKTD binary and return (features, prices, mask)."""
    with open(path, "rb") as f:
        hdr = f.read(64)
        _, _, syms, ts, feats, price_feats = struct.unpack_from("<4sIIIII", hdr)
        f.read(syms * SYM_NAME_LEN)  # skip symbol table
        features = np.fromfile(f, dtype=np.float32, count=ts * syms * feats).reshape(ts, syms, feats)
        prices = np.fromfile(f, dtype=np.float32, count=ts * syms * price_feats).reshape(ts, syms, price_feats)
        mask = np.frombuffer(f.read(), dtype=np.uint8).reshape(ts, syms)
    return features, prices, mask


def test_concat_two_files_correct_size():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        f1, p1, m1 = _make_mktd(d / "a.bin", num_symbols=3, num_timesteps=10, features_per_sym=16, seed=1)
        f2, p2, m2 = _make_mktd(d / "b.bin", num_symbols=3, num_timesteps=5, features_per_sym=16, seed=2)
        out = d / "out.bin"
        _concat_binaries([d / "a.bin", d / "b.bin"], out)

        # Parse output
        feats, prices, mask = _parse_mktd(out)
        assert feats.shape == (15, 3, 16), f"unexpected shape: {feats.shape}"
        assert prices.shape == (15, 3, 5)
        assert mask.shape == (15, 3)

        # Feature values should match originals exactly
        np.testing.assert_array_equal(feats[:10], f1)
        np.testing.assert_array_equal(feats[10:], f2)
        np.testing.assert_array_equal(prices[:10], p1)
        np.testing.assert_array_equal(prices[10:], p2)
        np.testing.assert_array_equal(mask[:10], m1)
        np.testing.assert_array_equal(mask[10:], m2)


def test_concat_updates_header_timestep_count():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _make_mktd(d / "a.bin", num_symbols=2, num_timesteps=7, features_per_sym=16, seed=3)
        _make_mktd(d / "b.bin", num_symbols=2, num_timesteps=4, features_per_sym=16, seed=4)
        out = d / "out.bin"
        _concat_binaries([d / "a.bin", d / "b.bin"], out)

        with open(out, "rb") as f:
            hdr = f.read(64)
        _, _, syms, ts, feats, _ = struct.unpack_from("<4sIIIII", hdr)
        assert syms == 2
        assert ts == 11
        assert feats == 16


def test_concat_file_size_matches_expected():
    """File size must equal header(64) + sym_table(S*16) + features + prices + mask."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        S, T1, T2, F = 4, 20, 15, 16
        _make_mktd(d / "a.bin", num_symbols=S, num_timesteps=T1, features_per_sym=F, seed=5)
        _make_mktd(d / "b.bin", num_symbols=S, num_timesteps=T2, features_per_sym=F, seed=6)
        out = d / "out.bin"
        _concat_binaries([d / "a.bin", d / "b.bin"], out)

        T = T1 + T2
        expected_size = 64 + S * 16 + T * S * F * 4 + T * S * PRICE_FEATURES * 4 + T * S
        assert out.stat().st_size == expected_size, (
            f"size mismatch: got {out.stat().st_size}, expected {expected_size}"
        )


def test_concat_single_file_passthrough():
    """Concatenating one file should produce an identical data section."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        f1, p1, m1 = _make_mktd(d / "a.bin", num_symbols=5, num_timesteps=30, features_per_sym=16, seed=7)
        out = d / "out.bin"
        _concat_binaries([d / "a.bin"], out)
        feats, prices, mask = _parse_mktd(out)
        np.testing.assert_array_equal(feats, f1)
        np.testing.assert_array_equal(prices, p1)
        np.testing.assert_array_equal(mask, m1)


def test_no_nan_in_augmented_bins():
    """The actual regenerated augmented training bins must be NaN/inf-free."""
    repo = Path(__file__).resolve().parents[1]
    for name in ["stocks17_augmented_train.bin", "stocks17_augmented_val.bin"]:
        p = repo / "pufferlib_market" / "data" / name
        if not p.exists():
            pytest.skip(f"{name} not found (run build_augmented_daily_training.py first)")
        feats, prices, mask = _parse_mktd(p)
        assert not np.isnan(feats).any(), f"NaN in {name} features"
        assert not np.isinf(feats).any(), f"Inf in {name} features"
        assert not np.isnan(prices).any(), f"NaN in {name} prices"
        assert not np.isinf(prices).any(), f"Inf in {name} prices"
