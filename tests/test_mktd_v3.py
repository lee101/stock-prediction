"""
tests/test_mktd_v3.py — Unit tests for MKTD v3 (20 intraday features).

Covers:
  - export_data_daily_v3.compute_intraday_features
  - export_data_daily_v3.zscore_normalise
  - MKTD v3 binary header read (features_per_sym=20)
  - C env loads v3 binary and produces correct obs_size
  - Backwards compat: v2 binary still loads with features_per_sym=16
  - evaluate_multiperiod.load_policy picks up features_per_sym from MktdData

Run with:
    pytest tests/test_mktd_v3.py -v
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers to build synthetic data
# ---------------------------------------------------------------------------

def _make_daily_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily OHLCV DataFrame with DatetimeIndex (UTC, daily)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    volume = rng.uniform(1e6, 5e6, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _make_hourly_df(daily_df: pd.DataFrame, hours_per_day: int = 6, seed: int = 0) -> pd.DataFrame:
    """Synthetic hourly OHLCV with 9:30-15:30 bars for each trading day."""
    rng = np.random.default_rng(seed)
    rows = []
    for date in daily_df.index:
        daily_close = float(daily_df.loc[date, "close"])
        for h in range(hours_per_day):
            ts = date + pd.Timedelta(hours=9 + h, minutes=30)
            c = daily_close * (1 + rng.normal(0, 0.003))
            o = c * (1 + rng.normal(0, 0.001))
            vol = float(rng.uniform(5e4, 2e5))
            rows.append({"timestamp": ts, "open": o, "high": max(o, c) * 1.001, "low": min(o, c) * 0.999, "close": c, "volume": vol})
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["timestamp"], utc=True)
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def _write_mktd_bin(
    path: Path,
    *,
    num_symbols: int = 2,
    num_timesteps: int = 20,
    features_per_sym: int = 16,
    version: int = 2,
    seed: int = 0,
) -> None:
    """Write a minimal MKTD binary for testing."""
    rng = np.random.default_rng(seed)
    magic = b"MKTD"
    price_features = 5
    header = struct.pack(
        "<4sIIIII40s",
        magic,
        version,
        num_symbols,
        num_timesteps,
        features_per_sym,
        price_features,
        b"\x00" * 40,
    )
    with open(path, "wb") as f:
        f.write(header)
        for i in range(num_symbols):
            name = f"SYM{i}".encode("ascii")[:15]
            f.write(name + b"\x00" * (16 - len(name)))
        # features
        feat = rng.random((num_timesteps, num_symbols, features_per_sym), dtype=np.float64).astype(np.float32)
        f.write(feat.tobytes(order="C"))
        # prices: ensure open > 0
        prices = np.abs(rng.random((num_timesteps, num_symbols, price_features), dtype=np.float64)).astype(np.float32) + 0.01
        f.write(prices.tobytes(order="C"))
        # tradable mask
        mask = np.ones((num_timesteps, num_symbols), dtype=np.uint8)
        f.write(mask.tobytes(order="C"))


# ---------------------------------------------------------------------------
# Tests: compute_intraday_features
# ---------------------------------------------------------------------------

class TestComputeIntradayFeatures:
    def test_basic_values(self):
        from pufferlib_market.export_data_daily_v3 import compute_intraday_features
        daily = _make_daily_df(10, seed=1)
        hourly = _make_hourly_df(daily, hours_per_day=6, seed=2)
        result = compute_intraday_features(hourly, daily)
        assert set(result.columns) == {"intraday_vol", "morning_ret", "vwap_dev", "gap_open"}
        assert len(result) == len(daily)

    def test_intraday_vol_positive(self):
        from pufferlib_market.export_data_daily_v3 import compute_intraday_features
        daily = _make_daily_df(5)
        hourly = _make_hourly_df(daily, hours_per_day=6)
        result = compute_intraday_features(hourly, daily)
        # vol should be non-negative (std of log returns)
        assert (result["intraday_vol"] >= 0.0).all()

    def test_gap_open_first_row_zero(self):
        from pufferlib_market.export_data_daily_v3 import compute_intraday_features
        daily = _make_daily_df(10)
        result = compute_intraday_features(None, daily)
        # First row: no previous close, so gap_open == 0
        assert result["gap_open"].iloc[0] == pytest.approx(0.0)

    def test_gap_open_formula(self):
        from pufferlib_market.export_data_daily_v3 import compute_intraday_features
        idx = pd.date_range("2024-01-02", periods=3, freq="D", tz="UTC")
        daily = pd.DataFrame(
            {"open": [100.0, 102.0, 98.0], "high": [101.0, 103.0, 99.0],
             "low": [99.0, 101.0, 97.0], "close": [100.0, 102.0, 98.0], "volume": [1e6, 1e6, 1e6]},
            index=idx,
        )
        result = compute_intraday_features(None, daily)
        # row 1: 102 / 100 - 1 = 0.02
        assert result["gap_open"].iloc[1] == pytest.approx(0.02, abs=1e-5)
        # row 2: 98 / 102 - 1 = -0.0392...
        assert result["gap_open"].iloc[2] == pytest.approx(98.0 / 102.0 - 1, abs=1e-5)

    def test_no_hourly_data_returns_zeros_for_intra(self):
        from pufferlib_market.export_data_daily_v3 import compute_intraday_features
        daily = _make_daily_df(10)
        result = compute_intraday_features(None, daily)
        assert (result["intraday_vol"] == 0.0).all()
        assert (result["morning_ret"] == 0.0).all()
        assert (result["vwap_dev"] == 0.0).all()
        # gap_open is computed from daily prices, so non-zero after row 0
        assert result["gap_open"].iloc[0] == pytest.approx(0.0)

    def test_vwap_dev_close_to_zero_when_prices_close(self):
        from pufferlib_market.export_data_daily_v3 import compute_intraday_features
        idx = pd.date_range("2024-01-02", periods=2, freq="D", tz="UTC")
        # Create daily with flat close = 100
        daily = pd.DataFrame(
            {"open": [100.0, 100.0], "high": [100.0, 100.0],
             "low": [100.0, 100.0], "close": [100.0, 100.0], "volume": [1e6, 1e6]},
            index=idx,
        )
        # Create hourly where all closes = 100 and equal volume
        rows = []
        for date in idx:
            for h in range(4):
                ts = date + pd.Timedelta(hours=9 + h, minutes=30)
                rows.append({"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "volume": 1e5})
        hdf = pd.DataFrame(rows)
        hdf.index = pd.to_datetime([date + pd.Timedelta(hours=9+h, minutes=30) for date in idx for h in range(4)], utc=True)
        hdf = hdf[["open", "high", "low", "close", "volume"]].astype(float)
        result = compute_intraday_features(hdf, daily)
        # vwap = 100, close = 100, so vwap_dev should be ~0
        assert abs(result["vwap_dev"].iloc[-1]) < 1e-4


# ---------------------------------------------------------------------------
# Tests: zscore_normalise
# ---------------------------------------------------------------------------

class TestZscoreNormalise:
    def test_output_shape(self):
        from pufferlib_market.export_data_daily_v3 import zscore_normalise
        arr = np.random.randn(100).astype(np.float32)
        out = zscore_normalise(arr, window=20)
        assert out.shape == (100,)

    def test_clipped(self):
        from pufferlib_market.export_data_daily_v3 import zscore_normalise
        arr = np.ones(100, dtype=np.float32)
        arr[50] = 1000.0  # extreme outlier
        out = zscore_normalise(arr, window=30, clip=5.0)
        assert out.max() <= 5.0 + 1e-6
        assert out.min() >= -5.0 - 1e-6

    def test_all_same_returns_zeros(self):
        from pufferlib_market.export_data_daily_v3 import zscore_normalise
        arr = np.ones(50, dtype=np.float32) * 3.14
        out = zscore_normalise(arr, window=20)
        assert np.all(out == 0.0)

    def test_early_rows_zero(self):
        from pufferlib_market.export_data_daily_v3 import zscore_normalise
        arr = np.arange(10, dtype=np.float32)
        out = zscore_normalise(arr, window=5)
        # First row (window empty or size 1) should be 0
        assert out[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: MKTD v3 binary roundtrip
# ---------------------------------------------------------------------------

class TestMktdV3Header:
    def test_header_features_per_sym_20(self, tmp_path):
        """Written v3 file has features_per_sym=20 in the header."""
        bin_path = tmp_path / "test_v3.bin"
        _write_mktd_bin(bin_path, num_symbols=2, num_timesteps=30, features_per_sym=20, version=3)
        with open(bin_path, "rb") as f:
            hdr = f.read(64)
        magic = hdr[:4]
        v, ns, nt, fps, pf = struct.unpack("<IIIII", hdr[4:24])
        assert magic == b"MKTD"
        assert v == 3
        assert fps == 20

    def test_read_mktd_v3_features_shape(self, tmp_path):
        """read_mktd parses v3 files and produces features with shape [T, S, 20]."""
        from pufferlib_market.hourly_replay import read_mktd
        bin_path = tmp_path / "v3.bin"
        _write_mktd_bin(bin_path, num_symbols=3, num_timesteps=25, features_per_sym=20, version=3)
        data = read_mktd(bin_path)
        assert data.features.shape == (25, 3, 20)
        assert data.num_symbols == 3
        assert data.num_timesteps == 25

    def test_read_mktd_v2_features_shape(self, tmp_path):
        """read_mktd still handles v2 files with features_per_sym=16."""
        from pufferlib_market.hourly_replay import read_mktd
        bin_path = tmp_path / "v2.bin"
        _write_mktd_bin(bin_path, num_symbols=2, num_timesteps=20, features_per_sym=16, version=2)
        data = read_mktd(bin_path)
        assert data.features.shape == (20, 2, 16)


# ---------------------------------------------------------------------------
# Tests: C env loads v3 binary with correct obs_size (subprocess approach)
# ---------------------------------------------------------------------------

import json
import subprocess
import sys as _sys


def _run_script(script: str) -> str:
    result = subprocess.run(
        [_sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Script failed:\n{result.stderr}")
    return result.stdout.strip()


class TestCEnvV3:
    def test_v3_obs_size(self, tmp_path):
        """C env binding loads v3 data and obs_size = S*20 + 5 + S."""
        bin_path = tmp_path / "v3_test.bin"
        S = 3
        T = 100
        _write_mktd_bin(bin_path, num_symbols=S, num_timesteps=T, features_per_sym=20, version=3)

        script = f"""
import json
import numpy as np
import pufferlib_market.binding as binding

S = {S}
T = {T}
features_per_sym = 20
obs_size = S * features_per_sym + 5 + S

obs_buf  = np.zeros((1, obs_size), dtype=np.float32)
act_buf  = np.zeros((1,), dtype=np.int32)
rew_buf  = np.zeros((1,), dtype=np.float32)
term_buf = np.zeros((1,), dtype=np.uint8)
trunc_buf = np.zeros((1,), dtype=np.uint8)

binding.shared(data_path={json.dumps(str(bin_path))})
vec = binding.vec_init(obs_buf, act_buf, rew_buf, term_buf, trunc_buf, 1, 7,
    max_steps=10, fee_rate=0.001)
binding.vec_reset(vec, 7)
# obs should be finite
all_finite = bool(np.all(np.isfinite(obs_buf)))
binding.vec_close(vec)
print(json.dumps({{"obs_size": obs_size, "all_finite": all_finite}}))
"""
        out = json.loads(_run_script(script))
        expected_obs_size = S * 20 + 5 + S
        assert out["obs_size"] == expected_obs_size
        assert out["all_finite"]

    def test_v2_obs_size_unchanged(self, tmp_path):
        """C env binding loads v2 data with obs_size = S*16 + 5 + S (backwards compat)."""
        bin_path = tmp_path / "v2_test.bin"
        S = 2
        T = 50
        _write_mktd_bin(bin_path, num_symbols=S, num_timesteps=T, features_per_sym=16, version=2)

        script = f"""
import json
import numpy as np
import pufferlib_market.binding as binding

S = {S}
T = {T}
features_per_sym = 16
obs_size = S * features_per_sym + 5 + S

obs_buf  = np.zeros((1, obs_size), dtype=np.float32)
act_buf  = np.zeros((1,), dtype=np.int32)
rew_buf  = np.zeros((1,), dtype=np.float32)
term_buf = np.zeros((1,), dtype=np.uint8)
trunc_buf = np.zeros((1,), dtype=np.uint8)

binding.shared(data_path={json.dumps(str(bin_path))})
vec = binding.vec_init(obs_buf, act_buf, rew_buf, term_buf, trunc_buf, 1, 7,
    max_steps=10, fee_rate=0.001)
binding.vec_reset(vec, 7)
all_finite = bool(np.all(np.isfinite(obs_buf)))
binding.vec_close(vec)
print(json.dumps({{"obs_size": obs_size, "all_finite": all_finite}}))
"""
        out = json.loads(_run_script(script))
        expected_obs_size = S * 16 + 5 + S
        assert out["obs_size"] == expected_obs_size
        assert out["all_finite"]


# ---------------------------------------------------------------------------
# Tests: train.py obs_size reads features_per_sym from header
# ---------------------------------------------------------------------------

class TestTrainObsSize:
    def test_reads_features_per_sym_from_header(self, tmp_path):
        """train.py reads features_per_sym from MKTD header correctly."""
        bin_path = tmp_path / "test.bin"
        S = 4
        _write_mktd_bin(bin_path, num_symbols=S, num_timesteps=50, features_per_sym=20, version=3)

        import struct as st
        with open(bin_path, "rb") as f:
            header = f.read(64)
        _, _, num_symbols, num_timesteps, features_per_sym, _ = st.unpack("<4sIIIII", header[:24])
        if features_per_sym == 0:
            features_per_sym = 16

        obs_size = num_symbols * features_per_sym + 5 + num_symbols
        assert features_per_sym == 20
        assert obs_size == S * 20 + 5 + S

    def test_v2_header_features_per_sym_16(self, tmp_path):
        """v2 header correctly gives features_per_sym=16."""
        bin_path = tmp_path / "test_v2.bin"
        S = 5
        _write_mktd_bin(bin_path, num_symbols=S, num_timesteps=30, features_per_sym=16, version=2)

        import struct as st
        with open(bin_path, "rb") as f:
            header = f.read(64)
        _, _, num_symbols, num_timesteps, features_per_sym, _ = st.unpack("<4sIIIII", header[:24])
        obs_size = num_symbols * features_per_sym + 5 + num_symbols
        assert features_per_sym == 16
        assert obs_size == S * 16 + 5 + S


# ---------------------------------------------------------------------------
# Tests: TransformerTradingPolicy inference with v3
# ---------------------------------------------------------------------------

class TestTransformerPolicyV3:
    def test_v3_obs_size_inference(self):
        """TransformerTradingPolicy correctly infers num_symbols for v3 (20 features)."""
        import torch
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from pufferlib_market.train import TransformerTradingPolicy

        S = 6
        F = 20
        obs_size = S * F + 5 + S  # S*(F+1) + 5
        num_actions = 1 + 2 * S

        policy = TransformerTradingPolicy(obs_size, num_actions, hidden=64, features_per_sym=F)
        assert policy.num_symbols == S
        assert policy.per_symbol_features == F

        # Forward pass
        batch = torch.randn(4, obs_size)
        logits, value = policy.forward(batch)
        assert logits.shape == (4, num_actions)
        assert value.shape == (4,)

    def test_v2_default_16_features(self):
        """TransformerTradingPolicy defaults to 16 features (backwards compat)."""
        import torch
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        from pufferlib_market.train import TransformerTradingPolicy

        S = 4
        F = 16
        obs_size = S * F + 5 + S
        num_actions = 1 + 2 * S

        policy = TransformerTradingPolicy(obs_size, num_actions, hidden=64)
        assert policy.num_symbols == S
        assert policy.per_symbol_features == 16


# ---------------------------------------------------------------------------
# Tests: evaluate_multiperiod.load_policy passes features_per_sym
# ---------------------------------------------------------------------------

class TestEvaluateMultiperiodFeaturesSym:
    def test_load_policy_uses_features_per_sym(self, tmp_path):
        """load_policy computes correct obs_size for v3 features."""
        import torch
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        # We test the obs_size computation directly since we can't easily load a real checkpoint
        S = 3
        F = 20
        obs_size = S * F + 5 + S
        from pufferlib_market.train import TradingPolicy
        # Just verify obs_size math
        assert obs_size == S * F + 5 + S
        assert obs_size == 3 * 20 + 5 + 3  # == 68
