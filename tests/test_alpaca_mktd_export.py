"""Tests for export_data_alpaca_daily.py — MKTD v2 binary format verification."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from export_data_alpaca_daily import (
    FEATURES_PER_SYM,
    MAGIC,
    MIN_TRAIN_DAYS,
    MIN_VAL_DAYS,
    PRICE_FEATURES,
    VERSION,
    compute_daily_features,
    export_alpaca_daily,
    load_symbol_daily,
)
from src.trade_directions import DEFAULT_ALPACA_LIVE8_STOCKS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_ROOT = REPO_ROOT / "trainingdata" / "train"
HOURLY_ROOT = REPO_ROOT / "trainingdatahourly" / "stocks"
OUTPUT_DIR = REPO_ROOT / "pufferlib_market" / "data"
TRAIN_BIN = OUTPUT_DIR / "alpaca_daily_train.bin"
VAL_BIN = OUTPUT_DIR / "alpaca_daily_val.bin"


# ---------------------------------------------------------------------------
# Helper: parse MKTD v2 binary
# ---------------------------------------------------------------------------


def parse_mktd_v2(path: Path) -> dict:
    """Parse an MKTD v2 binary and return a dict with parsed arrays."""
    with open(path, "rb") as fh:
        raw_magic = fh.read(4)
        version, num_symbols, num_timesteps, features_per_sym, price_features = struct.unpack(
            "<IIIII", fh.read(20)
        )
        _padding = fh.read(40)

        sym_names = []
        for _ in range(num_symbols):
            raw = fh.read(16)
            sym_names.append(raw.rstrip(b"\x00").decode("ascii", errors="replace"))

        n_feat_floats = num_timesteps * num_symbols * features_per_sym
        features = np.frombuffer(fh.read(n_feat_floats * 4), dtype=np.float32).reshape(
            num_timesteps, num_symbols, features_per_sym
        )

        n_price_floats = num_timesteps * num_symbols * price_features
        prices = np.frombuffer(fh.read(n_price_floats * 4), dtype=np.float32).reshape(
            num_timesteps, num_symbols, price_features
        )

        n_mask_bytes = num_timesteps * num_symbols
        mask = np.frombuffer(fh.read(n_mask_bytes), dtype=np.uint8).reshape(
            num_timesteps, num_symbols
        )

    return {
        "magic": raw_magic,
        "version": version,
        "num_symbols": num_symbols,
        "num_timesteps": num_timesteps,
        "features_per_sym": features_per_sym,
        "price_features": price_features,
        "sym_names": sym_names,
        "features": features,
        "prices": prices,
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# Unit tests: data loading and feature computation
# ---------------------------------------------------------------------------


class TestLoadSymbolDaily:
    def test_loads_nvda_daily(self):
        df = load_symbol_daily("NVDA", DAILY_ROOT, HOURLY_ROOT)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}
        assert len(df) > 200

    def test_index_is_utc_datetimeindex(self):
        df = load_symbol_daily("NVDA", DAILY_ROOT, HOURLY_ROOT)
        assert hasattr(df.index, "tz")
        assert str(df.index.tz) == "UTC"

    def test_no_duplicate_dates(self):
        df = load_symbol_daily("NVDA", DAILY_ROOT, HOURLY_ROOT)
        assert not df.index.duplicated().any()

    def test_index_sorted(self):
        df = load_symbol_daily("NVDA", DAILY_ROOT, HOURLY_ROOT)
        assert df.index.is_monotonic_increasing

    def test_hourly_extends_range(self):
        # Hourly data (2025-02 to 2026-03) should push the max date beyond daily-only coverage.
        df_full = load_symbol_daily("NVDA", DAILY_ROOT, HOURLY_ROOT)
        df_daily_only = load_symbol_daily("NVDA", DAILY_ROOT, Path("/nonexistent"))
        # Full data should end later than daily-only (or at least as late).
        assert df_full.index.max() >= df_daily_only.index.max()

    def test_missing_symbol_raises(self):
        with pytest.raises(FileNotFoundError):
            load_symbol_daily("ZZZNOTREAL", DAILY_ROOT, HOURLY_ROOT)

    @pytest.mark.parametrize("sym", list(DEFAULT_ALPACA_LIVE8_STOCKS))
    def test_all_live8_symbols_load(self, sym):
        df = load_symbol_daily(sym, DAILY_ROOT, HOURLY_ROOT)
        assert len(df) > 50, f"{sym} has too few rows: {len(df)}"


class TestComputeDailyFeatures:
    @pytest.fixture
    def sample_price_df(self):
        df = load_symbol_daily("NVDA", DAILY_ROOT, HOURLY_ROOT)
        return df

    def test_output_shape(self, sample_price_df):
        feat = compute_daily_features(sample_price_df)
        assert feat.shape == (len(sample_price_df), FEATURES_PER_SYM)

    def test_dtype_float32(self, sample_price_df):
        feat = compute_daily_features(sample_price_df)
        assert feat.dtypes.unique().tolist() == [np.float32]

    def test_no_nan(self, sample_price_df):
        feat = compute_daily_features(sample_price_df)
        assert not feat.isnull().any().any()

    def test_feature_ranges(self, sample_price_df):
        feat = compute_daily_features(sample_price_df)
        # return_1d clipped to [-0.5, 0.5]
        assert feat["return_1d"].between(-0.5, 0.5).all()
        # volatility >= 0
        assert (feat["volatility_5d"] >= 0).all()
        # drawdown <= 0
        assert (feat["drawdown_20d"] <= 0).all()
        assert (feat["drawdown_60d"] <= 0).all()


# ---------------------------------------------------------------------------
# Integration tests: run the export and verify binary files
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exported_files():
    """Run the export once per test session; return (train_path, val_path)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        train_path, val_path = export_alpaca_daily(
            output_dir=out_dir,
            val_split_date="2025-09-01",
        )
        yield train_path, val_path


@pytest.fixture(scope="module")
def train_parsed(exported_files):
    train_path, _ = exported_files
    return parse_mktd_v2(train_path)


@pytest.fixture(scope="module")
def val_parsed(exported_files):
    _, val_path = exported_files
    return parse_mktd_v2(val_path)


class TestMktdV2Format:
    def test_train_magic(self, train_parsed):
        assert train_parsed["magic"] == MAGIC

    def test_val_magic(self, val_parsed):
        assert val_parsed["magic"] == MAGIC

    def test_train_version(self, train_parsed):
        assert train_parsed["version"] == VERSION

    def test_val_version(self, val_parsed):
        assert val_parsed["version"] == VERSION

    def test_features_per_sym(self, train_parsed, val_parsed):
        assert train_parsed["features_per_sym"] == FEATURES_PER_SYM
        assert val_parsed["features_per_sym"] == FEATURES_PER_SYM

    def test_price_features(self, train_parsed, val_parsed):
        assert train_parsed["price_features"] == PRICE_FEATURES
        assert val_parsed["price_features"] == PRICE_FEATURES

    def test_num_symbols_matches_live8(self, train_parsed):
        expected = len(DEFAULT_ALPACA_LIVE8_STOCKS)
        assert train_parsed["num_symbols"] == expected

    def test_sym_names_are_live8(self, train_parsed):
        exported = set(train_parsed["sym_names"])
        expected = set(DEFAULT_ALPACA_LIVE8_STOCKS)
        assert exported == expected

    def test_train_timesteps_sufficient(self, train_parsed):
        assert train_parsed["num_timesteps"] >= MIN_TRAIN_DAYS

    def test_val_timesteps_sufficient(self, val_parsed):
        assert val_parsed["num_timesteps"] >= MIN_VAL_DAYS

    def test_train_val_same_num_symbols(self, train_parsed, val_parsed):
        assert train_parsed["num_symbols"] == val_parsed["num_symbols"]

    def test_train_val_same_features_layout(self, train_parsed, val_parsed):
        assert train_parsed["features_per_sym"] == val_parsed["features_per_sym"]
        assert train_parsed["price_features"] == val_parsed["price_features"]


class TestMktdV2Prices:
    def test_open_positive_on_tradable_days(self, train_parsed):
        prices = train_parsed["prices"]  # [T, S, 5]
        mask = train_parsed["mask"]      # [T, S]
        open_prices = prices[:, :, 0]
        # Where tradable, open should be > 0
        tradable_open = open_prices[mask == 1]
        assert (tradable_open > 0).mean() > 0.99, "Most tradable days should have positive open"

    def test_close_positive_on_tradable_days(self, train_parsed):
        prices = train_parsed["prices"]
        mask = train_parsed["mask"]
        close_prices = prices[:, :, 3]
        tradable_close = close_prices[mask == 1]
        assert (tradable_close > 0).mean() > 0.99

    def test_high_gte_low(self, train_parsed):
        prices = train_parsed["prices"]
        mask = train_parsed["mask"]
        high = prices[:, :, 1]
        low = prices[:, :, 2]
        # On tradable days, high >= low always
        assert (high[mask == 1] >= low[mask == 1]).all()

    def test_volume_nonneg_on_tradable(self, train_parsed):
        prices = train_parsed["prices"]
        mask = train_parsed["mask"]
        volume = prices[:, :, 4]
        assert (volume[mask == 1] >= 0).all()

    def test_volume_positive_most_tradable_days(self, train_parsed):
        prices = train_parsed["prices"]
        mask = train_parsed["mask"]
        volume = prices[:, :, 4]
        tradable_vol = volume[mask == 1]
        assert (tradable_vol > 0).mean() > 0.90


class TestMktdV2Features:
    def test_features_no_inf(self, train_parsed):
        assert not np.any(np.isinf(train_parsed["features"]))

    def test_features_no_nan(self, train_parsed):
        assert not np.any(np.isnan(train_parsed["features"]))

    def test_features_not_all_zero(self, train_parsed):
        # Each symbol should have non-zero features on most tradable days
        mask = train_parsed["mask"]
        features = train_parsed["features"]
        for si in range(train_parsed["num_symbols"]):
            tradable_rows = mask[:, si] == 1
            sym_feats = features[tradable_rows, si, :]
            assert sym_feats.any(), f"Symbol index {si} has all-zero features on tradable days"

    def test_return_1d_clipped(self, train_parsed):
        ret = train_parsed["features"][:, :, 0]  # feature 0 is return_1d
        assert (ret >= -0.5).all() and (ret <= 0.5).all()


class TestMktdV2TradableMask:
    def test_mask_binary(self, train_parsed):
        mask = train_parsed["mask"]
        unique_vals = set(np.unique(mask))
        assert unique_vals <= {0, 1}

    def test_mask_has_ones(self, train_parsed):
        assert train_parsed["mask"].sum() > 0

    def test_val_mask_binary(self, val_parsed):
        mask = val_parsed["mask"]
        unique_vals = set(np.unique(mask))
        assert unique_vals <= {0, 1}

    def test_weekends_not_tradable(self, train_parsed):
        """Tradable fraction should be ~5/7 of calendar days (weekdays minus holidays)."""
        mask = train_parsed["mask"]
        frac_tradable = mask.mean()
        # Stocks trade ~5 days/7 (~71%). Allow [40%, 90%] to tolerate holidays and short histories.
        assert 0.40 < frac_tradable < 0.90, f"Unexpected tradable fraction: {frac_tradable:.2%}"


# ---------------------------------------------------------------------------
# File-level tests: verify the pre-built files if they exist
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TRAIN_BIN.exists(), reason="alpaca_daily_train.bin not yet generated")
class TestPrebuiltTrainFile:
    def test_parses_cleanly(self):
        parsed = parse_mktd_v2(TRAIN_BIN)
        assert parsed["magic"] == MAGIC
        assert parsed["num_timesteps"] >= MIN_TRAIN_DAYS

    def test_num_symbols(self):
        parsed = parse_mktd_v2(TRAIN_BIN)
        assert parsed["num_symbols"] == len(DEFAULT_ALPACA_LIVE8_STOCKS)


@pytest.mark.skipif(not VAL_BIN.exists(), reason="alpaca_daily_val.bin not yet generated")
class TestPrebuiltValFile:
    def test_parses_cleanly(self):
        parsed = parse_mktd_v2(VAL_BIN)
        assert parsed["magic"] == MAGIC
        assert parsed["num_timesteps"] >= MIN_VAL_DAYS
