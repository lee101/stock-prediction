"""Tests for multivariate Chronos2 LoRA trainer extension."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chronos2_trainer import (
    TrainerConfig, _prepare_inputs, _prepare_inputs_multivariate,
    _load_hourly_frame, _split_windows,
)


def _make_hourly_df(n_rows=100, seed=42):
    rng = np.random.RandomState(seed)
    base = 100 + rng.randn(n_rows).cumsum() * 0.5
    ts = pd.date_range("2024-01-01", periods=n_rows, freq="h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": base + rng.randn(n_rows) * 0.1,
        "high": base + abs(rng.randn(n_rows)) * 0.5,
        "low": base - abs(rng.randn(n_rows)) * 0.5,
        "close": base,
    })


def test_prepare_inputs_shape():
    df = _make_hourly_df(50)
    inputs = _prepare_inputs(df, ("open", "high", "low", "close"))
    assert len(inputs) == 1
    assert inputs[0]["target"].shape == (4, 50)


def test_prepare_inputs_multivariate_single_covariate():
    target_df = _make_hourly_df(50, seed=42).set_index("timestamp")
    cov_df = _make_hourly_df(50, seed=99).set_index("timestamp")
    inputs = _prepare_inputs_multivariate(
        target_df, ("open", "high", "low", "close"),
        [cov_df], ("close",),
    )
    assert len(inputs) == 1
    assert inputs[0]["target"].shape == (5, 50)


def test_prepare_inputs_multivariate_two_covariates():
    target_df = _make_hourly_df(50, seed=42).set_index("timestamp")
    cov1 = _make_hourly_df(50, seed=10).set_index("timestamp")
    cov2 = _make_hourly_df(50, seed=20).set_index("timestamp")
    inputs = _prepare_inputs_multivariate(
        target_df, ("open", "high", "low", "close"),
        [cov1, cov2], ("close",),
    )
    assert len(inputs) == 1
    assert inputs[0]["target"].shape == (6, 50)


def test_prepare_inputs_multivariate_multi_cols():
    target_df = _make_hourly_df(50, seed=42).set_index("timestamp")
    cov_df = _make_hourly_df(50, seed=99).set_index("timestamp")
    inputs = _prepare_inputs_multivariate(
        target_df, ("open", "high", "low", "close"),
        [cov_df], ("open", "close"),
    )
    assert len(inputs) == 1
    assert inputs[0]["target"].shape == (6, 50)


def test_prepare_inputs_multivariate_preserves_target_channels():
    target_df = _make_hourly_df(50, seed=42).set_index("timestamp")
    cov_df = _make_hourly_df(50, seed=99).set_index("timestamp")
    inputs = _prepare_inputs_multivariate(
        target_df, ("open", "high", "low", "close"),
        [cov_df], ("close",),
    )
    target_only = _prepare_inputs(target_df, ("open", "high", "low", "close"))
    np.testing.assert_array_equal(
        inputs[0]["target"][:4],
        target_only[0]["target"],
    )


def test_trainer_config_covariates_default():
    cfg = TrainerConfig(symbol="TEST", data_root=None, output_root=Path("/tmp"))
    assert cfg.covariate_symbols == ()
    assert cfg.covariate_cols == ("close",)


def test_trainer_config_covariates_set():
    cfg = TrainerConfig(
        symbol="TEST", data_root=None, output_root=Path("/tmp"),
        covariate_symbols=("PEPEUSDT", "BTCFDUSD"),
        covariate_cols=("open", "close"),
    )
    assert len(cfg.covariate_symbols) == 2
    assert cfg.covariate_cols == ("open", "close")


def test_split_windows():
    df = _make_hourly_df(200)
    train, val, test = _split_windows(df, 30, 30)
    assert len(train) == 140
    assert len(val) == 30
    assert len(test) == 30
