"""Tests for export_data_weekly.py — weekly MKTD binary exporter."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pufferlib_market.export_data_weekly import (
    BASE_FEATURE_NAMES,
    FEATURES_PER_SYM_BASE,
    FEATURES_PER_SYM_CHRONOS,
    MAGIC,
    VERSION,
    _load_daily_forecast_cache,
    _resample_weekly,
    compute_weekly_chronos_features,
    compute_weekly_features,
    export_binary,
)

REPO = Path(__file__).resolve().parents[1]


def _make_daily_prices(n: int = 300, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start, periods=n, freq="B", tz="UTC")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.012, n))
    high = close * (1.0 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def _make_weekly_prices(n_weeks: int = 60) -> pd.DataFrame:
    return _resample_weekly(_make_daily_prices(n=n_weeks * 7))


def test_resample_weekly_columns():
    weekly = _resample_weekly(_make_daily_prices(200))
    for col in ("open", "high", "low", "close", "volume", "_tradable"):
        assert col in weekly.columns


def test_resample_weekly_high_ge_low():
    weekly = _resample_weekly(_make_daily_prices(200))
    assert (weekly["high"] >= weekly["low"]).all()


def test_resample_weekly_tradable_flag():
    weekly = _resample_weekly(_make_daily_prices(100))
    assert weekly["_tradable"].sum() >= int(len(weekly) * 0.8)


def test_weekly_features_shape():
    feats = compute_weekly_features(_make_weekly_prices(60))
    assert feats.shape == (len(_make_weekly_prices(60)), FEATURES_PER_SYM_BASE)


def test_weekly_features_dtype():
    feats = compute_weekly_features(_make_weekly_prices(60))
    assert feats.dtypes.unique().tolist() == [np.dtype("float32")]


def test_weekly_features_names():
    feats = compute_weekly_features(_make_weekly_prices(60))
    assert list(feats.columns) == BASE_FEATURE_NAMES


def test_weekly_features_no_nan():
    feats = compute_weekly_features(_make_weekly_prices(60))
    assert not feats.isna().any().any()


def test_weekly_features_clipped():
    feats = compute_weekly_features(_make_weekly_prices(60))
    assert (feats["return_1w"].abs() <= 0.5).all()
    assert (feats["rsi_14w"] >= -1.0).all() and (feats["rsi_14w"] <= 1.0).all()
    assert (feats["drawdown_13w"] <= 0.0).all()


def _make_daily_forecast_cache(close_series: pd.Series) -> pd.DataFrame:
    fc = pd.DataFrame(index=close_series.index)
    fc["predicted_close"] = close_series * 1.005
    fc["predicted_close_p90"] = close_series * 1.02
    fc["predicted_close_p10"] = close_series * 0.99
    return fc


def test_chronos_features_default_zeros_if_no_cache():
    weekly = _make_weekly_prices(30)
    daily = _make_daily_prices(250)
    result = compute_weekly_chronos_features(None, weekly.index, daily["close"])
    assert result.shape == (len(weekly), 4)
    assert (result["weekly_close_delta"] == 0.0).all()
    assert (result["weekly_confidence"] == 0.5).all()


def test_chronos_features_with_cache():
    daily = _make_daily_prices(500)
    weekly = _resample_weekly(daily)
    fc = _make_daily_forecast_cache(daily["close"])
    daily_close = daily["close"].copy()
    daily_close.index = daily_close.index.floor("D")
    result = compute_weekly_chronos_features(fc, weekly.index, daily_close)
    assert result.shape == (len(weekly), 4)
    assert result.dtypes.unique().tolist() == [np.dtype("float32")]
    nonzero = (result["weekly_close_delta"] != 0.0).sum()
    assert nonzero >= int(len(weekly) * 0.5), f"Only {nonzero}/{len(weekly)} non-zero"


def test_chronos_features_clipped():
    daily = _make_daily_prices(500)
    weekly = _resample_weekly(daily)
    fc = _make_daily_forecast_cache(daily["close"])
    daily_close = daily["close"].copy()
    daily_close.index = daily_close.index.floor("D")
    result = compute_weekly_chronos_features(fc, weekly.index, daily_close)
    assert (result["weekly_close_delta"].abs() <= 1.0).all()
    assert (result["weekly_confidence"] >= 0.0).all() and (result["weekly_confidence"] <= 1.0).all()


@pytest.fixture
def real_data_root() -> Path:
    p = REPO / "trainingdata"
    if not p.exists():
        pytest.skip("trainingdata/ not present")
    return p


def test_export_binary_base(tmp_path, real_data_root):
    out = tmp_path / "weekly_base.bin"
    export_binary(symbols=["AAPL", "MSFT"], data_root=real_data_root, output_path=out,
                  start_date="2022-01-01", end_date="2024-12-31", min_weeks=20)
    assert out.exists()
    with open(out, "rb") as fh:
        magic, ver, nsym, nts, nfeat, nprice = struct.unpack_from("<4sIIIII", fh.read(24))
    assert magic == MAGIC and ver == VERSION and nsym == 2 and nts >= 20
    assert nfeat == FEATURES_PER_SYM_BASE and nprice == 5
    import os
    expected = 64 + nsym * 16 + nts * nsym * nfeat * 4 + nts * nsym * nprice * 4 + nts * nsym
    assert os.path.getsize(out) == expected


def test_export_binary_chronos(tmp_path, real_data_root):
    chronos_cache = REPO / "strategytraining" / "forecast_cache"
    if not chronos_cache.exists():
        pytest.skip("strategytraining/forecast_cache not present")
    out = tmp_path / "weekly_chronos.bin"
    export_binary(symbols=["AAPL", "MSFT"], data_root=real_data_root, output_path=out,
                  start_date="2022-01-01", end_date="2024-12-31", min_weeks=20, chronos_cache=chronos_cache)
    with open(out, "rb") as fh:
        raw = fh.read()
    _, _, nsym, nts, nfeat, _ = struct.unpack_from("<4sIIIII", raw[:24])
    assert nfeat == FEATURES_PER_SYM_CHRONOS
    feats = np.frombuffer(raw[64 + nsym * 16: 64 + nsym * 16 + nts * nsym * nfeat * 4], dtype=np.float32)
    feats = feats.reshape(nts, nsym, nfeat)
    assert np.count_nonzero(feats[:, 0, 16]) >= int(nts * 0.3)


def test_export_binary_no_nan(tmp_path, real_data_root):
    out = tmp_path / "weekly_nan.bin"
    export_binary(symbols=["AAPL"], data_root=real_data_root, output_path=out,
                  start_date="2020-01-01", end_date="2024-12-31", min_weeks=20)
    with open(out, "rb") as fh:
        raw = fh.read()
    _, _, nsym, nts, nfeat, _ = struct.unpack_from("<4sIIIII", raw[:24])
    feats = np.frombuffer(raw[64 + nsym * 16: 64 + nsym * 16 + nts * nsym * nfeat * 4], dtype=np.float32)
    assert not np.isnan(feats).any()
    assert float(feats.reshape(nts, nsym, nfeat)[:, 0, 0].max()) <= 0.5  # return_1w clipped


def test_export_binary_wrong_symbols_raises(tmp_path, real_data_root):
    with pytest.raises(FileNotFoundError):
        export_binary(symbols=["FAKESYMBOLABC"], data_root=real_data_root, output_path=tmp_path / "fake.bin")
