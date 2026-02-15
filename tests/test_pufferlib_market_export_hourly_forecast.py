from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pufferlib_market.export_data_hourly_forecast import export_binary
from pufferlib_market.hourly_replay import read_mktd


def _write_symbol_csv(path: Path, index: pd.DatetimeIndex, *, base: float) -> None:
    close = np.linspace(base, base * 1.1, num=len(index), dtype=np.float64)
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.linspace(10_000, 20_000, num=len(index), dtype=np.float64),
        }
    )
    frame.to_csv(path, index=False)


def _write_forecast_parquet(path: Path, index: pd.DatetimeIndex, *, close: np.ndarray) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "issued_at": index,
            "predicted_close_p50": close * 1.01,
            "predicted_close_p10": close * 0.99,
            "predicted_close_p90": close * 1.03,
            "predicted_high_p50": close * 1.02,
            "predicted_low_p50": close * 0.98,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_export_binary_writes_v2_with_tradable_mask_and_forecast_features(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "stocks"
    data_root.mkdir(parents=True)

    cache_root = tmp_path / "cache"
    (cache_root / "h1").mkdir(parents=True)
    (cache_root / "h24").mkdir(parents=True)

    idx = pd.date_range("2026-01-01 00:00:00+00:00", periods=72, freq="h", tz="UTC")

    # AAA is complete; BBB misses one bar so tradable mask should contain a 0.
    _write_symbol_csv(data_root / "AAA.csv", idx, base=100.0)
    bbb_idx = idx.delete(10)
    _write_symbol_csv(data_root / "BBB.csv", bbb_idx, base=200.0)

    # Forecasts exist on the full index (forecast features should not be all zeros).
    aaa_close = np.linspace(100.0, 110.0, num=len(idx), dtype=np.float64)
    bbb_close = np.linspace(200.0, 220.0, num=len(idx), dtype=np.float64)
    _write_forecast_parquet(cache_root / "h1" / "AAA.parquet", idx, close=aaa_close)
    _write_forecast_parquet(cache_root / "h24" / "AAA.parquet", idx, close=aaa_close)
    _write_forecast_parquet(cache_root / "h1" / "BBB.parquet", idx, close=bbb_close)
    _write_forecast_parquet(cache_root / "h24" / "BBB.parquet", idx, close=bbb_close)

    out = tmp_path / "export.bin"
    report = export_binary(
        symbols=["AAA", "BBB"],
        data_root=tmp_path / "data",
        forecast_cache_root=cache_root,
        output_path=out,
        min_hours=24,
        min_coverage=0.95,
    )
    assert report["num_symbols"] == 2
    assert report["num_timesteps"] == 72

    parsed = read_mktd(out)
    assert parsed.version == 2
    assert parsed.symbols == ["AAA", "BBB"]
    assert parsed.features.shape == (72, 2, 16)
    assert parsed.prices.shape == (72, 2, 5)
    assert parsed.tradable is not None
    assert parsed.tradable.shape == (72, 2)
    assert parsed.tradable[10, 1] == 0
    assert parsed.tradable[:, 0].min() == 1

    # Feature[0] is chronos_close_delta_h1; mean should be close to +1%.
    chronos_close_delta_h1 = parsed.features[:, 0, 0]
    assert float(np.mean(chronos_close_delta_h1)) == pytest.approx(0.01, rel=1e-2, abs=5e-3)


def test_export_binary_prefers_issued_at_alignment(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "stocks"
    data_root.mkdir(parents=True)

    cache_root = tmp_path / "cache"
    (cache_root / "h1").mkdir(parents=True)
    (cache_root / "h24").mkdir(parents=True)

    idx = pd.date_range("2026-01-01 00:00:00+00:00", periods=10, freq="h", tz="UTC")

    # Constant prices make the expected delta exact.
    close = np.full((len(idx),), 100.0, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full((len(idx),), 1000.0, dtype=np.float64),
        }
    )
    frame.to_csv(data_root / "AAA.csv", index=False)

    # Forecast rows are keyed by `issued_at` (what we want), while `timestamp` is shifted.
    fc = pd.DataFrame(
        {
            "timestamp": idx + pd.Timedelta(hours=1),
            "issued_at": idx,
            "predicted_close_p50": np.full((len(idx),), 101.0, dtype=np.float64),
            "predicted_close_p10": np.full((len(idx),), 99.0, dtype=np.float64),
            "predicted_close_p90": np.full((len(idx),), 103.0, dtype=np.float64),
            "predicted_high_p50": np.full((len(idx),), 102.0, dtype=np.float64),
            "predicted_low_p50": np.full((len(idx),), 98.0, dtype=np.float64),
        }
    )
    (cache_root / "h1" / "AAA.parquet").parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(cache_root / "h1" / "AAA.parquet", index=False)
    fc.to_parquet(cache_root / "h24" / "AAA.parquet", index=False)

    out = tmp_path / "export.bin"
    export_binary(
        symbols=["AAA"],
        data_root=tmp_path / "data",
        forecast_cache_root=cache_root,
        output_path=out,
        min_hours=5,
        min_coverage=1.0,
    )
    parsed = read_mktd(out)

    # If we align by issued_at, the first feature should reflect (101-100)/100 = 0.01.
    assert float(parsed.features[0, 0, 0]) == pytest.approx(0.01, rel=1e-6, abs=1e-6)


def test_export_binary_feature_lag_shifts_features_only(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "stocks"
    data_root.mkdir(parents=True)

    cache_root = tmp_path / "cache"
    (cache_root / "h1").mkdir(parents=True)
    (cache_root / "h24").mkdir(parents=True)

    idx = pd.date_range("2026-01-01 00:00:00+00:00", periods=72, freq="h", tz="UTC")

    _write_symbol_csv(data_root / "AAA.csv", idx, base=100.0)
    bbb_idx = idx.delete(10)
    _write_symbol_csv(data_root / "BBB.csv", bbb_idx, base=200.0)

    aaa_close = np.linspace(100.0, 110.0, num=len(idx), dtype=np.float64)
    bbb_close = np.linspace(200.0, 220.0, num=len(idx), dtype=np.float64)
    _write_forecast_parquet(cache_root / "h1" / "AAA.parquet", idx, close=aaa_close)
    _write_forecast_parquet(cache_root / "h24" / "AAA.parquet", idx, close=aaa_close)
    _write_forecast_parquet(cache_root / "h1" / "BBB.parquet", idx, close=bbb_close)
    _write_forecast_parquet(cache_root / "h24" / "BBB.parquet", idx, close=bbb_close)

    out0 = tmp_path / "export_nolag.bin"
    export_binary(
        symbols=["AAA", "BBB"],
        data_root=tmp_path / "data",
        forecast_cache_root=cache_root,
        output_path=out0,
        feature_lag=0,
        min_hours=24,
        min_coverage=0.95,
    )
    parsed0 = read_mktd(out0)

    out1 = tmp_path / "export_lag1.bin"
    export_binary(
        symbols=["AAA", "BBB"],
        data_root=tmp_path / "data",
        forecast_cache_root=cache_root,
        output_path=out1,
        feature_lag=1,
        min_hours=24,
        min_coverage=0.95,
    )
    parsed1 = read_mktd(out1)

    np.testing.assert_allclose(parsed1.prices, parsed0.prices)
    assert parsed0.tradable is not None
    assert parsed1.tradable is not None
    np.testing.assert_array_equal(parsed1.tradable, parsed0.tradable)

    # Lagged features should shift by one timestep, padded with zeros at t=0.
    assert float(np.max(np.abs(parsed1.features[0]))) == 0.0
    np.testing.assert_allclose(parsed1.features[1:], parsed0.features[:-1])
