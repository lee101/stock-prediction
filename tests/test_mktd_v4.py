from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd

from pufferlib_market.export_data_daily_v4 import (
    FEATURES_PER_SYM_V4,
    compute_daily_forecast_features,
    compute_hourly_forecast_context_features,
    export_binary,
)
from pufferlib_market.hourly_replay import read_mktd


def _make_daily_df(n: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq="D", tz="UTC")
    close = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    open_ = close * (1 + rng.normal(0.0, 0.002, n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, n))
    volume = rng.uniform(1e6, 5e6, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _make_hourly_price_df(daily_df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | pd.Timestamp]] = []
    for date, row in daily_df.iterrows():
        for hour in range(6):
            ts = date + pd.Timedelta(hours=14 + hour)
            close = float(row["close"]) * (1.0 + rng.normal(0.0, 0.003))
            open_ = close * (1.0 + rng.normal(0.0, 0.001))
            rows.append(
                {
                    "timestamp": ts,
                    "open": open_,
                    "high": max(open_, close) * 1.001,
                    "low": min(open_, close) * 0.999,
                    "close": close,
                    "volume": float(rng.uniform(5e4, 2e5)),
                }
            )
    return pd.DataFrame(rows)


def _make_daily_forecast_df(daily_df: pd.DataFrame) -> pd.DataFrame:
    close = daily_df["close"].values
    idx = daily_df.index
    return pd.DataFrame(
        {
            "timestamp": idx + pd.Timedelta(hours=5),
            "date": idx,
            "symbol": "AAA",
            "predicted_close": close * 1.02,
            "predicted_high": close * 1.05,
            "predicted_low": close * 0.98,
            "predicted_close_p10": close * 0.99,
            "predicted_close_p90": close * 1.03,
            "forecast_move_pct": np.full(len(idx), 0.02),
            "forecast_volatility_pct": np.full(len(idx), 0.04),
        }
    )


def _make_hourly_forecast_df(daily_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, row in daily_df.iterrows():
        base = float(row["close"])
        for hour_idx in range(3):
            issued_at = date + pd.Timedelta(hours=14 + hour_idx)
            scale = 1.0 + (0.005 * (hour_idx + 1)) if horizon == 1 else 1.0 + (0.01 * (hour_idx + 1))
            rows.append(
                {
                    "timestamp": issued_at + pd.Timedelta(hours=1),
                    "symbol": "AAA",
                    "issued_at": issued_at,
                    "target_timestamp": issued_at + pd.Timedelta(hours=horizon),
                    "horizon_hours": horizon,
                    "predicted_close_p50": base * scale,
                    "predicted_close_p10": base * (scale - 0.01),
                    "predicted_close_p90": base * (scale + 0.01),
                    "predicted_high_p50": base * (scale + 0.01),
                    "predicted_low_p50": base * (scale - 0.01),
                }
            )
    return pd.DataFrame(rows)


def test_compute_daily_forecast_features_uses_deltas_and_confidence() -> None:
    daily = _make_daily_df(5)
    forecast = _make_daily_forecast_df(daily).set_index(daily.index)

    result = compute_daily_forecast_features(forecast, daily)

    assert list(result.columns) == [
        "daily_close_delta",
        "daily_high_delta",
        "daily_low_delta",
        "daily_confidence",
    ]
    assert result["daily_close_delta"].iloc[0] == np.float32(0.02)
    assert result["daily_high_delta"].iloc[0] == np.float32(0.05)
    assert result["daily_low_delta"].iloc[0] == np.float32(-0.02)
    assert 0.0 < float(result["daily_confidence"].iloc[0]) <= 1.0


def test_compute_hourly_forecast_context_features_uses_mean_and_slope() -> None:
    daily = _make_daily_df(4)
    fc_h1 = _make_hourly_forecast_df(daily, horizon=1).set_index("issued_at")
    fc_h24 = _make_hourly_forecast_df(daily, horizon=24).set_index("issued_at")

    result = compute_hourly_forecast_context_features(fc_h1, fc_h24, daily)

    assert list(result.columns) == [
        "hourly_close_delta_mean_h1",
        "hourly_close_delta_slope_h1",
        "hourly_close_delta_mean_h24",
        "hourly_confidence_mean_h24",
    ]
    assert float(result["hourly_close_delta_mean_h1"].iloc[0]) > 0.0
    assert float(result["hourly_close_delta_slope_h1"].iloc[0]) > 0.0
    assert float(result["hourly_close_delta_mean_h24"].iloc[0]) > 0.0
    assert 0.0 < float(result["hourly_confidence_mean_h24"].iloc[0]) <= 1.0


def test_missing_forecasts_return_zero_features() -> None:
    daily = _make_daily_df(3)

    daily_features = compute_daily_forecast_features(None, daily)
    hourly_features = compute_hourly_forecast_context_features(None, None, daily)

    assert np.allclose(daily_features.values, 0.0)
    assert np.allclose(hourly_features.values, 0.0)


def test_export_binary_writes_v4_header_and_feature_width(tmp_path: Path) -> None:
    data_root = tmp_path / "daily"
    hourly_root = tmp_path / "hourly" / "stocks"
    daily_fc_root = tmp_path / "daily_fc"
    hourly_fc_root = tmp_path / "hourly_fc"
    output_path = tmp_path / "out" / "fused.bin"
    data_root.mkdir(parents=True)
    hourly_root.mkdir(parents=True)
    daily_fc_root.mkdir(parents=True)
    (hourly_fc_root / "h1").mkdir(parents=True)
    (hourly_fc_root / "h24").mkdir(parents=True)

    daily = _make_daily_df(30)
    hourly_prices = _make_hourly_price_df(daily)
    daily_forecast = _make_daily_forecast_df(daily)
    hourly_forecast_h1 = _make_hourly_forecast_df(daily, horizon=1)
    hourly_forecast_h24 = _make_hourly_forecast_df(daily, horizon=24)

    daily_reset = daily.reset_index().rename(columns={"index": "date"})
    daily_reset.to_csv(data_root / "AAA.csv", index=False)
    hourly_prices.to_csv(hourly_root / "AAA.csv", index=False)
    daily_forecast.to_parquet(daily_fc_root / "AAA.parquet", index=False)
    hourly_forecast_h1.to_parquet(hourly_fc_root / "h1" / "AAA.parquet", index=False)
    hourly_forecast_h24.to_parquet(hourly_fc_root / "h24" / "AAA.parquet", index=False)

    export_binary(
        symbols=["AAA"],
        data_root=data_root,
        hourly_root=hourly_root,
        daily_forecast_root=daily_fc_root,
        hourly_forecast_root=hourly_fc_root,
        output_path=output_path,
        min_days=10,
    )

    with open(output_path, "rb") as handle:
        header = handle.read(64)
    magic = header[:4]
    version, num_symbols, num_timesteps, features_per_sym, price_features = struct.unpack("<IIIII", header[4:24])

    assert magic == b"MKTD"
    assert version == 4
    assert num_symbols == 1
    assert num_timesteps == 30
    assert features_per_sym == FEATURES_PER_SYM_V4
    assert price_features == 5

    data = read_mktd(output_path)
    assert data.features.shape == (30, 1, FEATURES_PER_SYM_V4)
    assert float(np.abs(data.features[:, 0, 20:]).sum()) > 0.0
