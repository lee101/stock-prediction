from __future__ import annotations

import pandas as pd
import torch

from RLgpt.config import DailyPlanDataConfig
from RLgpt.data import TensorNormalizer, prepare_daily_plan_tensors


def test_prepare_daily_plan_tensors_aligns_days_and_features(monkeypatch):
    frames = {
        "AAA": _make_hourly_feature_frame(price_offset=0.0),
        "BBB": _make_hourly_feature_frame(price_offset=5.0),
    }
    monkeypatch.setattr(
        "RLgpt.data.load_symbol_hourly_feature_frame",
        lambda symbol, _config: frames[str(symbol).upper()].copy(),
    )

    bundle = prepare_daily_plan_tensors(
        DailyPlanDataConfig(
            symbols=("AAA", "BBB"),
            forecast_horizons=(1, 24),
            validation_days=1,
        )
    )

    assert bundle.features.shape == (2, 2, 11)
    assert bundle.hourly_open.shape == (2, 2, 2)
    assert bundle.hourly_mask.sum().item() == 8.0
    assert bundle.feature_names[0] == "open_gap_pct"
    assert bundle.days[0] == pd.Timestamp("2024-01-02T00:00:00Z")

    aaa_open_gap = bundle.features[0, 0, 0].item()
    expected_gap = (110.0 - 103.0) / 103.0
    assert aaa_open_gap == pytest_approx(expected_gap)

    bbb_prev_close_delta_h24 = bundle.features[0, 1, 10].item()
    assert bbb_prev_close_delta_h24 == pytest_approx(-0.03)


def test_tensor_normalizer_stabilizes_constant_columns():
    features = torch.tensor(
        [
            [[1.0, 3.0], [1.0, 5.0]],
            [[1.0, 7.0], [1.0, 9.0]],
        ]
    )
    normalizer = TensorNormalizer.fit(features)
    transformed = normalizer.transform(features)

    assert normalizer.std[0].item() == 1.0
    assert transformed[..., 0].abs().max().item() == 0.0
    assert transformed[..., 1].mean().abs().item() < 1e-6


def _make_hourly_feature_frame(*, price_offset: float) -> pd.DataFrame:
    rows = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "open": 100.0 + price_offset,
            "high": 102.0 + price_offset,
            "low": 99.0 + price_offset,
            "close": 101.0 + price_offset,
            "return_24h": 0.01,
            "volatility_24h": 0.10,
            "range_pct": 0.03,
            "volume_z": 0.20,
            "chronos_close_delta_h1": 0.02,
            "chronos_high_delta_h1": 0.03,
            "chronos_low_delta_h1": -0.01,
            "chronos_close_delta_h24": 0.05,
            "chronos_high_delta_h24": 0.07,
            "chronos_low_delta_h24": -0.02,
        },
        {
            "timestamp": "2024-01-01T01:00:00Z",
            "open": 101.0 + price_offset,
            "high": 104.0 + price_offset,
            "low": 100.0 + price_offset,
            "close": 103.0 + price_offset,
            "return_24h": 0.04,
            "volatility_24h": 0.11,
            "range_pct": 0.04,
            "volume_z": 0.30,
            "chronos_close_delta_h1": 0.04,
            "chronos_high_delta_h1": 0.06,
            "chronos_low_delta_h1": -0.02,
            "chronos_close_delta_h24": 0.08,
            "chronos_high_delta_h24": 0.09,
            "chronos_low_delta_h24": -0.03,
        },
        {
            "timestamp": "2024-01-02T00:00:00Z",
            "open": 110.0 + price_offset,
            "high": 111.0 + price_offset,
            "low": 109.0 + price_offset,
            "close": 110.5 + price_offset,
            "return_24h": 0.02,
            "volatility_24h": 0.12,
            "range_pct": 0.02,
            "volume_z": 0.25,
            "chronos_close_delta_h1": 0.03,
            "chronos_high_delta_h1": 0.04,
            "chronos_low_delta_h1": -0.01,
            "chronos_close_delta_h24": 0.06,
            "chronos_high_delta_h24": 0.08,
            "chronos_low_delta_h24": -0.02,
        },
        {
            "timestamp": "2024-01-02T01:00:00Z",
            "open": 110.5 + price_offset,
            "high": 112.0 + price_offset,
            "low": 108.5 + price_offset,
            "close": 111.0 + price_offset,
            "return_24h": 0.01,
            "volatility_24h": 0.13,
            "range_pct": 0.03,
            "volume_z": 0.35,
            "chronos_close_delta_h1": 0.02,
            "chronos_high_delta_h1": 0.05,
            "chronos_low_delta_h1": -0.02,
            "chronos_close_delta_h24": 0.07,
            "chronos_high_delta_h24": 0.10,
            "chronos_low_delta_h24": -0.01,
        },
        {
            "timestamp": "2024-01-03T00:00:00Z",
            "open": 120.0 + price_offset,
            "high": 121.5 + price_offset,
            "low": 119.0 + price_offset,
            "close": 120.5 + price_offset,
            "return_24h": 0.03,
            "volatility_24h": 0.14,
            "range_pct": 0.02,
            "volume_z": 0.40,
            "chronos_close_delta_h1": 0.05,
            "chronos_high_delta_h1": 0.07,
            "chronos_low_delta_h1": -0.02,
            "chronos_close_delta_h24": 0.09,
            "chronos_high_delta_h24": 0.11,
            "chronos_low_delta_h24": -0.04,
        },
        {
            "timestamp": "2024-01-03T01:00:00Z",
            "open": 121.0 + price_offset,
            "high": 123.0 + price_offset,
            "low": 120.0 + price_offset,
            "close": 122.0 + price_offset,
            "return_24h": 0.06,
            "volatility_24h": 0.15,
            "range_pct": 0.025,
            "volume_z": 0.45,
            "chronos_close_delta_h1": 0.06,
            "chronos_high_delta_h1": 0.08,
            "chronos_low_delta_h1": -0.01,
            "chronos_close_delta_h24": 0.10,
            "chronos_high_delta_h24": 0.12,
            "chronos_low_delta_h24": -0.03,
        },
    ]
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value)
