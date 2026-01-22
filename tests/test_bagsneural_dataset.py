from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from bagsneural.dataset import build_features_and_targets, build_window_features, fit_normalizer


def _make_df(rows: int) -> pd.DataFrame:
    timestamps = [datetime(2026, 1, 1) + timedelta(minutes=10 * i) for i in range(rows)]
    base = np.linspace(1.0, 1.0 + 0.01 * (rows - 1), rows)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base * 1.005,
        }
    )
    return df


def test_build_window_features_shape():
    df = _make_df(5)
    features = build_window_features(
        df["open"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
    )
    assert features.shape[0] == 5 * 3


def test_build_features_and_targets_shapes():
    df = _make_df(10)
    features, signal, size, timestamps = build_features_and_targets(
        df,
        context_bars=3,
        horizon=2,
        cost_bps=5.0,
        min_return=0.0,
        size_scale=0.01,
    )
    assert features.shape[0] == len(signal) == len(size) == len(timestamps)
    assert features.shape[1] == 3 * 3


def test_fit_normalizer_roundtrip():
    data = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=np.float32)
    normalizer = fit_normalizer(data)
    transformed = normalizer.transform(data)
    assert np.isfinite(transformed).all()
