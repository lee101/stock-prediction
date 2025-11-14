from __future__ import annotations

import pandas as pd

from src.kronos_prediction_cache import KronosPredictionCache


def _make_frame(rows: int = 8) -> pd.DataFrame:
    rng = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": rng,
        "Close": pd.Series(range(rows), dtype=float) + 100.0,
        "High": pd.Series(range(rows), dtype=float) + 101.0,
    })


def test_kronos_cache_tolerates_float_noise(tmp_path):
    cache = KronosPredictionCache(cache_dir=str(tmp_path / "kronos"), enabled=True, ttl_seconds=0, value_precision=6)
    frame = _make_frame()
    payload = {"predictions": [1.0, 2.0], "absolute_last": 1.0}

    cache.set(
        symbol="AAPL",
        column="Close",
        data=frame,
        pred_len=7,
        result=payload,
        lookback=6,
        temperature=0.2,
        top_p=0.9,
        top_k=10,
        sample_count=32,
    )

    noisy = frame.copy()
    noisy.loc[noisy.index[-1], "Close"] += 1e-7

    hit = cache.get(
        symbol="AAPL",
        column="Close",
        data=noisy,
        pred_len=7,
        lookback=6,
        temperature=0.2,
        top_p=0.9,
        top_k=10,
        sample_count=32,
    )

    assert hit is not None
    assert hit["predictions"] == payload["predictions"]


def test_kronos_cache_miss_on_large_delta(tmp_path):
    cache = KronosPredictionCache(cache_dir=str(tmp_path / "kronos"), enabled=True, ttl_seconds=0, value_precision=6)
    frame = _make_frame()
    payload = {"predictions": [1.0, 2.0], "absolute_last": 1.0}

    cache.set(
        symbol="ETHUSD",
        column="Close",
        data=frame,
        pred_len=7,
        result=payload,
        lookback=6,
        temperature=0.2,
        top_p=0.9,
        top_k=10,
        sample_count=32,
    )

    shifted = frame.copy()
    shifted.loc[shifted.index[-1], "Close"] += 1e-2

    miss = cache.get(
        symbol="ETHUSD",
        column="Close",
        data=shifted,
        pred_len=7,
        lookback=6,
        temperature=0.2,
        top_p=0.9,
        top_k=10,
        sample_count=32,
    )

    assert miss is None
