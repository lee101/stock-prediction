from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import trade_daily_stock_prod as daily_stock
from src.daily_stock_feature_schema import daily_feature_dimension, resolve_daily_feature_schema


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC"),
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
        }
    )


def test_resolve_daily_feature_schema_uses_legacy_for_prod_ensemble() -> None:
    schema = resolve_daily_feature_schema(
        "pufferlib_market/prod_ensemble/tp10.pt",
        extra_checkpoints=["pufferlib_market/prod_ensemble/s15.pt"],
    )

    assert schema == "legacy_prod"


def test_resolve_daily_feature_schema_uses_rsi_for_v5_checkpoints() -> None:
    schema = resolve_daily_feature_schema(
        "pufferlib_market/checkpoints/stocks12_v5_rsi/tp05_s42/best.pt",
    )

    assert schema == "rsi_v5"


def test_daily_feature_dimension_is_explicit_per_schema() -> None:
    assert daily_feature_dimension("legacy_prod") == 16
    assert daily_feature_dimension("rsi_v5") == 16


def test_resolve_daily_feature_schema_rejects_mixed_ensemble() -> None:
    with pytest.raises(ValueError, match="mixes incompatible feature schemas"):
        resolve_daily_feature_schema(
            "pufferlib_market/prod_ensemble/tp10.pt",
            extra_checkpoints=["pufferlib_market/checkpoints/stocks12_v5_rsi/tp05_s42/best.pt"],
        )


def test_build_signal_uses_schema_aware_feature_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[str] = []

    class _FakeTrader:
        SYMBOLS = ("AAPL",)
        num_symbols = 1
        obs_size = 22
        num_actions = 3
        device = "cpu"
        max_steps = 90

        def get_signal(self, features, prices):
            assert float(features[0, 0]) == 7.0
            return SimpleNamespace(
                action="buy",
                symbol="AAPL",
                direction="long",
                confidence=0.8,
                value_estimate=0.1,
                allocation_pct=1.0,
                level_offset_bps=0.0,
            )

    monkeypatch.setattr(daily_stock, "_load_cached_daily_trader", lambda *args, **kwargs: _FakeTrader())

    def _fake_compute(price_df, *, schema):
        recorded.append(schema)
        return np.full(16, 7.0, dtype=np.float32)

    monkeypatch.setattr(daily_stock, "compute_daily_feature_vector_for_schema", _fake_compute)

    signal, prices = daily_stock.build_signal(
        "pufferlib_market/prod_ensemble/tp10.pt",
        {"AAPL": _sample_frame()},
    )

    assert recorded == ["legacy_prod"]
    assert signal.symbol == "AAPL"
    assert prices == {"AAPL": 103.5}


def test_prepare_daily_backtest_data_uses_schema_aware_feature_history(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[str] = []

    frame = _sample_frame()

    class _FakeTrader:
        SYMBOLS = ("AAPL",)
        obs_size = 22
        num_actions = 3

    monkeypatch.setattr(
        daily_stock,
        "load_local_daily_frames",
        lambda symbols, data_dir, min_days: {"AAPL": frame},
    )
    monkeypatch.setattr(daily_stock, "_load_cached_daily_trader", lambda *args, **kwargs: _FakeTrader())
    monkeypatch.setattr(daily_stock, "_load_bare_policy", lambda *args, **kwargs: object())

    def _fake_build(price_df, *, schema):
        recorded.append(schema)
        return pd.DataFrame(
            np.full((len(price_df), 16), 3.0, dtype=np.float32),
            index=price_df.index,
        )

    monkeypatch.setattr(daily_stock, "build_daily_feature_history_for_schema", _fake_build)

    prepared = daily_stock._prepare_daily_backtest_data(
        checkpoint="pufferlib_market/prod_ensemble/tp10.pt",
        symbols=["AAPL"],
        data_dir="trainingdata",
        days=2,
    )

    assert recorded == ["legacy_prod"]
    assert prepared.feature_schema == "legacy_prod"
    assert prepared.feature_cube.shape == (4, 1, 16)


def _make_ohlcv_with_spike(n: int = 100) -> pd.DataFrame:
    """Return a price DataFrame with a +70% spike at bar 50 to expose clipping bugs."""
    np.random.seed(7)
    close = np.cumprod(1 + np.random.normal(0, 0.01, n)) * 100.0
    close[50] = close[49] * 1.7  # +70% move
    high = close * 1.005
    low = close * 0.995
    volume = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )


def test_rsi_v5_training_inference_feature_parity() -> None:
    """export_data_daily and inference_daily must produce identical feature vectors.

    Bug: inference_daily clipped ret_1d before computing volatility_5d/volatility_20d,
    while export_data_daily used the raw (unclipped) returns.  On extreme-move days
    (±50%+) the volatility features would diverge by up to ~0.08, causing the model
    to see different inputs at inference time vs training time.
    """
    from pufferlib_market.export_data_daily import compute_daily_features as export_features
    from pufferlib_market.inference_daily import compute_daily_features as infer_features

    df = _make_ohlcv_with_spike()

    # export_features returns a DataFrame; infer_features returns a 1-D array
    train_df = export_features(df)
    infer_vec = infer_features(df)

    # Both cover the final bar (the same day)
    train_vec = train_df.iloc[-1].to_numpy(dtype=np.float32)

    assert train_vec.shape == infer_vec.shape == (16,), (
        f"shape mismatch: train={train_vec.shape} infer={infer_vec.shape}"
    )

    max_diff = float(np.abs(train_vec - infer_vec).max())
    assert max_diff < 1e-5, (
        f"training/inference feature mismatch (max diff={max_diff:.6f}): "
        f"train={train_vec}, infer={infer_vec}"
    )


def test_rsi_v5_volatility_uses_unclipped_returns() -> None:
    """volatility_5d must use unclipped daily returns (same as training)."""
    from pufferlib_market.inference_daily import compute_daily_features as infer_features
    from pufferlib_market.export_data_daily import compute_daily_features as export_features

    df = _make_ohlcv_with_spike()

    # On the bar right after the +70% spike (bar 51), both vol features should match.
    train_bar = export_features(df).iloc[51].to_numpy(dtype=np.float32)
    infer_full = infer_features(df[:52])  # feed exactly 52 bars, take the last
    infer_bar = infer_full  # inference returns only the last bar vector

    # vol_5d = index 3, vol_20d = index 4
    assert abs(float(train_bar[3]) - float(infer_bar[3])) < 1e-5, (
        f"volatility_5d mismatch at spike bar: train={train_bar[3]:.6f} infer={infer_bar[3]:.6f}"
    )
    assert abs(float(train_bar[4]) - float(infer_bar[4])) < 1e-5, (
        f"volatility_20d mismatch at spike bar: train={train_bar[4]:.6f} infer={infer_bar[4]:.6f}"
    )
