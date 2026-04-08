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
    assert prepared.feature_cube.shape == (4, 1, 16)
