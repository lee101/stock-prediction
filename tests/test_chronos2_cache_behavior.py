from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, List

import backtest_test3_inline as marketsim
import numpy as np
import pandas as pd
import pytest

from src.models import chronos2_wrapper
from src.models.chronos2_wrapper import Chronos2OHLCWrapper
from src.models.model_cache import ModelCacheManager


def test_load_chronos2_wrapper_reuses_cache_for_equivalent_params(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.reload(marketsim)
    cache = getattr(module, "_chronos2_wrapper_cache", None)
    if cache is None:  # pragma: no cover - defensive guard for partial imports
        pytest.skip("Chronos2 wrapper cache unavailable in module context.")
    cache.clear()
    try:
        created_wrappers: List[Any] = []

        def _fake_loader(cls, **_kwargs):
            sentinel = object()
            created_wrappers.append(sentinel)
            return sentinel

        monkeypatch.setattr(
            Chronos2OHLCWrapper,
            "from_pretrained",
            classmethod(_fake_loader),
        )

        params = {
            "model_id": "amazon/chronos-2",
            "device_map": "cuda",
            "context_length": 512,
            "batch_size": 128,
            "quantile_levels": (0.1, 0.5, 0.9),
        }

        wrapper_one = module.load_chronos2_wrapper(params)
        wrapper_two = module.load_chronos2_wrapper(dict(params))
        assert wrapper_one is wrapper_two
        assert len(created_wrappers) == 1

        jittered = dict(params)
        jittered["quantile_levels"] = (0.1, 0.5 + 1e-13, 0.9)
        wrapper_three = module.load_chronos2_wrapper(jittered)
        assert wrapper_three is wrapper_one
        assert len(created_wrappers) == 1

        monkeypatch.setenv("CHRONOS_COMPILE_BACKEND", "nvfuser")
        wrapper_four = module.load_chronos2_wrapper(dict(params))
        assert wrapper_four is not wrapper_one
        assert len(created_wrappers) == 2
    finally:
        cache.clear()


def test_chronos2_from_pretrained_uses_model_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeModel:
        def save_pretrained(self, target: str, safe_serialization: bool = True) -> None:  # pragma: no cover - exercised
            path = Path(target)
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text("{}", encoding="utf-8")
            (path / "model.safetensors").write_text("data", encoding="utf-8")

        def to(self, **_kwargs):
            return self

    class _FakePipeline:
        load_sources: List[str] = []

        def __init__(self) -> None:
            self.model = _FakeModel()

        @classmethod
        def from_pretrained(cls, model_id: str, **_kwargs):
            cls.load_sources.append(model_id)
            return cls()

    monkeypatch.setattr(chronos2_wrapper, "_Chronos2Pipeline", _FakePipeline)

    manager = ModelCacheManager("chronos2-test", root=tmp_path)
    kwargs = dict(
        model_id="amazon/chronos-2",
        device_map="cpu",
        default_context_length=64,
        default_batch_size=16,
        torch_compile=False,
        cache_policy="prefer",
        cache_manager=manager,
    )

    Chronos2OHLCWrapper.from_pretrained(**kwargs)
    assert _FakePipeline.load_sources == ["amazon/chronos-2"]
    weights_dir = manager.weights_dir("amazon/chronos-2", "fp32")
    assert (weights_dir / "config.json").exists()

    Chronos2OHLCWrapper.from_pretrained(**kwargs)
    assert len(_FakePipeline.load_sources) == 2
    assert _FakePipeline.load_sources[-1] == str(weights_dir)


def _build_context(rows: int = 64) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": index,
            "open": np.linspace(50.0, 80.0, rows, dtype=np.float32),
            "high": np.linspace(60.0, 90.0, rows, dtype=np.float32),
            "low": np.linspace(40.0, 70.0, rows, dtype=np.float32),
            "close": np.linspace(55.0, 85.0, rows, dtype=np.float32),
            "symbol": ["TEST"] * rows,
        }
    )


class _CountingPipeline:
    call_count = 0

    def __init__(self) -> None:
        self.model = object()

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):  # pragma: no cover - simple stub
        return cls()

    def predict_df(self, context_df, **kwargs):  # type: ignore[override]
        type(self).call_count += 1
        prediction_length = int(kwargs.get("prediction_length", 1))
        quantiles = kwargs.get("quantile_levels", [0.5])
        target_cols = kwargs.get("target", ["close"])
        start = pd.to_datetime(context_df["timestamp"].iloc[-1])
        timestamps = pd.date_range(start + pd.Timedelta(days=1), periods=prediction_length, freq="D", tz="UTC")
        rows = []
        for ts in timestamps:
            for target in target_cols:
                payload = {
                    "timestamp": ts,
                    "target_name": target,
                }
                for level in quantiles:
                    payload[format(level, "g")] = float(level) + float(type(self).call_count)
                rows.append(payload)
        return pd.DataFrame(rows)


def test_chronos2_prediction_cache_reuses_results(monkeypatch: pytest.MonkeyPatch) -> None:
    _CountingPipeline.call_count = 0
    monkeypatch.setenv("CHRONOS2_PREDICTION_CACHE", "1")
    monkeypatch.setenv("CHRONOS_COMPILE", "0")
    monkeypatch.setattr(chronos2_wrapper, "_Chronos2Pipeline", _CountingPipeline)

    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id="stub/chronos2",
        device_map="cpu",
        default_context_length=32,
        default_batch_size=8,
        torch_compile=False,
        prediction_cache_enabled=True,
        prediction_cache_size=4,
        prediction_cache_decimals=6,
        cache_policy="never",
    )

    context = _build_context(40)
    first = wrapper.predict_ohlc(context, symbol="TEST", prediction_length=4, batch_size=16)
    assert _CountingPipeline.call_count == 1

    second = wrapper.predict_ohlc(context, symbol="TEST", prediction_length=4, batch_size=2)
    assert _CountingPipeline.call_count == 1
    pd.testing.assert_frame_equal(first.raw_dataframe, second.raw_dataframe)


def test_chronos2_prediction_cache_tolerates_small_float_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    _CountingPipeline.call_count = 0
    monkeypatch.setenv("CHRONOS2_PREDICTION_CACHE", "1")
    monkeypatch.setenv("CHRONOS_COMPILE", "0")
    monkeypatch.setattr(chronos2_wrapper, "_Chronos2Pipeline", _CountingPipeline)

    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id="stub/chronos2",
        device_map="cpu",
        default_context_length=32,
        default_batch_size=4,
        torch_compile=False,
        prediction_cache_enabled=True,
        prediction_cache_size=4,
        prediction_cache_decimals=6,
        cache_policy="never",
    )

    base = _build_context(40)
    wrapper.predict_ohlc(base, symbol="TEST", prediction_length=3)
    assert _CountingPipeline.call_count == 1

    noisy = base.copy()
    noisy.loc[noisy.index[-1], "close"] += 5e-7
    wrapper.predict_ohlc(noisy, symbol="TEST", prediction_length=3)
    assert _CountingPipeline.call_count == 1

    shifted = base.copy()
    shifted.loc[shifted.index[-1], "close"] += 1e-2
    wrapper.predict_ohlc(shifted, symbol="TEST", prediction_length=3)
    assert _CountingPipeline.call_count == 2
