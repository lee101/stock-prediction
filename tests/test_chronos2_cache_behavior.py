from __future__ import annotations

import importlib
import sys
import types
from contextlib import nullcontext
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
    try:
        module = importlib.reload(marketsim)
    except ModuleNotFoundError:
        module = importlib.import_module(marketsim.__name__)
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

        def predict_df(self, context_df, **_kwargs):  # type: ignore[no-untyped-def]
            return context_df

    monkeypatch.setattr(chronos2_wrapper, "_ChronosBasePipeline", _FakePipeline)

    manager = ModelCacheManager("chronos2-test", root=tmp_path)
    kwargs = dict(
        model_id="amazon/chronos-t5-small",
        device_map="cpu",
        default_context_length=64,
        default_batch_size=16,
        torch_compile=False,
        cache_policy="prefer",
        cache_manager=manager,
    )

    Chronos2OHLCWrapper.from_pretrained(**kwargs)
    assert _FakePipeline.load_sources == ["amazon/chronos-t5-small"]
    weights_dir = manager.weights_dir("amazon/chronos-t5-small", "fp32")
    assert (weights_dir / "config.json").exists()

    Chronos2OHLCWrapper.from_pretrained(**kwargs)
    assert len(_FakePipeline.load_sources) == 2
    assert _FakePipeline.load_sources[-1] == str(weights_dir)


def test_chronos2_from_pretrained_persists_eager_model_when_compile_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eager_model = object()
    compiled_model = object()
    captured: dict[str, Any] = {}

    class _FakePipeline:
        def __init__(self) -> None:
            self.model = eager_model

        @classmethod
        def from_pretrained(cls, _model_id: str, **_kwargs):
            return cls()

        def predict_df(self, context_df, **_kwargs):  # type: ignore[no-untyped-def]
            return context_df

    class _RecordingManager:
        def compilation_env(self, *_args, **_kwargs):
            return nullcontext()

        def load_metadata(self, *_args, **_kwargs):
            return None

        def metadata_matches(self, *_args, **_kwargs) -> bool:
            return False

        def load_pretrained_path(self, *_args, **_kwargs):
            return None

        def persist_model_state(self, *, model_id, dtype_token, model, metadata, force):  # type: ignore[no-untyped-def]
            captured["model_id"] = model_id
            captured["dtype_token"] = dtype_token
            captured["model"] = model
            captured["metadata"] = metadata
            captured["force"] = force

    monkeypatch.setattr(chronos2_wrapper, "_ChronosBasePipeline", _FakePipeline)
    monkeypatch.setattr(chronos2_wrapper.torch, "compile", lambda model, **_kwargs: compiled_model)

    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id="amazon/chronos-t5-small",
        device_map="cpu",
        default_context_length=64,
        default_batch_size=16,
        torch_compile=True,
        cache_policy="prefer",
        cache_manager=_RecordingManager(),
    )

    assert wrapper.pipeline.model is compiled_model
    assert wrapper._eager_model is eager_model
    assert captured["model"] is eager_model
    assert captured["model"] is not compiled_model


def test_chronos2_from_pretrained_supports_cutechronos_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCutePipeline:
        load_kwargs: dict[str, Any] | None = None

        def __init__(self) -> None:
            self.model = object()

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):  # type: ignore[no-untyped-def]
            cls.load_kwargs = {"model_id": model_id, **kwargs}
            return cls()

        def predict_quantiles(self, inputs, prediction_length=None, quantile_levels=None, limit_prediction_length=False):
            horizon = int(prediction_length or 1)
            levels = list(quantile_levels or [0.5])
            quantiles = [np.zeros((1, horizon, len(levels)), dtype=np.float32) for _ in inputs]
            means = [np.zeros((1, horizon), dtype=np.float32) for _ in inputs]
            return quantiles, means

    cute_pkg = types.ModuleType("cutechronos")
    cute_pipeline_mod = types.ModuleType("cutechronos.pipeline")
    cute_pipeline_mod.CuteChronos2Pipeline = _FakeCutePipeline
    cute_pkg.pipeline = cute_pipeline_mod
    monkeypatch.setitem(sys.modules, "cutechronos", cute_pkg)
    monkeypatch.setitem(sys.modules, "cutechronos.pipeline", cute_pipeline_mod)

    monkeypatch.setenv("CHRONOS2_PIPELINE_BACKEND", "cutechronos")
    monkeypatch.setenv("CHRONOS_COMPILE", "1")

    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id="stub/chronos2",
        device_map="cpu",
        default_context_length=32,
        default_batch_size=8,
        torch_compile=True,
        compile_mode="reduce-overhead",
        cache_policy="never",
    )

    assert _FakeCutePipeline.load_kwargs is not None
    assert _FakeCutePipeline.load_kwargs["model_id"] == "stub/chronos2"
    assert _FakeCutePipeline.load_kwargs["device"] == "cpu"
    assert _FakeCutePipeline.load_kwargs["use_cute"] is True
    assert _FakeCutePipeline.load_kwargs["compile_mode"] == "reduce-overhead"
    assert wrapper._torch_compile_enabled is False


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
    monkeypatch.setattr(chronos2_wrapper, "_ChronosBasePipeline", _CountingPipeline)

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
    monkeypatch.setattr(chronos2_wrapper, "_ChronosBasePipeline", _CountingPipeline)

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


class _NaNPredictDfPipeline:
    def __init__(self) -> None:
        self.model = object()

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):  # pragma: no cover - simple stub
        return cls()

    def predict_df(self, context_df, **kwargs):  # type: ignore[override]
        prediction_length = int(kwargs.get("prediction_length", 1))
        quantiles = kwargs.get("quantile_levels", [0.5])
        target_cols = kwargs.get("target", ["close"])
        start = pd.to_datetime(context_df["timestamp"].iloc[-1], utc=True)
        timestamps = pd.date_range(start + pd.Timedelta(days=1), periods=prediction_length, freq="D", tz="UTC")
        rows = []
        for ts in timestamps:
            for target in target_cols:
                payload = {
                    "timestamp": ts,
                    "target_name": target,
                    "predictions": np.nan,
                }
                for level in quantiles:
                    payload[format(level, "g")] = np.nan
                rows.append(payload)
        return pd.DataFrame(rows)


class _TensorOnlyPipeline:
    def __init__(self) -> None:
        self.model = object()

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):  # pragma: no cover - simple stub
        return cls()

    def predict_quantiles(self, inputs, prediction_length=None, quantile_levels=None, limit_prediction_length=False):
        horizon = int(prediction_length or 1)
        levels = list(quantile_levels or [0.5])
        quantiles = []
        means = []
        for idx, _series in enumerate(inputs):
            base = 100.0 + idx
            q = np.zeros((1, horizon, len(levels)), dtype=np.float32)
            for step in range(horizon):
                for q_idx, level in enumerate(levels):
                    q[0, step, q_idx] = base + step + float(level)
            m = np.array([[base + step + 0.5 for step in range(horizon)]], dtype=np.float32)
            quantiles.append(q)
            means.append(m)
        return quantiles, means


class _CompileOnlyNaNPipeline:
    def __init__(self) -> None:
        self.eager_model = object()
        self.compiled_model = object()
        self.model = self.eager_model

    def predict_df(self, context_df, **kwargs):  # type: ignore[override]
        prediction_length = int(kwargs.get("prediction_length", 1))
        quantiles = kwargs.get("quantile_levels", [0.5])
        target_cols = kwargs.get("target", ["close"])
        start = pd.to_datetime(context_df["timestamp"].iloc[-1], utc=True)
        timestamps = pd.date_range(start + pd.Timedelta(days=1), periods=prediction_length, freq="D", tz="UTC")
        rows = []
        emit_nan = self.model is self.compiled_model
        for ts in timestamps:
            for target in target_cols:
                payload = {
                    "timestamp": ts,
                    "target_name": target,
                    "predictions": np.nan if emit_nan else 101.5,
                }
                for level in quantiles:
                    payload[format(level, "g")] = np.nan if emit_nan else 100.0 + float(level)
                rows.append(payload)
        return pd.DataFrame(rows)


class _NaNPredictDfButFiniteTensorPipeline:
    def __init__(self) -> None:
        self.model = object()

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):  # pragma: no cover - simple stub
        return cls()

    def predict_df(self, context_df, **kwargs):  # type: ignore[override]
        prediction_length = int(kwargs.get("prediction_length", 1))
        quantiles = kwargs.get("quantile_levels", [0.5])
        target_cols = kwargs.get("target", ["close"])
        start = pd.to_datetime(context_df["timestamp"].iloc[-1], utc=True)
        timestamps = pd.date_range(start + pd.Timedelta(days=1), periods=prediction_length, freq="D", tz="UTC")
        rows = []
        for ts in timestamps:
            for target in target_cols:
                payload = {
                    "timestamp": ts,
                    "target_name": target,
                    "predictions": np.nan,
                }
                for level in quantiles:
                    payload[format(level, "g")] = np.nan
                rows.append(payload)
        return pd.DataFrame(rows)

    def predict_quantiles(self, inputs, prediction_length=None, quantile_levels=None, limit_prediction_length=False):
        horizon = int(prediction_length or 1)
        levels = list(quantile_levels or [0.5])
        quantiles = []
        means = []
        for idx, _series in enumerate(inputs):
            base = 200.0 + idx
            q = np.zeros((1, horizon, len(levels)), dtype=np.float32)
            for step in range(horizon):
                for q_idx, level in enumerate(levels):
                    q[0, step, q_idx] = base + step + float(level)
            m = np.array([[base + step + 0.5 for step in range(horizon)]], dtype=np.float32)
            quantiles.append(q)
            means.append(m)
        return quantiles, means


class _ReloadablePrimaryFallbackPipeline:
    load_count = 0

    def __init__(self, *, finite: bool) -> None:
        self.model = object()
        self._finite = finite

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):  # pragma: no cover - simple stub
        cls.load_count += 1
        return cls(finite=cls.load_count > 1)

    def predict_df(self, context_df, **kwargs):  # type: ignore[override]
        prediction_length = int(kwargs.get("prediction_length", 1))
        quantiles = kwargs.get("quantile_levels", [0.5])
        target_cols = kwargs.get("target", ["close"])
        start = pd.to_datetime(context_df["timestamp"].iloc[-1], utc=True)
        timestamps = pd.date_range(start + pd.Timedelta(days=1), periods=prediction_length, freq="D", tz="UTC")
        rows = []
        for step, ts in enumerate(timestamps):
            for idx, target in enumerate(target_cols):
                base = 300.0 + idx + step
                payload = {
                    "timestamp": ts,
                    "target_name": target,
                    "predictions": base + 0.5 if self._finite else np.nan,
                }
                for level in quantiles:
                    payload[format(level, "g")] = base + float(level) if self._finite else np.nan
                rows.append(payload)
        return pd.DataFrame(rows)

    def predict_quantiles(self, inputs, prediction_length=None, quantile_levels=None, limit_prediction_length=False):
        horizon = int(prediction_length or 1)
        levels = list(quantile_levels or [0.5])
        quantiles = []
        means = []
        for idx, _series in enumerate(inputs):
            base = 300.0 + idx
            if self._finite:
                q = np.zeros((1, horizon, len(levels)), dtype=np.float32)
                for step in range(horizon):
                    for q_idx, level in enumerate(levels):
                        q[0, step, q_idx] = base + step + float(level)
                m = np.array([[base + step + 0.5 for step in range(horizon)]], dtype=np.float32)
            else:
                q = np.full((1, horizon, len(levels)), np.nan, dtype=np.float32)
                m = np.full((1, horizon), np.nan, dtype=np.float32)
            quantiles.append(q)
            means.append(m)
        return quantiles, means


class _MovableModel:
    def __init__(self) -> None:
        self.moves: List[str] = []

    def to(self, device: str):
        self.moves.append(str(device))
        return self


class _MovablePipeline:
    def __init__(self) -> None:
        self.model = _MovableModel()

    def predict_quantiles(self, inputs, prediction_length=None, quantile_levels=None, limit_prediction_length=False):
        horizon = int(prediction_length or 1)
        levels = list(quantile_levels or [0.5])
        quantiles = [np.zeros((1, horizon, len(levels)), dtype=np.float32) for _ in inputs]
        means = [np.zeros((1, horizon), dtype=np.float32) for _ in inputs]
        return quantiles, means


def test_chronos2_prediction_falls_back_from_nonfinite_cpu_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    fallback_loads: List[dict[str, Any]] = []

    def _fake_cute_loader(**kwargs):  # type: ignore[no-untyped-def]
        fallback_loads.append(dict(kwargs))
        return _TensorOnlyPipeline()

    monkeypatch.setenv("CHRONOS_COMPILE", "0")
    monkeypatch.setattr(chronos2_wrapper, "_ChronosBasePipeline", _NaNPredictDfPipeline)
    monkeypatch.setattr(chronos2_wrapper, "_load_cutechronos_pipeline", _fake_cute_loader)

    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id="stub/chronos2",
        device_map="cpu",
        default_context_length=32,
        default_batch_size=8,
        torch_compile=False,
        cache_policy="never",
    )

    result = wrapper.predict_ohlc(_build_context(40), symbol="TEST", prediction_length=3, batch_size=16)

    assert len(fallback_loads) == 1
    assert fallback_loads[0]["model_id"] == "stub/chronos2"
    assert fallback_loads[0]["torch_compile"] is False
    assert fallback_loads[0]["compile_mode"] is None
    assert np.isfinite(result.median.to_numpy()).all()
    assert result.median["close"].tolist() == pytest.approx([103.5, 104.5, 105.5])


def test_chronos2_prediction_supports_tensor_only_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRONOS_COMPILE", "0")
    monkeypatch.setattr(chronos2_wrapper, "_load_cutechronos_pipeline", lambda **_kwargs: _TensorOnlyPipeline())

    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id="stub/chronos2",
        device_map="cpu",
        default_context_length=32,
        default_batch_size=8,
        torch_compile=False,
        pipeline_backend="cutechronos",
        cache_policy="never",
    )

    result = wrapper.predict_ohlc(_build_context(40), symbol="TEST", prediction_length=2)

    assert np.isfinite(result.median.to_numpy()).all()
    assert result.median["open"].tolist() == pytest.approx([100.5, 101.5])
    assert result.median["close"].tolist() == pytest.approx([103.5, 104.5])


def test_chronos2_prediction_retries_eager_before_safe_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _CompileOnlyNaNPipeline()
    wrapper = Chronos2OHLCWrapper(
        pipeline,
        device_hint="cpu",
        default_context_length=32,
        default_batch_size=8,
        torch_compile=False,
    )
    wrapper._torch_compile_enabled = True
    wrapper._torch_compile_success = True
    wrapper._eager_model = pipeline.eager_model
    wrapper._resolved_model_id = "stub/chronos2"
    pipeline.model = pipeline.compiled_model

    monkeypatch.setattr(
        chronos2_wrapper,
        "_load_cutechronos_pipeline",
        lambda **_kwargs: pytest.fail("safe fallback should not be used when eager retry succeeds"),
    )

    result = wrapper.predict_ohlc(_build_context(40), symbol="TEST", prediction_length=2)

    assert wrapper._torch_compile_success is False
    assert pipeline.model is pipeline.eager_model
    assert np.isfinite(result.median.to_numpy()).all()
    assert result.median["close"].tolist() == pytest.approx([100.5, 100.5])


def test_chronos2_prediction_prefers_primary_tensor_fallback_before_safe_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHRONOS_COMPILE", "0")
    monkeypatch.setattr(chronos2_wrapper, "_ChronosBasePipeline", _NaNPredictDfButFiniteTensorPipeline)
    monkeypatch.setattr(
        chronos2_wrapper,
        "_load_cutechronos_pipeline",
        lambda **_kwargs: pytest.fail("safe cutechronos backend should not be used when primary tensor fallback succeeds"),
    )

    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id="stub/chronos2",
        device_map="cpu",
        default_context_length=32,
        default_batch_size=8,
        torch_compile=False,
        cache_policy="never",
    )

    result = wrapper.predict_ohlc(_build_context(40), symbol="TEST", prediction_length=3, batch_size=16)

    assert np.isfinite(result.median.to_numpy()).all()
    assert result.median["close"].tolist() == pytest.approx([203.5, 204.5, 205.5])


def test_chronos2_prediction_reloads_primary_backend_before_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ReloadablePrimaryFallbackPipeline.load_count = 0
    monkeypatch.setenv("CHRONOS_COMPILE", "0")
    monkeypatch.setattr(chronos2_wrapper, "_ChronosBasePipeline", _ReloadablePrimaryFallbackPipeline)
    monkeypatch.setattr(
        chronos2_wrapper,
        "_load_cutechronos_pipeline",
        lambda **_kwargs: pytest.fail("safe cutechronos backend should not be used when eager Chronos reload succeeds"),
    )

    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id="stub/chronos2",
        device_map="cpu",
        default_context_length=32,
        default_batch_size=8,
        torch_compile=False,
        cache_policy="never",
    )

    result = wrapper.predict_ohlc(_build_context(40), symbol="TEST", prediction_length=3, batch_size=16)

    assert _ReloadablePrimaryFallbackPipeline.load_count == 2
    assert np.isfinite(result.median.to_numpy()).all()
    assert result.median["close"].tolist() == pytest.approx([303.5, 304.5, 305.5])


def test_chronos2_unload_offloads_primary_and_safe_fallback_pipelines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chronos2_wrapper, "gpu_should_offload_to_cpu", lambda _device: True)

    primary = _MovablePipeline()
    eager_fallback = _MovablePipeline()
    safe_fallback = _MovablePipeline()
    wrapper = Chronos2OHLCWrapper(
        primary,
        device_hint="cuda",
        default_context_length=32,
        default_batch_size=8,
        torch_compile=False,
    )
    wrapper._primary_eager_prediction_pipeline = eager_fallback
    wrapper._safe_prediction_pipeline = safe_fallback

    wrapper.unload()

    assert primary.model.moves == ["cpu"]
    assert eager_fallback.model.moves == ["cpu"]
    assert safe_fallback.model.moves == ["cpu"]
    assert wrapper.pipeline is None
    assert wrapper._primary_eager_prediction_pipeline is None
    assert wrapper._safe_prediction_pipeline is None
