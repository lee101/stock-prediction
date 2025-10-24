from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_backtest_module_from_path():
    module_path = _REPO_ROOT / "backtest_test3_inline.py"
    root_str = str(_REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    spec = importlib.util.spec_from_file_location("backtest_test3_inline", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load backtest_test3_inline from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["backtest_test3_inline"] = module
    spec.loader.exec_module(module)
    return module


def _fresh_module():
    try:
        base_module = importlib.import_module("backtest_test3_inline")
    except ModuleNotFoundError:
        module = _load_backtest_module_from_path()
    else:
        try:
            module = importlib.reload(base_module)
        except ModuleNotFoundError:
            importlib.invalidate_caches()
            module = _load_backtest_module_from_path()
    # Ensure globals start from a clean state even if cache clearing helpers are added later.
    if hasattr(module, "_reset_model_caches"):
        module._reset_model_caches()
    else:  # pragma: no cover - exercised pre-implementation
        reason = getattr(module, "__import_error__", None)
        pytest.skip(f"backtest_test3_inline unavailable: {reason!r}")
    return module


def test_resolve_toto_params_cached(monkeypatch):
    module = _fresh_module()
    call_count = {"value": 0}
    record = SimpleNamespace(config={"num_samples": 11, "samples_per_batch": 7, "aggregate": "median"})

    def fake_load_best_config(model: str, symbol: str):
        assert model == "toto"
        assert symbol == "ETHUSD"
        call_count["value"] += 1
        return record

    monkeypatch.setattr(module, "load_best_config", fake_load_best_config)

    params_first = module.resolve_toto_params("ETHUSD")
    params_second = module.resolve_toto_params("ETHUSD")

    expected = {
        "num_samples": module.TOTO_MIN_NUM_SAMPLES,
        "samples_per_batch": module.TOTO_MIN_SAMPLES_PER_BATCH,
        "aggregate": "median",
    }
    assert params_first == params_second == expected
    assert call_count["value"] == 1


def test_resolve_kronos_params_cached(monkeypatch):
    module = _fresh_module()
    call_count = {"value": 0}
    record = SimpleNamespace(
        config={
            "temperature": 0.2,
            "top_p": 0.85,
            "top_k": 42,
            "sample_count": 256,
            "max_context": 320,
            "clip": 1.7,
        }
    )

    def fake_load_best_config(model: str, symbol: str):
        assert model == "kronos"
        assert symbol == "ETHUSD"
        call_count["value"] += 1
        return record

    monkeypatch.setattr(module, "load_best_config", fake_load_best_config)

    params_first = module.resolve_kronos_params("ETHUSD")
    params_second = module.resolve_kronos_params("ETHUSD")

    assert params_first == params_second == {
        "temperature": 0.2,
        "top_p": 0.85,
        "top_k": 42,
        "sample_count": 256,
        "max_context": 320,
        "clip": 1.7,
    }
    assert call_count["value"] == 1


def test_resolve_best_model_cached(monkeypatch):
    module = _fresh_module()
    call_count = {"value": 0}

    def fake_load_model_selection(symbol: str):
        assert symbol == "ETHUSD"
        call_count["value"] += 1
        return {"model": "toto"}

    monkeypatch.delenv("MARKETSIM_FORCE_KRONOS", raising=False)
    monkeypatch.setattr(module, "_in_test_mode", lambda: False)
    monkeypatch.setattr(module, "load_model_selection", fake_load_model_selection)

    assert module.resolve_best_model("ETHUSD") == "toto"
    assert module.resolve_best_model("ETHUSD") == "toto"
    assert call_count["value"] == 1


def test_load_kronos_unloads_toto_pipeline(monkeypatch):
    module = _fresh_module()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)

    class DummyPipeline:
        def __init__(self):
            self.model = SimpleNamespace(to=lambda *a, **k: None)

    pipeline_obj = DummyPipeline()

    def fake_from_pretrained(cls, *args, **kwargs):
        return pipeline_obj

    monkeypatch.setattr(module.TotoPipeline, "from_pretrained", classmethod(fake_from_pretrained))

    class DummyWrapper:
        def __init__(self, *args, **kwargs):
            self.unloaded = False

    monkeypatch.setattr(module, "KronosForecastingWrapper", DummyWrapper)

    module.pipeline = None
    module.kronos_wrapper_cache.clear()

    module.load_toto_pipeline()
    assert module.pipeline is pipeline_obj

    params = {
        "temperature": 0.15,
        "top_p": 0.9,
        "top_k": 32,
        "sample_count": 192,
        "max_context": 256,
        "clip": 1.8,
    }

    module.load_kronos_wrapper(params)
    assert module.pipeline is None


def test_load_toto_clears_kronos_cache(monkeypatch):
    module = _fresh_module()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)

    class DummyWrapper:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(module, "KronosForecastingWrapper", DummyWrapper)

    params = {
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 16,
        "sample_count": 128,
        "max_context": 224,
        "clip": 1.5,
    }

    module.load_kronos_wrapper(params)
    assert module.kronos_wrapper_cache  # cache populated

    class DummyPipeline:
        def __init__(self):
            self.model = SimpleNamespace(to=lambda *a, **k: None)

    dummy_pipeline = DummyPipeline()

    def fake_from_pretrained(cls, *args, **kwargs):
        return dummy_pipeline

    monkeypatch.setattr(module.TotoPipeline, "from_pretrained", classmethod(fake_from_pretrained))

    module.load_toto_pipeline()
    assert module.pipeline is dummy_pipeline
    assert module.kronos_wrapper_cache == {}


def test_require_cuda_raises_without_fallback(monkeypatch):
    module = _fresh_module()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("MARKETSIM_ALLOW_CPU_FALLBACK", raising=False)

    with pytest.raises(RuntimeError, match="requires a CUDA-capable GPU"):
        module._require_cuda("Toto forecasting", symbol="ETHUSD")


def test_require_cuda_warns_when_fallback_enabled(monkeypatch, caplog):
    module = _fresh_module()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("MARKETSIM_ALLOW_CPU_FALLBACK", "1")

    with caplog.at_level("WARNING"):
        module._require_cuda("Toto forecasting", symbol="ETHUSD")

    assert ("Toto forecasting", "ETHUSD") in module._cpu_fallback_log_state
