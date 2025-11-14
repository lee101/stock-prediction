from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, List

import backtest_test3_inline as marketsim
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
