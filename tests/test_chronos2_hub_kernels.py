from __future__ import annotations

from contextlib import nullcontext

from src.models import chronos2_wrapper as wrapper_mod


class _DummyModel:
    def to(self, *, dtype=None):
        return self


class _DummyPipeline:
    def __init__(self):
        self.model = _DummyModel()

    def predict_df(self, *args, **kwargs):
        raise NotImplementedError


class _DummyCacheManager:
    def compilation_env(self, *args, **kwargs):
        return nullcontext()

    def load_metadata(self, *args, **kwargs):
        return None

    def metadata_matches(self, *args, **kwargs):
        return False

    def load_pretrained_path(self, *args, **kwargs):
        return None

    def persist_model_state(self, *args, **kwargs):
        return None


def test_from_pretrained_passes_kernel_config_and_sets_hub_env(monkeypatch):
    captured: dict[str, object] = {}

    class _FakePipelineCls:
        @classmethod
        def from_pretrained(cls, model_id, *, device_map=None, **kwargs):
            captured["model_id"] = model_id
            captured["device_map"] = device_map
            captured["use_hub_kernels_env"] = wrapper_mod.os.getenv("USE_HUB_KERNELS")
            captured["kwargs"] = kwargs
            return _DummyPipeline()

    monkeypatch.setattr(wrapper_mod, "_require_chronos_pipeline", lambda: _FakePipelineCls)

    wrapper = wrapper_mod.Chronos2OHLCWrapper.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        cache_policy="never",
        use_hub_kernels=True,
        hub_kernel_config={"RMSNorm": "kernels-community/liger_kernels:LigerRMSNorm"},
        cache_manager=_DummyCacheManager(),
    )

    assert isinstance(wrapper, wrapper_mod.Chronos2OHLCWrapper)
    assert captured["model_id"] == "amazon/chronos-2"
    assert captured["device_map"] == "cpu"
    assert captured["use_hub_kernels_env"] == "1"
    kernel_config = captured["kwargs"]["kernel_config"]
    assert kernel_config.kernel_mapping == {"RMSNorm": "kernels-community/liger_kernels:LigerRMSNorm"}
    assert wrapper_mod.os.getenv("USE_HUB_KERNELS") is None


def test_from_pretrained_temporarily_disables_hub_env_when_explicitly_off(monkeypatch):
    captured: dict[str, object] = {}

    class _FakePipelineCls:
        @classmethod
        def from_pretrained(cls, model_id, *, device_map=None, **kwargs):
            captured["use_hub_kernels_env"] = wrapper_mod.os.getenv("USE_HUB_KERNELS")
            return _DummyPipeline()

    monkeypatch.setattr(wrapper_mod, "_require_chronos_pipeline", lambda: _FakePipelineCls)
    monkeypatch.setenv("USE_HUB_KERNELS", "1")

    wrapper_mod.Chronos2OHLCWrapper.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        cache_policy="never",
        use_hub_kernels=False,
        cache_manager=_DummyCacheManager(),
    )

    assert captured["use_hub_kernels_env"] == "0"
    assert wrapper_mod.os.getenv("USE_HUB_KERNELS") == "1"
