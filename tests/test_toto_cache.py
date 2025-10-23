from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import toto_wrapper as tw


class DummyForecast:
    def __init__(self, dtype: torch.dtype):
        self.samples = torch.zeros((1, 4, 1), dtype=dtype)


class DummyForecaster:
    def __init__(self, model):
        self.model = model

    def forecast(
        self,
        inputs,
        *,
        prediction_length: int,
        num_samples: int,
        samples_per_batch: int,
        **_: object,
    ):
        return DummyForecast(dtype=self.model._dtype)


class DummyMaskedTimeseries:
    def __init__(self, **_: object) -> None:
        pass


class DummyToto:
    calls: list[str] = []

    def __init__(self) -> None:
        self._dtype = torch.float32
        self.model = SimpleNamespace()
        self.model.model = self

    @classmethod
    def from_pretrained(cls, model_id: str, **_: object) -> "DummyToto":
        cls.calls.append(model_id)
        inst = cls()
        inst._source = model_id
        return inst

    def to(self, *, device=None, dtype=None):  # type: ignore[override]
        if dtype is not None:
            self._dtype = dtype
        self._device = device
        return self

    def eval(self):
        return self

    def parameters(self):
        yield torch.zeros(())

    def compile(self, mode=None):  # type: ignore[override]
        self._compile_mode = mode
        return self

    def save_pretrained(self, directory: str, safe_serialization: bool = True):  # type: ignore[override]
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        # Sentinel file so the directory is never empty.
        (path / "model.safetensors").write_bytes(b"")

    def state_dict(self):
        return {}


@pytest.fixture(autouse=True)
def _patch_toto(monkeypatch, tmp_path):
    os.environ["COMPILED_MODELS_DIR"] = str(tmp_path)
    monkeypatch.setattr(tw, "_IMPORT_ERROR", None)
    monkeypatch.setattr(tw, "Toto", DummyToto)
    monkeypatch.setattr(tw, "TotoForecaster", DummyForecaster)
    monkeypatch.setattr(tw, "MaskedTimeseries", DummyMaskedTimeseries)
    DummyToto.calls.clear()
    yield
    os.environ.pop("COMPILED_MODELS_DIR", None)


def test_toto_pipeline_persists_and_reuses_compiled_cache(tmp_path):
    pipeline = tw.TotoPipeline.from_pretrained(
        model_id="Fake/Model",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        compile_model=False,
        torch_compile=False,
        warmup_sequence=0,
    )
    assert pipeline.model is not None
    assert DummyToto.calls[0] == "Fake/Model"

    cache_root = Path(os.environ["COMPILED_MODELS_DIR"]) / "toto" / "Fake-Model" / "bf16" / "cpu"
    metadata_path = cache_root / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_id"] == "Fake/Model"
    assert metadata["dtype"] == "bf16"
    assert metadata["device"] == "cpu"
    assert metadata["device_variant"] == "cpu"
    assert metadata["device_requested"] == "cpu"

    # Second load should reuse local cache path.
    _ = tw.TotoPipeline.from_pretrained(
        model_id="Fake/Model",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        compile_model=False,
        torch_compile=False,
        warmup_sequence=0,
    )
    assert len(DummyToto.calls) >= 2
    second_call = DummyToto.calls[1]
    assert second_call.startswith(str(cache_root / "weights"))


def test_toto_cache_separates_device_variants(tmp_path):
    cpu_pipeline = tw.TotoPipeline.from_pretrained(
        model_id="Fake/Model",
        device_map="cpu",
        torch_dtype=torch.float32,
        compile_model=False,
        torch_compile=False,
        warmup_sequence=0,
    )
    assert cpu_pipeline.model is not None

    cpu_cache = Path(os.environ["COMPILED_MODELS_DIR"]) / "toto" / "Fake-Model" / "fp32" / "cpu"
    assert (cpu_cache / "metadata.json").exists()

    gpu_first = tw.TotoPipeline.from_pretrained(
        model_id="Fake/Model",
        device_map="cuda:0",
        torch_dtype=torch.float32,
        compile_model=False,
        torch_compile=False,
        warmup_sequence=0,
    )
    assert gpu_first.model is not None

    calls_after_first_gpu = list(DummyToto.calls)

    gpu_second = tw.TotoPipeline.from_pretrained(
        model_id="Fake/Model",
        device_map="cuda:0",
        torch_dtype=torch.float32,
        compile_model=False,
        torch_compile=False,
        warmup_sequence=0,
    )
    assert gpu_second.model is not None

    cache_root = Path(os.environ["COMPILED_MODELS_DIR"]) / "toto" / "Fake-Model" / "fp32"
    cpu_path = cache_root / "cpu"
    gpu_path = cache_root / "cuda"

    assert cpu_path.exists()
    assert gpu_path.exists()
    assert cpu_path != gpu_path

    new_calls = DummyToto.calls[len(calls_after_first_gpu) :]
    assert new_calls
    assert all(call.startswith(str(gpu_path / "weights")) for call in new_calls)

    gpu_metadata = json.loads((gpu_path / "metadata.json").read_text(encoding="utf-8"))
    assert gpu_metadata["device"] == "cuda"
    assert gpu_metadata["device_requested"] == "cuda:0"


def test_toto_cache_policy_only_requires_existing_cache(tmp_path):
    cache_dir = Path(os.environ["COMPILED_MODELS_DIR"])
    assert not (cache_dir / "toto").exists()
    with pytest.raises(RuntimeError):
        tw.TotoPipeline.from_pretrained(
            model_id="Fake/Model",
            device_map="cpu",
            torch_dtype=torch.float32,
            compile_model=False,
            torch_compile=False,
            warmup_sequence=0,
            cache_policy="only",
        )
