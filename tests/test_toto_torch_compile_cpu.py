from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import toto_wrapper as tw


class _ForecastResult:
    def __init__(self, samples: torch.Tensor):
        self.samples = samples


class _DummyMaskedTimeseries:
    def __init__(self, *, series, **_: object) -> None:
        self.series = series


class _CompiledCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor([[0.05, 0.08, -0.02, 0.03]], dtype=torch.float32)
            )
            self.linear.bias.zero_()

    def forward(self, series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tail = series[..., -4:]
        slope = self.linear(tail)
        last = tail[..., -1:]
        return last, slope


class _DummyToto:
    calls: list[str] = []

    def __init__(self) -> None:
        self._dtype = torch.float32
        self._device = "cpu"
        self._core = _CompiledCore()
        self.model = self._core

    @classmethod
    def from_pretrained(cls, model_id: str, **_: object) -> "_DummyToto":
        cls.calls.append(model_id)
        inst = cls()
        inst._source = model_id
        return inst

    def to(self, *, device=None, dtype=None):  # type: ignore[override]
        if device is not None:
            self._device = str(device)
        target_kwargs = {}
        if device is not None:
            target_kwargs["device"] = device
        if dtype is not None:
            self._dtype = dtype
            target_kwargs["dtype"] = dtype
        if target_kwargs:
            self._core = self._core.to(**target_kwargs)
        self.model = self._core
        return self

    def eval(self):
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self

    def parameters(self):
        yield torch.zeros((), dtype=self._dtype)

    def compile(self, mode=None):  # type: ignore[override]
        self._compile_mode = mode
        return self

    def save_pretrained(self, directory: str, safe_serialization: bool = True):  # type: ignore[override]
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"")

    def state_dict(self):
        return {}


class _DummyForecaster:
    def __init__(self, compiled_core):
        self._core = compiled_core
        self.invocations = 0

    def forecast(
        self,
        inputs,
        *,
        prediction_length: int,
        num_samples: int,
        samples_per_batch: int,
        **_: object,
    ):
        self.invocations += 1
        series = inputs.series
        last, slope = self._core(series)
        steps = torch.arange(
            1,
            prediction_length + 1,
            device=series.device,
            dtype=series.dtype,
        ).view(1, -1)
        trajectory = last + slope * steps
        samples = trajectory.unsqueeze(-1).repeat(1, 1, num_samples)
        return _ForecastResult(samples.unsqueeze(1))


@pytest.fixture(autouse=True)
def _patch_toto(monkeypatch, tmp_path):
    os.environ["COMPILED_MODELS_DIR"] = str(tmp_path)
    monkeypatch.setattr(tw, "_IMPORT_ERROR", None)
    monkeypatch.setattr(tw, "Toto", _DummyToto)
    monkeypatch.setattr(tw, "TotoForecaster", _DummyForecaster)
    monkeypatch.setattr(tw, "MaskedTimeseries", _DummyMaskedTimeseries)
    _DummyToto.calls.clear()
    yield
    os.environ.pop("COMPILED_MODELS_DIR", None)


def _expected_forecast(context: torch.Tensor, prediction_length: int, num_samples: int) -> np.ndarray:
    tail = context[..., -4:]
    slope = (
        tail * torch.tensor([0.05, 0.08, -0.02, 0.03], dtype=context.dtype)
    ).sum(dim=-1, keepdim=True)
    last = tail[..., -1:]
    steps = torch.arange(1, prediction_length + 1, dtype=context.dtype).view(1, -1)
    trajectory = last + slope * steps
    samples = trajectory.unsqueeze(-1).repeat(1, 1, num_samples)
    return samples.numpy()


def test_toto_torch_compile_cpu_executes_inference(tmp_path):
    pipeline = tw.TotoPipeline.from_pretrained(
        model_id="Fake/Compiled",
        device_map="cpu",
        torch_dtype=torch.float32,
        compile_model=False,
        torch_compile=True,
        warmup_sequence=0,
    )

    assert pipeline._torch_compile_success is True
    assert pipeline.compiled is True

    context = torch.tensor(
        [
            [10.0, 11.0, 12.5, 13.0, 14.0, 15.0],
            [5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
        ],
        dtype=torch.float32,
    )
    prediction_length = 3
    num_samples = 2

    forecasts = pipeline.predict(
        context=context,
        prediction_length=prediction_length,
        num_samples=num_samples,
        samples_per_batch=2,
    )

    assert len(forecasts) == context.shape[0]
    for idx, forecast in enumerate(forecasts):
        actual = forecast.numpy()
        expected = _expected_forecast(context[idx : idx + 1], prediction_length, num_samples)
        expected = expected.squeeze(0).T
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    cache_root = Path(os.environ["COMPILED_MODELS_DIR"]) / "toto" / "Fake-Compiled" / "fp32" / "cpu"
    assert (cache_root / "metadata.json").exists()
    metadata = json.loads((cache_root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dtype"] == "fp32"
    assert metadata["dtype_requested"] == "fp32"
    assert metadata["torch_compile"] is True
    assert metadata["device"] == "cpu"

    # Ensure compiled forecaster reused the cached callable.
    assert pipeline.forecaster.invocations == 1
