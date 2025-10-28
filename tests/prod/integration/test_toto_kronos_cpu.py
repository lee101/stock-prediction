"""
Validate device admission policies: Toto must preserve CPU fallback while Kronos remains GPU-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.kronos_wrapper import KronosForecastingWrapper, setup_kronos_wrapper_imports
from src.models.toto_wrapper import TotoPipeline, setup_toto_wrapper_imports


def test_toto_pipeline_allows_cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    setup_toto_wrapper_imports(torch_module=torch, numpy_module=np)
    module = __import__('src.models.toto_wrapper', fromlist=['TotoPipeline'])

    class DummyMaskedTimeseries:
        def __init__(self, *args, **kwargs):
            self.series = kwargs.get('series')

    class DummyForecaster:
        def __init__(self, model):
            self._model = model
            self._invocations = 0

        def forecast(self, *args, **kwargs):
            self._invocations += 1
            raise AssertionError('Forecast should not run during CPU admission test.')

    class DummyToto(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(2, 2)

        def forward(self, inputs):
            return self.model(inputs)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    monkeypatch.setattr(module, '_IMPORT_ERROR', None, raising=False)
    monkeypatch.setattr(module, 'MaskedTimeseries', DummyMaskedTimeseries, raising=False)
    monkeypatch.setattr(module, 'TotoForecaster', DummyForecaster, raising=False)
    monkeypatch.setattr(module, 'Toto', DummyToto, raising=False)

    pipeline = TotoPipeline.from_pretrained(
        device_map='cpu',
        compile_model=False,
        torch_compile=False,
        warmup_sequence=0,
        cache_policy='never',
        max_oom_retries=0,
        min_num_samples=1,
        min_samples_per_batch=1,
    )

    assert pipeline.device == 'cpu', 'Toto pipeline should admit CPU device overrides.'
    assert pipeline._autocast_dtype is None, 'CPU Toto pipeline must not enable autocast.'
    assert next(pipeline.model.parameters()).device.type == 'cpu', 'Toto model parameters did not move to CPU.'


def test_toto_pipeline_requires_available_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    setup_toto_wrapper_imports(torch_module=torch, numpy_module=np)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        TotoPipeline.from_pretrained(
            device_map="cuda",
            compile_model=False,
            warmup_sequence=0,
            max_oom_retries=0,
            min_num_samples=1,
            min_samples_per_batch=1,
        )


def test_kronos_wrapper_rejects_cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    setup_kronos_wrapper_imports(torch_module=torch, numpy_module=np, pandas_module=pd)

    with pytest.raises(RuntimeError, match="CUDA"):
        KronosForecastingWrapper(
            model_name="NeoQuasar/Kronos-small",
            tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
            device="cpu",
        )


def test_kronos_wrapper_requires_available_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    setup_kronos_wrapper_imports(torch_module=torch, numpy_module=np, pandas_module=pd)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        KronosForecastingWrapper(
            model_name="NeoQuasar/Kronos-small",
            tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
            device="cuda:0",
        )
