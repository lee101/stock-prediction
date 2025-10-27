from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import torch

from faltrain.forecasting import create_kronos_wrapper, create_toto_pipeline
from faltrain.hyperparams import HyperparamResolver, HyperparamResult


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_resolver_prefers_best_when_model_matches(tmp_path: Path) -> None:
    payload = {
        "symbol": "AAPL",
        "model": "toto",
        "config": {"aggregate": "mean", "num_samples": 64, "samples_per_batch": 16},
    }
    _write_payload(tmp_path / "best" / "AAPL.json", payload)
    resolver = HyperparamResolver(search_roots=(tmp_path,))

    result = resolver.load("AAPL", "toto")

    assert result is not None
    assert result.kind == "best"
    assert result.config["aggregate"] == "mean"


def test_resolver_skips_best_on_model_mismatch(tmp_path: Path) -> None:
    best_payload = {"symbol": "AAPL", "model": "toto", "config": {}}
    kronos_payload = {
        "symbol": "AAPL",
        "model": "kronos",
        "config": {"temperature": 0.2},
    }
    _write_payload(tmp_path / "best" / "AAPL.json", best_payload)
    _write_payload(tmp_path / "kronos" / "AAPL.json", kronos_payload)

    resolver = HyperparamResolver(search_roots=(tmp_path,))

    result = resolver.load("AAPL", "kronos")

    assert result is not None
    assert result.kind == "kronos"
    assert result.config["temperature"] == 0.2


class _FakeS3Client:
    def __init__(self, mapping: dict[str, dict[str, Any]]) -> None:
        self._mapping = mapping

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803 (match boto3 signature)
        if Key not in self._mapping:
            raise FileNotFoundError(Key)
        payload = self._mapping[Key]
        return {"Body": io.BytesIO(json.dumps(payload).encode("utf-8"))}


def test_resolver_remote_fallback(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    resolver = HyperparamResolver(search_roots=(), bucket="models", remote_prefix="stock", endpoint_url="https://example.com")

    fake_payload = {
        "symbol": "MSFT",
        "model": "kronos",
        "config": {"temperature": 0.4},
    }
    client = _FakeS3Client({"stock/kronos/MSFT.json": fake_payload})

    result = resolver.load("MSFT", "kronos", s3_client=client)

    assert result is not None
    assert result.source.startswith("s3://")
    assert result.config["temperature"] == 0.4


def test_create_kronos_wrapper_applies_config(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    payload = {
        "symbol": "AAPL",
        "model": "kronos",
        "config": {
            "temperature": 0.31,
            "top_p": 0.87,
            "top_k": 4,
            "sample_count": 128,
            "max_context": 256,
            "clip": 3.5,
            "model_name": "custom/kronos",
            "tokenizer_name": "custom/tokenizer",
        },
    }
    result = HyperparamResult(payload=payload, source="file://fake", kind="kronos")

    class _Resolver:
        def load(self, *_: Any, **__: Any) -> HyperparamResult:
            return result

    class _Ctor:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    bundle = create_kronos_wrapper(
        "AAPL",
        resolver=_Resolver(),
        wrapper_ctor=_Ctor,
        device="cpu",
    )

    assert isinstance(bundle.wrapper, _Ctor)
    assert bundle.wrapper.kwargs["temperature"] == 0.31
    assert bundle.wrapper.kwargs["top_p"] == 0.87
    assert bundle.wrapper.kwargs["sample_count"] == 128
    assert bundle.temperature == 0.31
    assert bundle.top_k == 4
    assert bundle.max_context == 256


def test_create_toto_pipeline_applies_config(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    payload = {
        "symbol": "AAPL",
        "model": "toto",
        "config": {
            "aggregate": " trimmed_mean_10 ",
            "num_samples": 256,
            "samples_per_batch": 32,
        },
    }
    result = HyperparamResult(payload=payload, source="file://fake", kind="toto")

    class _Resolver:
        def load(self, *_: Any, **__: Any) -> HyperparamResult:
            return result

    factory_calls: dict[str, Any] = {}

    def factory(**kwargs: Any) -> dict[str, Any]:
        factory_calls.update(kwargs)
        return kwargs

    bundle = create_toto_pipeline(
        "AAPL",
        resolver=_Resolver(),
        pipeline_factory=factory,
        device_map="cpu",
        cache_policy="prefer",
    )

    assert bundle.aggregate == "trimmed_mean_10"
    assert bundle.num_samples == 256
    assert bundle.samples_per_batch == 32
    assert factory_calls["device_map"] == "cpu"
    assert factory_calls["torch_dtype"] == torch.float32
    assert factory_calls["amp_dtype"] is None
    assert factory_calls["torch_compile"] is True
    assert factory_calls["compile_mode"] == "max-autotune"
    assert factory_calls["cache_policy"] == "prefer"
