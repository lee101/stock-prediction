from __future__ import annotations

import json
from pathlib import Path

from src.models.model_cache import ModelCacheManager, device_to_token


class _DummyModel:
    def save_pretrained(self, directory: str, safe_serialization: bool = True) -> None:  # type: ignore[override]
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "weights.bin").write_bytes(b"")

    def state_dict(self):
        return {}


def test_device_to_token_normalization():
    assert device_to_token("cuda") == "cuda"
    assert device_to_token("cuda:0") == "cuda"
    assert device_to_token("GPU0") == "cuda"
    assert device_to_token("cpu") == "cpu"
    assert device_to_token("mps:0") == "mps"
    assert device_to_token("auto") == "auto"


def test_model_cache_variant_directories(tmp_path: Path):
    manager = ModelCacheManager("demo", root=tmp_path)
    metadata = {"model_id": "demo/model", "dtype": "fp32"}
    model = _DummyModel()

    manager.persist_model_state(
        model_id="demo/model",
        dtype_token="fp32",
        model=model,
        metadata=dict(metadata),
        variant_token="cuda",
    )
    manager.persist_model_state(
        model_id="demo/model",
        dtype_token="fp32",
        model=model,
        metadata=dict(metadata),
        variant_token="cpu",
    )

    gpu_dir = manager.weights_dir("demo/model", "fp32", "cuda")
    cpu_dir = manager.weights_dir("demo/model", "fp32", "cpu")

    assert gpu_dir.exists()
    assert cpu_dir.exists()
    assert gpu_dir != cpu_dir
    assert (gpu_dir / "config.json").exists()
    assert (cpu_dir / "config.json").exists()

    gpu_metadata = manager.load_metadata("demo/model", "fp32", "cuda")
    cpu_metadata = manager.load_metadata("demo/model", "fp32", "cpu")
    assert gpu_metadata is not None and gpu_metadata["data_format"] == "pretrained"
    assert cpu_metadata is not None and cpu_metadata["data_format"] == "pretrained"

    gpu_path = manager.load_pretrained_path("demo/model", "fp32", "cuda")
    cpu_path = manager.load_pretrained_path("demo/model", "fp32", "cpu")
    assert gpu_path == gpu_dir
    assert cpu_path == cpu_dir


def test_model_cache_legacy_fallback(tmp_path: Path):
    manager = ModelCacheManager("legacy", root=tmp_path)
    metadata = {"model_id": "legacy/model", "dtype": "fp32"}
    legacy_metadata_path = manager.metadata_path("legacy/model", "fp32")
    legacy_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    legacy_weights_dir = manager.weights_dir("legacy/model", "fp32")
    legacy_weights_dir.mkdir(parents=True, exist_ok=True)
    (legacy_weights_dir / "config.json").write_text("{}", encoding="utf-8")

    recovered_metadata = manager.load_metadata("legacy/model", "fp32", "cuda")
    assert recovered_metadata == metadata

    recovered_path = manager.load_pretrained_path("legacy/model", "fp32", "cuda")
    assert recovered_path == legacy_weights_dir
