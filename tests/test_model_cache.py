from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.models.model_cache import ModelCacheError, ModelCacheManager


def test_model_cache_manager_uses_fallback_root_when_repo_disk_is_full(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    fallback_root = tmp_path / "fallback" / "compiled_models"
    fallback_root.parent.mkdir(parents=True)

    monkeypatch.chdir(repo_root)
    monkeypatch.delenv("COMPILED_MODELS_DIR", raising=False)
    monkeypatch.setenv("COMPILED_MODELS_FALLBACK_DIRS", str(fallback_root))
    monkeypatch.setenv("COMPILED_MODELS_MIN_FREE_BYTES", "1024")

    def fake_disk_usage(path: str | Path):
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        if repo_root in {candidate, *candidate.parents}:
            return SimpleNamespace(total=10_000, used=9_500, free=0)
        if fallback_root.parent in {candidate, *candidate.parents}:
            return SimpleNamespace(total=10_000, used=1_000, free=9_000)
        return SimpleNamespace(total=10_000, used=1_000, free=9_000)

    monkeypatch.setattr("src.models.model_cache.shutil.disk_usage", fake_disk_usage)

    manager = ModelCacheManager("chronos2")

    assert manager.root == fallback_root
    assert manager.weights_dir("amazon/chronos-2", "fp32").parent.parent.parent == fallback_root / "chronos2"


def test_persist_model_state_wraps_torch_save_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class DummyModel:
        def state_dict(self):
            return {"weight": 1}

    class DummyTorch:
        @staticmethod
        def save(_obj, _path):
            raise RuntimeError("disk full")

    monkeypatch.setitem(sys.modules, "torch", DummyTorch)

    manager = ModelCacheManager("chronos2", root=tmp_path / "compiled_models")

    with pytest.raises(ModelCacheError, match="Failed to persist model state"):
        manager.persist_model_state(
            model_id="amazon/chronos-2",
            dtype_token="fp32",
            model=DummyModel(),
            metadata={"model_id": "amazon/chronos-2"},
        )

    state_path = manager.weights_dir("amazon/chronos-2", "fp32") / "model_state.pt"
    tmp_state_path = state_path.with_suffix(".pt.tmp")
    assert not state_path.exists()
    assert not tmp_state_path.exists()
