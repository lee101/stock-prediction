import json
from types import SimpleNamespace

import faltrain.batch_size_tuner as bst


class _DummyCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def get_device_name(index: int) -> str:
        return "FakeGPU"

    @staticmethod
    def get_device_properties(index: int):
        return SimpleNamespace(total_memory=141 * 1024**3)


class _DummyTester:
    def __init__(self, **kwargs) -> None:
        pass

    @staticmethod
    def supports(value: int) -> bool:
        return value <= 512


def test_auto_tune_persists_and_reuses(monkeypatch, tmp_path):
    persist_path = tmp_path / "best_hyper_params.json"
    monkeypatch.setattr(bst, "_PERSIST_PATHS", (persist_path,))
    monkeypatch.setattr(bst, "_PERSISTED", {})
    monkeypatch.setattr(bst, "_CACHE", {})
    monkeypatch.setattr(
        bst,
        "_load_torch",
        lambda: SimpleNamespace(cuda=_DummyCuda),
    )
    monkeypatch.setattr(bst, "_HeuristicBatchSizeTester", _DummyTester)

    result = bst.auto_tune_batch_sizes(
        candidates=[128, 256, 512, 1024],
        context_lengths=[512],
        horizons=[30],
    )
    assert isinstance(result, bst.BatchSizeSelection)
    assert result.selected == 512
    assert result.signature is not None
    assert result.fallback_values() == [512, 256, 128]
    meta = result.meta()
    assert meta["candidates_desc"] == [1024, 512, 256, 128]
    assert meta["candidates_user"] == [128, 256, 512, 1024]
    assert persist_path.exists()

    with persist_path.open("r") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict)
    signature = next(iter(payload))
    entry = payload[signature]
    assert entry["batch_size"] == 512
    assert entry["context_length"] >= 512
    assert entry["horizon"] >= 30

    # Force cache miss and ensure persisted value is reused even if heuristics fail.
    monkeypatch.setattr(bst, "_CACHE", {})

    class _FailingTester:
        def __init__(self, **kwargs):
            raise AssertionError("Should not instantiate tester when persisted data exists")

    monkeypatch.setattr(bst, "_HeuristicBatchSizeTester", _FailingTester)
    reused = bst.auto_tune_batch_sizes(
        candidates=[128, 256, 512, 1024],
        context_lengths=[512],
        horizons=[30],
    )
    assert isinstance(reused, bst.BatchSizeSelection)
    assert reused.selected == 512
    assert reused.descending_candidates == (1024, 512, 256, 128)


def test_get_cached_batch_selection_uses_persisted(monkeypatch):
    import faltrain.batch_size_tuner as bst

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def get_device_name(index: int) -> str:
            return "CachedGPU"

        @staticmethod
        def get_device_properties(index: int):
            return SimpleNamespace(total_memory=256 * 1024**3)

    torch_stub = SimpleNamespace(cuda=FakeCuda)

    signature = "CachedGPU:274877906944"
    monkeypatch.setattr(bst, "_CACHE", {})
    monkeypatch.setattr(bst, "_PERSIST_PATHS", ())
    monkeypatch.setattr(
        bst,
        "_PERSISTED",
        {
            signature: {
                "batch_size": 512,
                "context_length": 1024,
                "horizon": 90,
                "updated_at": "2024-01-01T00:00:00Z",
            }
        },
    )
    monkeypatch.setattr(bst, "_load_torch", lambda: torch_stub)

    selection = bst.get_cached_batch_selection(
        candidates=[128, 256, 512],
        context_lengths=[512, 768],
        horizons=[30, 60],
    )

    assert selection is not None
    assert selection.selected == 512
    assert selection.signature == signature
    assert selection.fallback_values() == [512, 256, 128]
    assert bst._CACHE[signature] == 512


def test_setup_training_imports_assigns_modules(monkeypatch):
    import faltrain.batch_size_tuner as bst

    fake_torch = object()
    fake_numpy = object()

    monkeypatch.setattr(bst, "_TORCH", None)
    monkeypatch.setattr(bst, "_NUMPY", None)

    bst.setup_training_imports(fake_torch, fake_numpy)

    assert bst._TORCH is fake_torch
    assert bst._NUMPY is fake_numpy
