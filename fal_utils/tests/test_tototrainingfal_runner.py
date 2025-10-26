import json
from pathlib import Path
from types import SimpleNamespace

import tototrainingfal.runner as runner


def test_setup_training_imports_assigns_modules(monkeypatch):
    torch_stub = object()
    numpy_stub = object()
    pandas_stub = object()
    monkeypatch.setattr(runner, "_TORCH", None)
    monkeypatch.setattr(runner, "_NUMPY", None)
    monkeypatch.setattr(runner, "_PANDAS", None)

    runner.setup_training_imports(torch_stub, numpy_stub, pandas_stub)

    assert runner._TORCH is torch_stub
    assert runner._NUMPY is numpy_stub
    assert runner._PANDAS is pandas_stub


def test_run_training_invokes_train_module(monkeypatch, tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "series.npy").write_bytes(b"\x93NUMPY\x01\x00")  # minimal placeholder, not read

    def fake_ensure():
        return None

    captured = {}

    class FakeModule:
        def run_with_namespace(self, namespace):
            captured["args"] = namespace
            namespace.output_dir.mkdir(parents=True, exist_ok=True)
            summary = namespace.output_dir / "final_metrics.json"
            summary.write_text(json.dumps({"val": {"loss": 0.42}}))

    monkeypatch.setattr(runner, "_ensure_injected_modules", fake_ensure)
    monkeypatch.setattr(runner, "_load_train_module", lambda: FakeModule())

    metrics, metrics_path = runner.run_training(
        train_root=train_dir,
        val_root=None,
        context_length=128,
        prediction_length=24,
        stride=12,
        batch_size=16,
        epochs=3,
        learning_rate=1e-3,
        loss="huber",
        output_dir=tmp_path / "out",
        device="cpu",
        grad_accum=2,
        compile=False,
        quantiles=[0.2, 0.5, 0.8],
    )

    args = captured["args"]
    assert args.train_root == train_dir
    assert args.val_root is None
    assert args.context_length == 128
    assert args.prediction_length == 24
    assert args.stride == 12
    assert args.batch_size == 16
    assert args.epochs == 3
    assert args.learning_rate == 1e-3
    assert args.grad_accum == 2
    assert args.device == "cpu"
    assert args.compile is False
    assert args.quantiles == [0.2, 0.5, 0.8]
    assert args.prefetch_to_gpu is False
    assert metrics == {"val": {"loss": 0.42}}
    assert metrics_path.exists()
