from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

import fal_hftraining.runner as runner
from src import dependency_injection as deps


@pytest.fixture(autouse=True)
def _stub_dependencies(monkeypatch):
    deps._reset_for_tests()
    torch_stub = ModuleType("torch")
    torch_stub.manual_seed = lambda seed: None
    torch_stub.cuda = SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda seed: None,
    )
    numpy_stub = ModuleType("numpy")
    numpy_stub.isscalar = lambda value: not hasattr(value, "__len__")
    numpy_stub.bool_ = bool
    pandas_stub = ModuleType("pandas")
    runner.setup_training_imports(torch_stub, numpy_stub, pandas_stub)
    yield
    deps._reset_for_tests()


def test_run_training_invokes_hf_pipeline(monkeypatch, tmp_path):
    metrics_file = tmp_path / "final_metrics.json"

    fake_module = ModuleType("hftraining.run_training")

    def fake_run_training(config):
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps({"val_loss": 0.123}))
        return object(), object()

    fake_module.run_training = fake_run_training
    monkeypatch.setitem(sys.modules, "hftraining.run_training", fake_module)

    config = {
        "seed": 123,
        "training": {
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "transaction_cost_bps": 5,
        },
        "data": {
            "symbols": ["SPY"],
            "context_length": 64,
            "horizon": 30,
            "trainingdata_dir": "trainingdata",
            "use_toto_forecasts": False,
        },
        "output": {"dir": str(tmp_path)},
    }

    metrics, returned_path = runner.run_training(
        config=config,
        run_name="unit-test",
        output_dir=tmp_path,
    )
    assert metrics["val_loss"] == pytest.approx(0.123)
    assert returned_path == metrics_file
