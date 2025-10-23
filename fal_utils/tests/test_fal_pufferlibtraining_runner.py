from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import fal_pufferlibtraining.runner as runner
from src import dependency_injection as deps


@pytest.fixture(autouse=True)
def _stub_dependencies(monkeypatch):
    deps._reset_for_tests()
    torch_stub = ModuleType("torch")
    torch_stub.manual_seed = lambda seed: None
    torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
    numpy_stub = ModuleType("numpy")
    numpy_stub.isscalar = lambda value: not hasattr(value, "__len__")
    numpy_stub.bool_ = bool
    runner.setup_training_imports(torch_stub, numpy_stub, None)
    yield
    deps._reset_for_tests()


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--trainingdata-dir", default="trainingdata")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--tensorboard-dir", default="tensorboard")
    parser.add_argument("--rl-epochs", type=int, default=5)
    parser.add_argument("--rl-batch-size", type=int, default=32)
    parser.add_argument("--rl-learning-rate", type=float, default=3e-4)
    parser.add_argument("--rl-optimizer", default="adamw")
    parser.add_argument("--summary-path", default="summary.json")
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-group", default="")
    return parser


def test_run_training_calls_pipeline(monkeypatch, tmp_path):
    fake_module = ModuleType("pufferlibtraining.train_ppo")
    fake_module.build_argument_parser = _build_parser

    def fake_run_pipeline(args):
        assert Path(args.trainingdata_dir)
        return {"portfolio_pairs": {"AAPL_MSFT": {"return": 0.42}}}

    fake_module.run_pipeline = fake_run_pipeline
    monkeypatch.setitem(sys.modules, "pufferlibtraining.train_ppo", fake_module)

    summary, summary_path = runner.run_training(
        trainingdata_dir=tmp_path / "trainingdata",
        output_dir=tmp_path / "outputs",
        tensorboard_dir=tmp_path / "tensorboard",
        cfg={"batch_size": 16, "learning_rate": 1e-4, "transaction_cost_bps": 7.5},
        epochs=3,
        transaction_cost_bps=5.0,
        run_name="puffer-test",
    )

    assert "AAPL_MSFT" in summary["portfolio_pairs"]
    assert summary_path.exists()
    written = json.loads(summary_path.read_text())
    assert written["portfolio_pairs"]["AAPL_MSFT"]["return"] == 0.42
