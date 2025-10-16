#!/usr/bin/env python3
"""Sanity checks for the neural strategy experiment harness."""

import json
from pathlib import Path

import pytest

from experiments.neural_strategies.toto_distillation import TotoDistillationExperiment
from experiments.neural_strategies.dual_attention import DualAttentionPrototype


@pytest.mark.parametrize(
    "experiment_cls,config",
    [
        (
            TotoDistillationExperiment,
            {
                "name": "test_toto_cpu",
                "strategy": "toto_distillation",
                "data": {
                    "symbol": "AAPL",
                    "csv_path": "WIKI-AAPL.csv",
                    "sequence_length": 30,
                    "prediction_horizon": 3,
                    "train_split": 0.6,
                    "val_split": 0.2,
                },
                "model": {"hidden_size": 64, "num_layers": 1, "dropout": 0.0},
                "training": {
                    "epochs": 1,
                    "batch_size": 64,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "dtype": "fp32",
                    "gradient_checkpointing": False,
                },
            },
        ),
        (
            DualAttentionPrototype,
            {
                "name": "test_dual_attention_cpu",
                "strategy": "dual_attention_prototype",
                "data": {
                    "symbol": "AAPL",
                    "csv_path": "WIKI-AAPL.csv",
                    "context_length": 16,
                    "prediction_horizon": 3,
                    "train_split": 0.6,
                    "val_split": 0.2,
                },
                "model": {"embed_dim": 64, "num_heads": 4, "num_layers": 1, "dropout": 0.0},
                "training": {
                    "epochs": 1,
                    "batch_size": 32,
                    "learning_rate": 0.0005,
                    "weight_decay": 0.0,
                    "dtype": "fp32",
                    "gradient_checkpointing": False,
                },
            },
        ),
    ],
)
def test_experiments_run_end_to_end(tmp_path, experiment_cls, config):
    experiment = experiment_cls(config=config, config_path=None)
    result = experiment.run()
    assert "val_mse" in result.metrics
    assert not (Path(tmp_path) / "unused").exists()
    # Ensure JSON serialization works for downstream tooling
    json.loads(result.to_json())
