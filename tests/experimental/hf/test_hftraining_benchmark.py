#!/usr/bin/env python3
"""Integration test ensuring hftraining records timing benchmarks."""

import numpy as np
import pytest
import torch

from hftraining.train_hf import HFTrainer, StockDataset
from hftraining.hf_trainer import HFTrainingConfig, TransformerTradingModel


def test_hftrainer_records_epoch_and_step_speed(tmp_path, monkeypatch):
    """Runs a tiny training loop and verifies benchmark metrics are populated."""
    monkeypatch.setenv("AUTO_TUNE", "0")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    config = HFTrainingConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        learning_rate=1e-3,
        warmup_steps=0,
        batch_size=8,
        max_steps=12,
        eval_steps=10_000,
        save_steps=10_000,
        logging_steps=4,
        sequence_length=16,
        prediction_horizon=2,
        use_mixed_precision=False,
        use_gradient_checkpointing=False,
        use_data_parallel=False,
        use_compile=False,
        gradient_accumulation_steps=1,
        early_stopping_patience=50,
    )
    config.output_dir = str(tmp_path / "output")
    config.logging_dir = str(tmp_path / "logs")
    config.cache_dir = str(tmp_path / "cache")
    config.use_wandb = False
    config.input_features = 6
    config.length_bucketing = (config.sequence_length,)
    config.horizon_bucketing = (config.prediction_horizon,)
    config.max_tokens_per_batch = 0

    feature_dim = config.input_features
    raw_data = rng.standard_normal((256, feature_dim)).astype(np.float32)
    train_dataset = StockDataset(
        raw_data,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
    )

    model = TransformerTradingModel(config, input_dim=feature_dim)
    trainer = HFTrainer(model, config, train_dataset)

    trainer.train()

    summary = trainer.get_benchmark_summary()

    # Epoch-level assertions
    assert summary["epoch_stats"], "Expected epoch benchmark data to be recorded"
    assert len(summary["epoch_stats"]) == trainer.current_epoch
    epoch_stat = summary["epoch_stats"][0]
    assert epoch_stat["time_s"] > 0
    assert epoch_stat["steps"] > 0
    assert epoch_stat["avg_step_time_s"] > 0
    assert epoch_stat["avg_step_time_s"] == pytest.approx(epoch_stat["time_s"] / epoch_stat["steps"], rel=0.15)
    assert epoch_stat["steps_per_sec"] > 0
    assert epoch_stat["steps_per_sec"] == pytest.approx(epoch_stat["steps"] / epoch_stat["time_s"], rel=0.15)
    assert epoch_stat["samples_per_sec"] == pytest.approx(
        epoch_stat["steps_per_sec"] * config.batch_size, rel=0.15
    )
    assert "tokens_per_sec" in epoch_stat
    assert epoch_stat["tokens_per_sec"] == pytest.approx(
        epoch_stat["samples_per_sec"] * config.sequence_length, rel=0.15
    )

    # Step window assertions
    step_stats = summary["step_stats"]
    assert step_stats["window"] == trainer.global_step
    assert step_stats["avg_step_time_s"] > 0
    assert step_stats["median_step_time_s"] > 0
    assert step_stats["p90_step_time_s"] >= step_stats["median_step_time_s"]
    assert step_stats["max_step_time_s"] >= step_stats["p90_step_time_s"]
    assert step_stats["steps_per_sec"] > 0
    assert step_stats["steps_per_sec"] == pytest.approx(1.0 / step_stats["avg_step_time_s"], rel=0.15)
    assert step_stats["samples_per_sec"] == pytest.approx(
        step_stats["steps_per_sec"] * config.batch_size, rel=0.15
    )
    if "tokens_per_sec" in step_stats:
        assert step_stats["tokens_per_sec"] == pytest.approx(
            step_stats["samples_per_sec"] * config.sequence_length, rel=0.15
        )

    # Ensure the run completed the requested number of steps
    assert trainer.global_step == config.max_steps
