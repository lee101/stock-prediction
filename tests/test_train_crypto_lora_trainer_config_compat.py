from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.train_crypto_lora_sweep import TrainConfig, train_and_evaluate


@pytest.mark.unit
def test_train_and_evaluate_uses_trainer_config(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "BTCUSD.csv"
    rows = 24
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
            "open": np.linspace(100.0, 123.0, rows),
            "high": np.linspace(101.0, 124.0, rows),
            "low": np.linspace(99.0, 122.0, rows),
            "close": np.linspace(100.5, 123.5, rows),
            "volume": np.ones(rows),
        }
    )
    frame.to_csv(csv_path, index=False)

    captured: dict[str, object] = {}

    @dataclass
    class FakeTrainerConfig:
        symbol: str
        data_root: Path | None
        output_root: Path
        prediction_length: int = 1
        context_length: int = 1024
        batch_size: int = 64
        learning_rate: float = 1e-5
        num_steps: int = 1000
        finetune_mode: str = "full"
        lora_r: int = 16
        lora_alpha: int = 32
        lora_dropout: float = 0.05
        lora_targets: tuple[str, ...] = ("q", "k", "v", "o")
        merge_lora: bool = True
        lr_scheduler_type: str = "cosine"

    class DummyPipeline:
        quantiles = [0.5]

        def predict(self, inputs, prediction_length, batch_size):
            target = np.asarray(inputs[0], dtype=np.float32)
            forecast = torch.from_numpy(target[:, -prediction_length:]).unsqueeze(1)
            return [forecast]

    def fake_load_pipeline(model_id, device_map, torch_dtype):
        assert model_id == "amazon/chronos-2"
        assert device_map == "cuda"
        assert torch_dtype is None
        return DummyPipeline()

    def fake_fit_pipeline(pipeline, train_inputs, val_inputs, trainer_cfg, output_dir):
        captured["trainer_cfg"] = trainer_cfg
        assert isinstance(trainer_cfg, FakeTrainerConfig)
        assert trainer_cfg.lr_scheduler_type == "cosine"
        assert trainer_cfg.finetune_mode == "lora"
        return pipeline

    fake_module = types.SimpleNamespace(
        TrainerConfig=FakeTrainerConfig,
        _load_pipeline=fake_load_pipeline,
        _fit_pipeline=fake_fit_pipeline,
        _save_pipeline=lambda pipeline, output_dir, name: None,
    )
    monkeypatch.setitem(sys.modules, "chronos2_trainer", fake_module)

    cfg = TrainConfig(
        symbol="BTCUSD",
        context_length=4,
        prediction_length=2,
        batch_size=2,
        learning_rate=1e-4,
        num_steps=3,
        lora_r=8,
        lora_alpha=16,
        preaug="baseline",
        val_hours=4,
        test_hours=4,
    )
    result = train_and_evaluate(cfg, csv_path, tmp_path / "out")

    trainer_cfg = captured["trainer_cfg"]
    assert trainer_cfg.symbol == "BTCUSD"
    assert trainer_cfg.data_root == tmp_path
    assert trainer_cfg.output_root == tmp_path / "out"
    assert result["val"]["count"] > 0
    assert result["test"]["count"] > 0


@pytest.mark.unit
def test_train_and_evaluate_passes_seed_when_supported(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "BTCUSD.csv"
    rows = 24
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
            "open": np.linspace(100.0, 123.0, rows),
            "high": np.linspace(101.0, 124.0, rows),
            "low": np.linspace(99.0, 122.0, rows),
            "close": np.linspace(100.5, 123.5, rows),
            "volume": np.ones(rows),
        }
    )
    frame.to_csv(csv_path, index=False)

    captured: dict[str, object] = {}

    @dataclass
    class FakeTrainerConfig:
        symbol: str
        data_root: Path | None
        output_root: Path
        seed: int = 0
        prediction_length: int = 1
        context_length: int = 1024
        batch_size: int = 64
        learning_rate: float = 1e-5
        num_steps: int = 1000
        finetune_mode: str = "full"
        lora_r: int = 16
        lora_alpha: int = 32
        lora_dropout: float = 0.05
        lora_targets: tuple[str, ...] = ("q", "k", "v", "o")
        merge_lora: bool = True

    class DummyPipeline:
        quantiles = [0.5]

        def predict(self, inputs, prediction_length, batch_size):
            target = np.asarray(inputs[0], dtype=np.float32)
            forecast = torch.from_numpy(target[:, -prediction_length:]).unsqueeze(1)
            return [forecast]

    def fake_fit_pipeline(pipeline, train_inputs, val_inputs, trainer_cfg, output_dir):
        captured["trainer_cfg"] = trainer_cfg
        return pipeline

    fake_module = types.SimpleNamespace(
        TrainerConfig=FakeTrainerConfig,
        _load_pipeline=lambda *args, **kwargs: DummyPipeline(),
        _fit_pipeline=fake_fit_pipeline,
        _save_pipeline=lambda pipeline, output_dir, name: None,
    )
    monkeypatch.setitem(sys.modules, "chronos2_trainer", fake_module)

    cfg = TrainConfig(
        symbol="BTCUSD",
        context_length=4,
        prediction_length=2,
        batch_size=2,
        learning_rate=1e-4,
        num_steps=3,
        lora_r=8,
        lora_alpha=16,
        preaug="baseline",
        val_hours=4,
        test_hours=4,
        seed=2027,
    )

    train_and_evaluate(cfg, csv_path, tmp_path / "out")

    trainer_cfg = captured["trainer_cfg"]
    assert trainer_cfg.seed == 2027
