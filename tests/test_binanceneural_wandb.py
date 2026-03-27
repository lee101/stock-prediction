from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from binanceneural.config import TrainingConfig
from binanceneural.data import FeatureNormalizer
from binanceneural.trainer import BinanceHourlyTrainer


class _SyntheticDataset(Dataset):
    def __init__(self, n_features: int, seq_len: int, n_samples: int = 20) -> None:
        self.n_features = n_features
        self.seq_len = seq_len
        self.n_samples = n_samples
        base_price = 100.0
        prices = base_price + np.cumsum(np.random.randn(n_samples + seq_len) * 0.5)
        self.prices = np.maximum(prices, 1.0).astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx
        end = idx + self.seq_len
        closes = torch.from_numpy(self.prices[start:end].copy())
        highs = closes * 1.01
        lows = closes * 0.99
        opens = closes * (1.0 + 0.002 * torch.randn(self.seq_len))
        return {
            "features": torch.randn(self.seq_len, self.n_features),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "reference_close": closes.clone(),
            "chronos_high": highs * 1.005,
            "chronos_low": lows * 0.995,
            "can_long": torch.tensor(1.0),
            "can_short": torch.tensor(0.0),
        }


class _SyntheticDataModule:
    def __init__(self, n_features: int = 16, seq_len: int = 32, n_samples: int = 20) -> None:
        self.feature_columns = [f"feat_{idx}" for idx in range(n_features)]
        self.normalizer = FeatureNormalizer(
            mean=np.zeros(n_features, dtype=np.float32),
            std=np.ones(n_features, dtype=np.float32),
        )
        self._train = _SyntheticDataset(n_features, seq_len, n_samples)
        self._val = _SyntheticDataset(n_features, seq_len, n_samples)

    def train_dataloader(self, batch_size: int, num_workers: int = 0):
        return DataLoader(self._train, batch_size=batch_size, drop_last=True)

    def val_dataloader(self, batch_size: int, num_workers: int = 0):
        return DataLoader(self._val, batch_size=batch_size, drop_last=False)


class _DummyWandBoardLogger:
    instances: list["_DummyWandBoardLogger"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logs: list[tuple[dict[str, float], int | None]] = []
        self.texts: list[tuple[str, str, int | None]] = []
        self.hparams: list[tuple[dict[str, object], dict[str, float], int | None, str]] = []
        _DummyWandBoardLogger.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def log(self, metrics, *, step=None, commit=None):
        self.logs.append((dict(metrics), step))

    def log_text(self, name, text, *, step=None):
        self.texts.append((name, text, step))

    def log_hparams(self, hparams, metrics, *, step=None, table_name="hparams"):
        self.hparams.append((dict(hparams), dict(metrics), step, table_name))


def _make_config(tmpdir: str, **overrides) -> TrainingConfig:
    defaults = dict(
        epochs=2,
        batch_size=4,
        sequence_length=32,
        transformer_dim=32,
        transformer_layers=1,
        transformer_heads=4,
        learning_rate=1e-3,
        dry_train_steps=2,
        use_compile=False,
        use_amp=False,
        use_tf32=False,
        use_flash_attention=False,
        checkpoint_root=Path(tmpdir) / "ckpts",
        log_dir=Path(tmpdir) / "tb",
        seed=42,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def test_train_disables_wandb_when_project_missing(monkeypatch) -> None:
    monkeypatch.setattr("binanceneural.trainer.WandBoardLogger", _DummyWandBoardLogger)
    _DummyWandBoardLogger.instances.clear()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(tmpdir, wandb_project=None, run_name="torch_no_wandb")
        trainer = BinanceHourlyTrainer(cfg, _SyntheticDataModule())
        artifacts = trainer.train()

    assert artifacts.best_checkpoint is not None
    logger = _DummyWandBoardLogger.instances[-1]
    assert logger.kwargs["enable_wandb"] is False
    assert logger.logs
    assert logger.texts


def test_train_logs_metrics_via_wandboard(monkeypatch) -> None:
    monkeypatch.setattr("binanceneural.trainer.WandBoardLogger", _DummyWandBoardLogger)
    _DummyWandBoardLogger.instances.clear()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(
            tmpdir,
            wandb_project="test-project",
            wandb_group="torch-group",
            wandb_tags="torch,smoke",
            run_name="torch_wandboard_test",
            epochs=3,
        )
        trainer = BinanceHourlyTrainer(cfg, _SyntheticDataModule())
        artifacts = trainer.train()

    assert len(artifacts.history) == 3
    logger = _DummyWandBoardLogger.instances[-1]
    assert logger.kwargs["project"] == "test-project"
    assert logger.kwargs["group"] == "torch-group"
    assert logger.kwargs["tags"] == ("torch", "smoke")
    assert any("val/score" in payload and "learning_rate" in payload for payload, _step in logger.logs)
    assert any(name == "train/feature_columns" for name, _text, _step in logger.texts)
    assert any(table_name == "torch_train_summary" for _hp, _metrics, _step, table_name in logger.hparams)


def test_dry_train_completes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(tmpdir, dry_train_steps=2, epochs=2, wandb_project=None)
        trainer = BinanceHourlyTrainer(cfg, _SyntheticDataModule())
        artifacts = trainer.train()
    assert len(artifacts.history) == 2
    assert artifacts.best_checkpoint is not None
