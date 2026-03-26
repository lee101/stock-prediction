from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from binanceneural.config import TrainingConfig
from binanceneural.data import FeatureNormalizer
from binanceneural.jax_trainer import JaxClassicTrainer


class _SyntheticDataset(Dataset):
    def __init__(self, n_features: int, seq_len: int, n_samples: int = 10):
        self.n_features = n_features
        self.seq_len = seq_len
        self.n_samples = n_samples
        base = np.linspace(100.0, 104.0, n_samples + seq_len, dtype=np.float32)
        self.prices = base + np.sin(np.arange(n_samples + seq_len, dtype=np.float32)) * 0.2

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx
        end = idx + self.seq_len
        closes = torch.from_numpy(self.prices[start:end].copy())
        highs = closes * 1.01
        lows = closes * 0.99
        opens = closes * 0.999
        return {
            "features": torch.randn(self.seq_len, self.n_features),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "reference_close": closes.clone(),
            "chronos_high": highs * 1.002,
            "chronos_low": lows * 0.998,
            "can_long": torch.tensor(1.0),
            "can_short": torch.tensor(0.0),
        }


class _SyntheticDataModule:
    def __init__(self, n_features: int = 8, seq_len: int = 8) -> None:
        self.feature_columns = [f"feat_{idx}" for idx in range(n_features)]
        self.normalizer = FeatureNormalizer(
            mean=np.zeros(n_features, dtype=np.float32),
            std=np.ones(n_features, dtype=np.float32),
        )
        self._train = _SyntheticDataset(n_features, seq_len)
        self._val = _SyntheticDataset(n_features, seq_len)

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


def test_jax_trainer_logs_with_wandboard(monkeypatch) -> None:
    monkeypatch.setattr("binanceneural.jax_trainer.WandBoardLogger", _DummyWandBoardLogger)
    _DummyWandBoardLogger.instances.clear()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainingConfig(
            epochs=1,
            batch_size=2,
            sequence_length=8,
            learning_rate=1e-3,
            weight_decay=1e-4,
            transformer_dim=16,
            transformer_layers=1,
            transformer_heads=4,
            transformer_dropout=0.0,
            use_compile=False,
            dry_train_steps=1,
            checkpoint_root=Path(tmpdir) / "ckpts",
            log_dir=Path(tmpdir) / "tb",
            run_name="jax_wandboard_test",
            wandb_project="test-project",
            wandb_group="test-group",
            wandb_tags="jax,smoke",
        )
        trainer = JaxClassicTrainer(cfg, _SyntheticDataModule())
        artifacts = trainer.train()

    assert artifacts.best_checkpoint is not None
    assert _DummyWandBoardLogger.instances
    logger = _DummyWandBoardLogger.instances[-1]
    assert logger.kwargs["project"] == "test-project"
    assert logger.kwargs["group"] == "test-group"
    assert logger.kwargs["tags"] == ("jax", "smoke")
    assert logger.logs
    assert any("val/score" in payload for payload, _step in logger.logs)
    assert logger.texts
    assert logger.hparams
