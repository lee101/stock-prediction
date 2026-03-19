"""Tests for WandB integration in binanceneural trainer."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binanceneural.config import TrainingConfig
from binanceneural.data import FeatureNormalizer
from binanceneural.trainer import BinanceHourlyTrainer


class SyntheticDataset(Dataset):
    """Minimal dataset matching BinanceHourlyDataset interface."""

    def __init__(self, n_features: int, seq_len: int, n_samples: int = 20):
        self.n_features = n_features
        self.seq_len = seq_len
        self.n_samples = n_samples
        base_price = 100.0
        self.prices = base_price + np.cumsum(np.random.randn(n_samples + seq_len) * 0.5)
        self.prices = np.maximum(self.prices, 1.0).astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s = idx
        e = idx + self.seq_len
        p = self.prices[s:e]
        closes = torch.from_numpy(p.copy())
        highs = closes * 1.01
        lows = closes * 0.99
        opens = closes * (1.0 + 0.002 * torch.randn(self.seq_len))
        ref = closes.clone()
        ch = highs * 1.005
        cl = lows * 0.995
        features = torch.randn(self.seq_len, self.n_features)
        return {
            "features": features,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "reference_close": ref,
            "chronos_high": ch,
            "chronos_low": cl,
            "can_long": torch.tensor(1.0),
            "can_short": torch.tensor(0.0),
        }


class SyntheticDataModule:
    """Minimal data module matching BinanceHourlyDataModule interface."""

    def __init__(self, n_features: int = 16, seq_len: int = 32, n_samples: int = 20):
        self.feature_columns = [f"feat_{i}" for i in range(n_features)]
        self.normalizer = FeatureNormalizer(
            mean=np.zeros(n_features, dtype=np.float32),
            std=np.ones(n_features, dtype=np.float32),
        )
        self._train = SyntheticDataset(n_features, seq_len, n_samples)
        self._val = SyntheticDataset(n_features, seq_len, n_samples)

    def train_dataloader(self, batch_size, num_workers=0):
        return DataLoader(self._train, batch_size=batch_size, drop_last=True)

    def val_dataloader(self, batch_size, num_workers=0):
        return DataLoader(self._val, batch_size=batch_size, drop_last=False)


def _make_config(tmpdir, **overrides):
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
        seed=42,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def test_train_no_wandb_when_project_none():
    """wandb_project=None should not attempt wandb.init."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(tmpdir, wandb_project=None)
        dm = SyntheticDataModule(n_features=16, seq_len=32)
        trainer = BinanceHourlyTrainer(cfg, dm)
        with patch("binanceneural.trainer._wandb", None):
            artifacts = trainer.train()
        assert len(artifacts.history) == 2


def test_train_wandb_offline_mode():
    """Training with wandb in offline mode completes without error."""
    try:
        import wandb
    except ImportError:
        pytest.skip("wandb not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = tmpdir
        os.environ["WANDB_SILENT"] = "true"
        try:
            cfg = _make_config(tmpdir, wandb_project="test-project", wandb_entity=None)
            dm = SyntheticDataModule(n_features=16, seq_len=32)
            trainer = BinanceHourlyTrainer(cfg, dm)
            artifacts = trainer.train()
            assert len(artifacts.history) == 2
            assert artifacts.history[0].val_sortino is not None
        finally:
            os.environ.pop("WANDB_MODE", None)
            os.environ.pop("WANDB_DIR", None)
            os.environ.pop("WANDB_SILENT", None)


def test_train_wandb_init_failure_graceful():
    """If wandb.init raises, training continues without wandb."""
    mock_wandb = MagicMock()
    mock_wandb.init.side_effect = RuntimeError("mock wandb failure")

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(tmpdir, wandb_project="test-project")
        dm = SyntheticDataModule(n_features=16, seq_len=32)
        trainer = BinanceHourlyTrainer(cfg, dm)
        with patch("binanceneural.trainer._wandb", mock_wandb):
            artifacts = trainer.train()
        assert len(artifacts.history) == 2
        mock_wandb.init.assert_called_once()


def test_train_wandb_log_called_per_epoch():
    """wandb.log is called once per epoch with correct keys."""
    mock_run = MagicMock()
    mock_wandb = MagicMock()
    mock_wandb.init.return_value = mock_run

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(tmpdir, wandb_project="test-project", epochs=3)
        dm = SyntheticDataModule(n_features=16, seq_len=32)
        trainer = BinanceHourlyTrainer(cfg, dm)
        with patch("binanceneural.trainer._wandb", mock_wandb):
            artifacts = trainer.train()

        assert mock_run.log.call_count == 3
        assert mock_run.finish.call_count == 1
        first_call_kwargs = mock_run.log.call_args_list[0]
        logged = first_call_kwargs[0][0]
        for key in ("epoch", "train/loss", "train/sortino", "val/loss", "val/sortino", "learning_rate"):
            assert key in logged, f"Missing key: {key}"


def test_dry_train_completes():
    """dry_train_steps=2 with 2 epochs completes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _make_config(tmpdir, dry_train_steps=2, epochs=2, wandb_project=None)
        dm = SyntheticDataModule(n_features=16, seq_len=32)
        trainer = BinanceHourlyTrainer(cfg, dm)
        artifacts = trainer.train()
        assert len(artifacts.history) == 2
        assert artifacts.best_checkpoint is not None
