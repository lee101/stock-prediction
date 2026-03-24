"""Integration tests for SAP trainer with synthetic data."""

import numpy as np
import pandas as pd
import pytest
import torch

from sharpnessadjustedproximalpolicy.config import SAPConfig
from sharpnessadjustedproximalpolicy.trainer import SAPTrainer

# We need to build a minimal data module for testing
from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer, BinanceHourlyDataset
from torch.utils.data import DataLoader, Dataset


class SyntheticDataModule:
    """Minimal data module with synthetic price data for testing."""

    def __init__(self, n_samples=500, seq_len=48, n_features=22):
        self.feature_columns = [f"f{i}" for i in range(n_features)]
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(n_samples) * 0.5)
        prices = np.abs(prices) + 50.0

        features = np.random.randn(n_samples, n_features).astype(np.float32)
        self.normalizer = FeatureNormalizer.fit(features)
        norm_features = self.normalizer.transform(features)

        split = max(n_samples - seq_len * 3, seq_len + 1)
        self._train = SyntheticDataset(norm_features[:split], prices[:split], seq_len)
        val_start = max(0, split - seq_len)
        self._val = SyntheticDataset(norm_features[val_start:], prices[val_start:], seq_len)

    def train_dataloader(self, batch_size, num_workers=0):
        return DataLoader(self._train, batch_size=batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self, batch_size, num_workers=0):
        return DataLoader(self._val, batch_size=batch_size, shuffle=False, drop_last=False)


class SyntheticDataset(Dataset):
    def __init__(self, features, prices, seq_len):
        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.seq_len = seq_len
        self.n = len(prices) - seq_len + 1

    def __len__(self):
        return max(self.n, 1)

    def __getitem__(self, idx):
        s, e = idx, idx + self.seq_len
        close = self.prices[s:e]
        noise = np.random.randn(self.seq_len).astype(np.float32) * 0.5
        high = close + np.abs(noise)
        low = close - np.abs(noise)
        opn = close + noise * 0.3
        return {
            "features": torch.from_numpy(self.features[s:e]),
            "open": torch.from_numpy(opn),
            "high": torch.from_numpy(high),
            "low": torch.from_numpy(low),
            "close": torch.from_numpy(close),
            "reference_close": torch.from_numpy(close.copy()),
            "chronos_high": torch.from_numpy(high * 1.001),
            "chronos_low": torch.from_numpy(low * 0.999),
            "can_long": torch.tensor(1.0),
            "can_short": torch.tensor(0.0),
        }


def _make_tc(**kwargs):
    defaults = dict(
        epochs=2,
        batch_size=4,
        sequence_length=48,
        learning_rate=1e-3,
        weight_decay=0.01,
        maker_fee=0.001,
        fill_temperature=0.01,
        decision_lag_bars=0,
        transformer_dim=32,
        transformer_layers=1,
        transformer_heads=4,
        use_compile=False,
        use_tf32=False,
        use_flash_attention=False,
        use_flex_attention=False,
        validation_use_binary_fills=True,
        checkpoint_root="sharpnessadjustedproximalpolicy/test_checkpoints",
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


@pytest.mark.parametrize("sam_mode", ["none", "periodic", "looksam"])
def test_trainer_modes(sam_mode, tmp_path):
    tc = _make_tc(checkpoint_root=str(tmp_path))
    sc = SAPConfig(sam_mode=sam_mode, rho=0.05, probe_every=2, looksam_every=2, early_stop_patience=0)
    dm = SyntheticDataModule(n_samples=200, seq_len=48, n_features=22)
    trainer = SAPTrainer(tc, sc, dm)
    artifacts, history = trainer.train()

    assert len(history) == 2
    assert history[0].val_sortino != 0 or history[0].val_return != 0
    assert artifacts.best_checkpoint is not None
    assert artifacts.best_checkpoint.exists()


def test_early_stopping(tmp_path):
    tc = _make_tc(epochs=20, checkpoint_root=str(tmp_path))
    sc = SAPConfig(sam_mode="none", early_stop_patience=2, early_stop_min_epochs=2)
    dm = SyntheticDataModule(n_samples=200, seq_len=48, n_features=22)
    trainer = SAPTrainer(tc, sc, dm)
    _, history = trainer.train()
    # should stop before 20 epochs
    assert len(history) < 20


def test_sharpness_tracking(tmp_path):
    tc = _make_tc(epochs=3, checkpoint_root=str(tmp_path))
    sc = SAPConfig(sam_mode="periodic", rho=0.1, probe_every=1, early_stop_patience=0)
    dm = SyntheticDataModule(n_samples=200, seq_len=48, n_features=22)
    trainer = SAPTrainer(tc, sc, dm)
    _, history = trainer.train()

    # at least some sharpness should be recorded
    assert any(h.sharpness_ema != 0 for h in history)
    assert any(h.lr_scale != 1.0 for h in history[1:])  # first may be 1.0


def test_checkpoint_has_sharpness(tmp_path):
    tc = _make_tc(epochs=1, checkpoint_root=str(tmp_path))
    sc = SAPConfig(sam_mode="periodic", probe_every=1, early_stop_patience=0)
    dm = SyntheticDataModule(n_samples=200, seq_len=48, n_features=22)
    trainer = SAPTrainer(tc, sc, dm)
    artifacts, _ = trainer.train()

    ckpt = torch.load(artifacts.best_checkpoint, weights_only=False)
    assert "sharpness" in ckpt
    assert "sap_config" in ckpt
