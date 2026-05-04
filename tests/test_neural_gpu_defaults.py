from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import torch
from torch.utils.data import TensorDataset
from traininglib.prefetch import CudaPrefetcher

from binanceneural.config import TrainingConfig
from binanceneural.data import _make_dataloader
from binanceneural.trainer import BinanceHourlyTrainer


def test_training_config_uses_fast_cuda_safe_defaults() -> None:
    cfg = TrainingConfig()

    assert cfg.use_amp is True
    assert cfg.amp_dtype == "bfloat16"
    assert cfg.split_amp is True
    assert cfg.use_tf32 is True
    assert cfg.gpu_cache_dataset is True
    assert cfg.cuda_prefetch_batches is True
    assert cfg.dataloader_pin_memory is True
    assert cfg.dataloader_prefetch_factor == 2


def test_make_dataloader_uses_cuda_pin_memory_default(monkeypatch) -> None:
    dataset = TensorDataset(torch.arange(8))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    loader = _make_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=None,
    )

    assert loader.pin_memory is True


def test_make_dataloader_keeps_prefetch_factor_for_worker_loaders() -> None:
    dataset = TensorDataset(torch.arange(8))

    loader = _make_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        pin_memory=False,
        prefetch_factor=3,
    )

    assert loader.prefetch_factor == 3
    assert loader.persistent_workers is True


def test_cuda_prefetcher_preserves_loader_length() -> None:
    loader = _make_dataloader(
        TensorDataset(torch.arange(8)),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )

    assert len(CudaPrefetcher(loader, "cpu")) == len(loader)


class _KwargDataModule:
    feature_columns: ClassVar[list[str]] = ["x"]

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool | None, int | None]] = []

    def train_dataloader(self, batch_size, num_workers, *, pin_memory=None, prefetch_factor=None):
        self.calls.append(("train", pin_memory, prefetch_factor))
        return [batch_size, num_workers]

    def val_dataloader(self, batch_size, num_workers, *, pin_memory=None, prefetch_factor=None):
        self.calls.append(("val", pin_memory, prefetch_factor))
        return [batch_size, num_workers]


class _LegacyDataModule:
    feature_columns: ClassVar[list[str]] = ["x"]

    def train_dataloader(self, batch_size, num_workers):
        return [batch_size, num_workers]

    def val_dataloader(self, batch_size, num_workers):
        return [batch_size, num_workers]


def test_trainer_streaming_dataloader_passes_fast_loader_kwargs(tmp_path: Path) -> None:
    data = _KwargDataModule()
    cfg = TrainingConfig(
        checkpoint_root=tmp_path / "ckpt",
        log_dir=tmp_path / "logs",
        batch_size=7,
        num_workers=1,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        device="cuda",
    )
    trainer = BinanceHourlyTrainer(cfg, data)  # type: ignore[arg-type]

    assert trainer._streaming_dataloader("train") == [7, 1]
    assert trainer._streaming_dataloader("val") == [7, 1]
    assert data.calls == [("train", True, 4), ("val", True, 4)]


def test_trainer_streaming_dataloader_keeps_legacy_loader_compat(tmp_path: Path) -> None:
    cfg = TrainingConfig(checkpoint_root=tmp_path / "ckpt", log_dir=tmp_path / "logs", batch_size=5)
    trainer = BinanceHourlyTrainer(cfg, _LegacyDataModule())  # type: ignore[arg-type]

    assert trainer._streaming_dataloader("train") == [5, 0]
