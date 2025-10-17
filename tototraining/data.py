from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _load_close_prices(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    if path.suffix == ".npz":
        with np.load(path) as data:
            if "close" in data:
                return data["close"].astype(np.float32)
            return next(iter(data.values())).astype(np.float32)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        if "Close" not in df.columns:
            raise ValueError(f"'Close' column missing in {path}")
        return df["Close"].to_numpy(dtype=np.float32)
    raise ValueError(f"Unsupported file format: {path}")


def _iter_series_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for suffix in (".npy", ".npz", ".csv"):
        yield from sorted(root.rglob(f"*{suffix}"))


@dataclass
class WindowConfig:
    context_length: int
    prediction_length: int
    stride: int = 1


class SlidingWindowDataset(Dataset):
    """
    Simple dataset that turns raw price series into context/target windows for Toto.
    """

    def __init__(self, root: Path, config: WindowConfig):
        self.config = config
        self.windows: list[np.ndarray] = []
        for path in _iter_series_files(root):
            series = _load_close_prices(path)
            horizon = config.context_length + config.prediction_length
            if series.size < horizon:
                continue
            for start in range(0, series.size - horizon + 1, config.stride):
                window = series[start : start + horizon]
                self.windows.append(window)
        if not self.windows:
            raise ValueError(f"No usable windows found in {root}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[idx]
        ctx = window[: self.config.context_length]
        tgt = window[self.config.context_length :]
        context_tensor = torch.from_numpy(ctx).unsqueeze(0)  # (variates=1, time)
        target_tensor = torch.from_numpy(tgt).unsqueeze(0)
        return context_tensor, target_tensor


def _resolve_workers(num_workers: int) -> int:
    if num_workers > 0:
        return num_workers
    cpu_count = os.cpu_count() or 1
    return max(4, cpu_count // 2)


def build_dataloaders(
    train_root: Path,
    val_root: Path | None,
    config: WindowConfig,
    *,
    batch_size: int,
    num_workers: int = -1,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
) -> tuple[DataLoader, DataLoader | None]:
    train_ds = SlidingWindowDataset(train_root, config)
    workers = _resolve_workers(num_workers)
    pin = pin_memory and torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
        if prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )

    val_loader = None
    if val_root is not None:
        val_ds = SlidingWindowDataset(val_root, config)
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

    return train_loader, val_loader
