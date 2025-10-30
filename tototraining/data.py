from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from traininglib.dynamic_batcher import WindowSpec


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
        columns = {col.lower(): col for col in df.columns}
        close_key = columns.get("close")
        if close_key is None:
            raise ValueError(f"'Close' column missing in {path}")
        return df[close_key].to_numpy(dtype=np.float32)
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
        self.series_data: List[np.ndarray] = []
        for path in _iter_series_files(root):
            series = _load_close_prices(path)
            if series.size < config.context_length + config.prediction_length:
                continue
            self.series_data.append(series)
        if not self.series_data:
            raise ValueError(f"No usable windows found in {root}")
        self._default_specs = self.enumerate_window_specs(
            config.context_length, config.prediction_length, config.stride
        )
        if not self._default_specs:
            raise ValueError("Dataset initialisation produced zero windows with the provided configuration.")

    def __len__(self) -> int:
        return len(self._default_specs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self._default_specs[idx]
        return self.load_window(spec, self.config.context_length, self.config.prediction_length)

    @property
    def series_ids(self) -> tuple[int, ...]:
        return tuple(range(len(self.series_data)))

    def enumerate_window_specs(self, context: int, horizon: int, stride: int) -> List[WindowSpec]:
        if context <= 0 or horizon <= 0:
            raise ValueError("context and horizon must be positive for window enumeration.")
        specs: List[WindowSpec] = []
        for series_id, series in enumerate(self.series_data):
            limit = series.shape[0]
            max_context_end = limit - horizon
            if max_context_end < context:
                continue
            for context_end in range(context, max_context_end + 1, stride):
                left = context_end - context
                specs.append(WindowSpec(series_id=series_id, left=left))
        return specs

    def load_window(self, spec: WindowSpec, context: int, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        series = self.series_data[spec.series_id]
        start = spec.left
        ctx_end = start + context
        tgt_end = ctx_end + horizon
        if tgt_end > series.shape[0]:
            raise IndexError(f"Requested window exceeds series length for id={spec.series_id}")
        context_slice = torch.from_numpy(series[start:ctx_end].astype(np.float32, copy=False)).unsqueeze(0)
        target_slice = torch.from_numpy(series[ctx_end:tgt_end].astype(np.float32, copy=False)).unsqueeze(0)
        return context_slice, target_slice

    def collate_windows(
        self,
        samples: Sequence[tuple[torch.Tensor, torch.Tensor]],
        context: int,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not samples:
            raise ValueError("Cannot collate an empty Toto batch.")
        contexts, targets = zip(*samples)
        batch_context = torch.stack(contexts, dim=0)
        batch_target = torch.stack(targets, dim=0)
        return batch_context, batch_target


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
