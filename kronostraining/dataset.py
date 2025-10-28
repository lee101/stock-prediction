from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_utils import (
    ALL_FEATURES,
    TIME_FEATURES,
    iter_symbol_dataframes,
)
from traininglib.dynamic_batcher import WindowSpec


@dataclass
class SymbolWindowIndex:
    symbol: str
    symbol_idx: int
    start: int


class KronosMultiTickerDataset(Dataset):
    """
    Sliding-window dataset that samples across all symbols under ``trainingdata/``.

    The loader mirrors the behaviour of the upstream Kronos training code
    but replaces the Qlib dependency with a direct CSV ingest. Each window
    contains ``lookback + prediction_length + 1`` timesteps so that the
    autoregressive loss can form next-token targets.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        lookback: int,
        prediction_length: int,
        validation_days: int,
        clip: float,
        min_symbol_length: int,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.split = split
        self.lookback = lookback
        self.prediction_length = prediction_length
        self.validation_days = validation_days
        self.clip = clip
        self.window = lookback + prediction_length + 1

        self.symbol_data: List[dict[str, np.ndarray]] = []
        self.indices: List[SymbolWindowIndex] = []
        self._series_ids: List[int] = []

        self._build_index(data_dir, min_symbol_length)
        if not self.indices:
            raise ValueError(f"No samples found for split='{split}'. Check dataset lengths and parameters.")

    def _build_index(self, data_dir: str, min_symbol_length: int) -> None:
        for symbol_idx, (symbol, df) in enumerate(iter_symbol_dataframes(data_dir)):
            if len(df) < min_symbol_length:
                continue

            features = df[list(ALL_FEATURES)].astype(np.float32).to_numpy()
            time_feats = df[list(TIME_FEATURES)].astype(np.float32).to_numpy()
            timestamps = df["timestamps"].to_numpy()

            total_len = len(features)
            train_end = total_len - self.validation_days
            if train_end <= self.window:
                continue

            max_start = total_len - self.window
            if max_start < 0:
                continue

            prev_count = len(self.indices)

            entry = {
                "symbol": symbol,
                "features": features,
                "time": time_feats,
                "timestamps": timestamps,
                "train_end": train_end,
                "total_len": total_len,
            }
            base_index = len(self.symbol_data)

            if self.split == "train":
                last_start = train_end - self.window
                if last_start < 0:
                    last_start = -1
                for start in range(0, last_start + 1):
                    self.indices.append(SymbolWindowIndex(symbol, base_index, start))
            else:
                start_floor = max(0, train_end - self.lookback - self.prediction_length)
                for start in range(start_floor, max_start + 1):
                    pred_start = start + self.lookback
                    pred_end = pred_start + self.prediction_length
                    if pred_end <= train_end:
                        continue
                    self.indices.append(SymbolWindowIndex(symbol, base_index, start))

            added = len(self.indices) - prev_count
            if added > 0:
                self.symbol_data.append(entry)
                self._series_ids.append(base_index)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[idx]
        symbol_record = self.symbol_data[index.symbol_idx]
        start = index.start
        end = start + self.window

        feat_window = symbol_record["features"][start:end]
        time_window = symbol_record["time"][start:end]

        mean = feat_window.mean(axis=0, dtype=np.float32)
        std = feat_window.std(axis=0, dtype=np.float32)
        normed = (feat_window - mean) / (std + 1e-5)
        normed = np.clip(normed, -self.clip, self.clip)

        x_tensor = torch.from_numpy(normed.astype(np.float32))
        t_tensor = torch.from_numpy(time_window.astype(np.float32))
        return x_tensor, t_tensor

    def symbol_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for idx in self.indices:
            counts[idx.symbol] = counts.get(idx.symbol, 0) + 1
        return counts

    # Dynamic window batching helpers -------------------------------------
    @property
    def series_ids(self) -> Tuple[int, ...]:
        return tuple(self._series_ids)

    def enumerate_window_specs(self, context: int, horizon: int, stride: int) -> Iterable[WindowSpec]:
        if context <= 0 or horizon <= 0:
            raise ValueError("context and horizon must be positive integers.")
        specs: List[WindowSpec] = []
        for series_id in self._series_ids:
            entry = self.symbol_data[series_id]
            limit = entry["train_end"] if self.split == "train" else entry["total_len"]
            if limit <= context:
                continue
            max_context_end = limit - horizon
            if max_context_end < context:
                continue
            for context_end in range(context, max_context_end + 1, stride):
                left = context_end - context
                right = context_end + horizon
                if self.split == "train" and right > entry["train_end"]:
                    continue
                specs.append(WindowSpec(series_id=series_id, left=left))
        return specs

    def load_window(self, spec: WindowSpec, context: int, horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.symbol_data[spec.series_id]
        left = spec.left
        right = left + context + horizon
        if right > entry["total_len"]:
            raise IndexError(
                f"Window ({left}, {right}) exceeds available length {entry['total_len']} for series {spec.series_id}"
            )

        feat_window = entry["features"][left:right]
        time_window = entry["time"][left:right]

        mean = feat_window.mean(axis=0, dtype=np.float32)
        std = feat_window.std(axis=0, dtype=np.float32)
        normed = (feat_window - mean) / (std + 1e-5)
        normed = np.clip(normed, -self.clip, self.clip)

        features = torch.from_numpy(normed.astype(np.float32, copy=False))
        stamps = torch.from_numpy(time_window.astype(np.float32, copy=False))
        return features, stamps

    def collate_windows(
        self,
        samples: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        context: int,
        horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not samples:
            raise ValueError("Cannot collate an empty batch.")
        features, stamps = zip(*samples)
        batch_x = torch.stack(features, dim=0)
        batch_stamp = torch.stack(stamps, dim=0)
        return batch_x, batch_stamp
