from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_utils import (
    ALL_FEATURES,
    TIME_FEATURES,
    iter_symbol_dataframes,
)


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
