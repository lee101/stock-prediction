from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from differentiable_market.config import DataConfig
from differentiable_market.utils import resolve_device

from .config import KronosFeatureConfig


PRICE_COLUMNS: Tuple[str, ...] = ("open", "high", "low", "close")
VOLUME_COLUMN = "volume"


def _load_symbol_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    if "timestamp" not in df.columns and "timestamps" not in df.columns:
        raise ValueError(f"{path} missing timestamp column")
    ts_col = "timestamp" if "timestamp" in df.columns else "timestamps"
    df = df.rename(columns={ts_col: "timestamp"})
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"{path} missing price column '{col}'")
    if VOLUME_COLUMN not in df.columns:
        df[VOLUME_COLUMN] = 0.0
    df = df[["timestamp", *PRICE_COLUMNS, VOLUME_COLUMN]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    df = df.set_index("timestamp")
    df = df.astype(np.float32)
    mean_price = df[list(PRICE_COLUMNS)].mean(axis=1)
    df["amount"] = (mean_price * df[VOLUME_COLUMN]).astype(np.float32)
    return df


def _time_feature_matrix(index: pd.DatetimeIndex) -> torch.Tensor:
    df = pd.DataFrame(
        {
            "minute": index.minute.astype(np.float32),
            "hour": index.hour.astype(np.float32),
            "weekday": index.weekday.astype(np.float32),
            "day": index.day.astype(np.float32),
            "month": index.month.astype(np.float32),
        }
    )
    return torch.from_numpy(df.to_numpy(dtype=np.float32))


class KronosEmbeddingAdapter:
    """Compute frozen Kronos embeddings for aligned market data."""

    def __init__(
        self,
        cfg: KronosFeatureConfig,
        data_cfg: DataConfig,
        symbols: Sequence[str],
        index: pd.DatetimeIndex,
        *,
        tokenizer=None,
        model=None,
    ) -> None:
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.symbols = tuple(symbols)
        self.index = index
        self.asset_count = len(self.symbols)
        self.device = resolve_device(cfg.device)
        self._price_volume = self._load_price_volume_tensor()
        self._time_features = _time_feature_matrix(index)
        self._tokenizer = tokenizer
        self._model = model
        self._cache: Dict[Tuple[int, int, bool], torch.Tensor] = {}
        self._init_models()

    @property
    def embedding_dim(self) -> int:
        dims = 0
        if self.cfg.embedding_mode in ("context", "both"):
            dims += getattr(self._model, "d_model")
        if self.cfg.embedding_mode in ("bits", "both"):
            dims += getattr(self._tokenizer, "codebook_dim")
        return dims

    def _init_models(self) -> None:
        if self._tokenizer is None or self._model is None:
            from external.kronos.model import Kronos, KronosTokenizer

            if self._tokenizer is None:
                self._tokenizer = KronosTokenizer.from_pretrained(self.cfg.tokenizer_path)
            if self._model is None:
                self._model = Kronos.from_pretrained(self.cfg.model_path)

        self._tokenizer.to(self.device)
        self._model.to(self.device)
        self._tokenizer.eval()
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad_(False)

    def _resolve_symbol_paths(self) -> Dict[str, Path]:
        path_map: Dict[str, Path] = {}
        root = Path(self.data_cfg.root)
        for candidate in sorted(root.glob(self.data_cfg.glob)):
            symbol = candidate.stem.upper()
            path_map[symbol] = candidate
        return path_map

    def _load_price_volume_tensor(self) -> torch.Tensor:
        path_map = self._resolve_symbol_paths()
        aligned: list[np.ndarray] = []
        for symbol in self.symbols:
            path = path_map.get(symbol.upper())
            if path is None:
                raise FileNotFoundError(f"No data file found for symbol {symbol} under {self.data_cfg.root}")
            frame = _load_symbol_frame(path)
            frame = frame.reindex(self.index)
            frame[list(PRICE_COLUMNS)] = (
                frame[list(PRICE_COLUMNS)].interpolate(method="time").ffill().bfill()
            )
            frame[VOLUME_COLUMN] = frame[VOLUME_COLUMN].fillna(0.0)
            frame["amount"] = frame["amount"].fillna(0.0)
            values = frame[list(PRICE_COLUMNS) + [VOLUME_COLUMN, "amount"]].to_numpy(dtype=np.float32)
            aligned.append(values)
        stacked = np.stack(aligned, axis=1)  # [T, A, F]
        return torch.from_numpy(stacked)

    def embed_slice(self, start: int, length: int, *, add_cash: bool) -> torch.Tensor:
        key = (start, length, add_cash)
        if key in self._cache:
            return self._cache[key]
        if length <= 1:
            empty = torch.zeros((0, self.asset_count + (1 if add_cash else 0), self.embedding_dim), dtype=torch.float32)
            self._cache[key] = empty
            return empty
        pv_slice = self._price_volume[start : start + length]
        time_slice = self._time_features[start : start + length]
        embeddings = self._encode_slice(pv_slice, time_slice)
        if add_cash:
            zeros = torch.zeros((embeddings.shape[0], 1, embeddings.shape[2]), dtype=embeddings.dtype)
            embeddings = torch.cat([embeddings, zeros], dim=1)
        self._cache[key] = embeddings
        return embeddings

    def _encode_slice(self, pv_slice: torch.Tensor, time_slice: torch.Tensor) -> torch.Tensor:
        seq_len = pv_slice.shape[0]
        context = self.cfg.context_length
        if seq_len <= 1:
            return torch.zeros((0, self.asset_count, self.embedding_dim), dtype=torch.float32)

        asset_windows = []
        for asset_idx in range(self.asset_count):
            series = pv_slice[:, asset_idx, :]
            windows = self._window_series(series, context)
            asset_windows.append(windows[:-1].contiguous())
        asset_windows_tensor = torch.stack(asset_windows, dim=1)  # [T-1, A, context, F]

        time_windows = self._window_series(time_slice, context)[:-1].contiguous()  # [T-1, context, time_F]
        samples_flat = asset_windows_tensor.permute(1, 0, 2, 3).reshape(-1, context, pv_slice.shape[2])
        time_flat = time_windows.unsqueeze(1).repeat(1, self.asset_count, 1, 1).reshape(-1, context, time_slice.shape[1])

        embeddings = self._embed_windows(samples_flat, time_flat)
        total = embeddings.shape[0]
        per_asset = seq_len - 1
        embeddings = embeddings.view(self.asset_count, per_asset, -1).permute(1, 0, 2).contiguous()
        return embeddings

    def _window_series(self, series: torch.Tensor, context: int) -> torch.Tensor:
        if context == 1:
            return series.unsqueeze(1)
        first = series[0:1].repeat(context - 1, 1)
        padded = torch.cat([first, series], dim=0)
        windows = padded.unfold(0, context, 1)
        return windows

    def _embed_windows(self, samples: torch.Tensor, stamps: torch.Tensor) -> torch.Tensor:
        if samples.numel() == 0:
            return torch.zeros((0, self.embedding_dim), dtype=torch.float32)

        batch_size = self.cfg.batch_size
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, samples.shape[0], batch_size):
                end = min(start + batch_size, samples.shape[0])
                window_batch = samples[start:end]
                normed = self._normalise_windows(window_batch)
                stamp = stamps[start:end].to(self.device, dtype=torch.float32, non_blocking=True)
                token_pair = self._tokenizer.encode(normed, half=True)
                s1_ids, s2_ids = self._ensure_token_pair(token_pair)

                embeds: list[torch.Tensor] = []
                if self.cfg.embedding_mode in ("context", "both"):
                    _, context = self._model.decode_s1(s1_ids, s2_ids, stamp=stamp)
                    embeds.append(context[:, -1, :].to(torch.float32))
                if self.cfg.embedding_mode in ("bits", "both"):
                    bit_tensor = self._tokenizer.indices_to_bits(token_pair, half=True)
                    embeds.append(bit_tensor[:, -1, :].to(torch.float32))
                if not embeds:
                    raise RuntimeError("Kronos embedding mode produced no features")
                outputs.append(torch.cat(embeds, dim=-1).cpu())
        return torch.cat(outputs, dim=0)

    def _normalise_windows(self, windows: torch.Tensor) -> torch.Tensor:
        windows = windows.to(self.device, dtype=torch.float32, non_blocking=True)
        mean = windows.mean(dim=1, keepdim=True)
        std = windows.std(dim=1, unbiased=False, keepdim=True)
        normed = (windows - mean) / (std + 1e-5)
        normed = torch.clamp(normed, -self.cfg.clip, self.cfg.clip)
        return normed

    def _ensure_token_pair(self, token_pair) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(token_pair, (tuple, list)):
            s1_ids = token_pair[0].to(self.device, non_blocking=True)
            s2_ids = token_pair[1].to(self.device, non_blocking=True)
        else:
            raise TypeError("Expected KronosTokenizer.encode(..., half=True) to return a pair of token tensors")
        return s1_ids, s2_ids
