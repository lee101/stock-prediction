"""Bridges Kronos path-summary features into differentiable market training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from differentiable_market.config import DataConfig

from .config import KronosFeatureConfig
from .kronos_embedder import KronosEmbedder, KronosFeatureSpec, precompute_feature_table

PRICE_COLUMNS = ("open", "high", "low", "close")
DEFAULT_VOLUME_COL = "volume"
DEFAULT_AMOUNT_COL = "amount"


def _load_symbol_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns and "timestamps" not in df.columns:
        raise ValueError(f"{path} missing timestamp column")
    ts_col = "timestamp" if "timestamp" in df.columns else "timestamps"
    df = df.rename(columns={ts_col: "timestamp"})
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"{path} missing price column '{col}'")
    if DEFAULT_VOLUME_COL not in df.columns:
        df[DEFAULT_VOLUME_COL] = 0.0
    df = df[["timestamp", *PRICE_COLUMNS, DEFAULT_VOLUME_COL]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df = df.set_index("timestamp").astype(np.float32)
    mean_price = df[list(PRICE_COLUMNS)].mean(axis=1)
    df[DEFAULT_AMOUNT_COL] = (mean_price * df[DEFAULT_VOLUME_COL]).astype(np.float32)
    return df


@dataclass(slots=True)
class KronosFeatureAdapterCache:
    features: torch.Tensor
    symbols: Sequence[str]
    index: pd.DatetimeIndex


class KronosFeatureAdapter:
    def __init__(
        self,
        cfg: KronosFeatureConfig,
        data_cfg: DataConfig,
        symbols: Sequence[str],
        index: pd.DatetimeIndex,
        *,
        embedder: KronosEmbedder | None = None,
        frame_override: Dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.symbols = tuple(symbols)
        self.index = index
        self._embedder = embedder
        self._frame_override = frame_override or {}
        self._cache: Optional[KronosFeatureAdapterCache] = None

    @property
    def embedder(self) -> KronosEmbedder:
        if self._embedder is None:
            feature_spec = KronosFeatureSpec(
                horizons=self.cfg.horizons,
                quantiles=self.cfg.quantiles,
                include_path_stats=self.cfg.include_path_stats,
            )
            device = self.cfg.device if self.cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            self._embedder = KronosEmbedder(
                model_id=self.cfg.model_path,
                tokenizer_id=self.cfg.tokenizer_path,
                device=device,
                max_context=self.cfg.context_length,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                sample_count=self.cfg.sample_count,
                sample_chunk=self.cfg.sample_chunk,
                top_k=self.cfg.top_k,
                clip=self.cfg.clip,
                feature_spec=feature_spec,
                bf16=self.cfg.bf16,
                compile_model=self.cfg.compile,
            )
        return self._embedder

    def _load_frames(self) -> Dict[str, pd.DataFrame]:
        frames: Dict[str, pd.DataFrame] = {}
        root = Path(self.data_cfg.root)
        for symbol in self.symbols:
            if symbol in self._frame_override:
                frame = self._frame_override[symbol]
            else:
                path = root / f"{symbol}.csv"
                if not path.exists():
                    raise FileNotFoundError(f"Expected CSV for symbol {symbol} at {path}")
                frame = _load_symbol_frame(path)
            frame = frame.reindex(self.index)
            frame[list(PRICE_COLUMNS)] = frame[list(PRICE_COLUMNS)].interpolate(method="time").ffill().bfill()
            frame[DEFAULT_VOLUME_COL] = frame[DEFAULT_VOLUME_COL].fillna(0.0)
            frame[DEFAULT_AMOUNT_COL] = frame[DEFAULT_AMOUNT_COL].fillna(0.0)
            frames[symbol] = frame
        return frames

    def compute(self) -> KronosFeatureAdapterCache:
        if self._cache is not None:
            return self._cache
        frames = self._load_frames()
        feature_arrays: list[np.ndarray] = []
        horizon = max(self.cfg.horizons) if self.cfg.horizons else 1
        for idx, symbol in enumerate(self.symbols):
            frame = frames[symbol]
            numeric = frame.reset_index()
            if "timestamp" not in numeric.columns:
                numeric = numeric.rename(columns={"index": "timestamp"})
            ts_series = numeric["timestamp"]
            data_df = numeric[[*PRICE_COLUMNS, DEFAULT_VOLUME_COL, DEFAULT_AMOUNT_COL]].rename(
                columns={
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    DEFAULT_VOLUME_COL: "volume",
                    DEFAULT_AMOUNT_COL: "amount",
                }
            )
            feat_df = precompute_feature_table(
                df=data_df,
                ts=ts_series,
                lookback=self.cfg.context_length,
                horizon_main=horizon,
                embedder=self.embedder,
            )
            feat_df = feat_df.reindex(self.index).fillna(0.0)
            feature_arrays.append(feat_df.to_numpy(dtype=np.float32))
            print(f"[kronos-adapter] computed features for {symbol} ({idx + 1}/{len(self.symbols)})")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not feature_arrays:
            raise ValueError("No Kronos features computed")
        stacked = np.stack(feature_arrays, axis=1)
        tensor = torch.from_numpy(stacked)
        self._cache = KronosFeatureAdapterCache(features=tensor, symbols=self.symbols, index=self.index)
        return self._cache

    def features_tensor(self, *, add_cash: bool, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        cache = self.compute()
        feat = cache.features.to(dtype=dtype)
        if add_cash:
            zeros = torch.zeros(feat.shape[0], 1, feat.shape[2], dtype=dtype)
            feat = torch.cat([feat, zeros], dim=1)
        return feat
