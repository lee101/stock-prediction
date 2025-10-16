#!/usr/bin/env python3
"""
Utilities for enriching price data with Amazon Toto forecasts.

The generator attempts to use the real Toto model when the dependency stack is
available; otherwise it falls back to light-weight statistical approximations so
the training pipeline remains usable even without the Toto runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple, List
import hashlib
import json
import os

import numpy as np
import torch

try:
    from totoembedding.embedding_model import TotoEmbeddingModel  # type: ignore

    _HAS_TOTO = True
except Exception:  # pragma: no cover - Toto is optional
    TotoEmbeddingModel = None  # type: ignore
    _HAS_TOTO = False


@dataclass
class TotoOptions:
    """Configuration for Toto forecast feature generation."""

    use_toto: bool = True
    horizon: int = 8
    context_length: int = 60
    num_samples: int = 2048
    toto_model_id: str = "Datadog/Toto-Open-Base-1.0"
    toto_device: str = "cuda"
    target_columns: Sequence[str] = field(
        default_factory=lambda: ("close", "open", "high", "low")
    )
    cache_dir: Optional[str] = "hftraining/cache/toto"
    enable_cache: bool = True


class TotoFeatureGenerator:
    """
    Create forward-looking features derived from Amazon Toto forecasts.

    The generator produces a matrix whose width equals
    (2 * horizon * len(target_columns)), containing the forecast means and
    standard deviations for each requested target line.
    """

    def __init__(self, options: TotoOptions):
        self.options = options
        self._target_columns = [col.lower() for col in options.target_columns]
        self._toto_model: Optional[TotoEmbeddingModel] = None
        self._cache_dir: Optional[Path] = None
        if self.options.enable_cache and self.options.cache_dir:
            cache_path = Path(os.path.expanduser(self.options.cache_dir))
            cache_path.mkdir(parents=True, exist_ok=True)
            self._cache_dir = cache_path

    @property
    def uses_real_toto(self) -> bool:
        return self._toto_model is not None

    def _ensure_model(self, feature_dim: int) -> Optional[TotoEmbeddingModel]:
        """Instantiate Toto backbone lazily once feature dimensionality is known."""
        if not self.options.use_toto or not _HAS_TOTO:
            return None
        if (
            self._toto_model is None
            or getattr(self._toto_model, "input_feature_dim", None) != feature_dim
        ):
            try:
                self._toto_model = TotoEmbeddingModel(
                    use_toto=True,
                    toto_model_id=self.options.toto_model_id,
                    toto_device=self.options.toto_device,
                    toto_horizon=self.options.horizon,
                    toto_num_samples=self.options.num_samples,
                    freeze_backbone=True,
                    input_feature_dim=feature_dim,
                )
                target_device = torch.device(
                    self.options.toto_device if torch.cuda.is_available() else "cpu"
                )
                try:
                    self._toto_model.to(target_device)
                except Exception:
                    pass
                self._toto_model.eval()
            except Exception:
                self._toto_model = None
        return self._toto_model

    def compute_features(
        self,
        price_matrix: np.ndarray,
        column_order: Sequence[str],
        symbol_prefix: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate Toto forecast features for a price matrix.

        Args:
            price_matrix: Array shaped [timesteps, features] containing OHLCV data.
            column_order: Column names corresponding to the feature axis in `price_matrix`.
            symbol_prefix: Optional symbol identifier to prefix generated column names.

        Returns:
            Tuple of (feature_matrix, column_names)
        """
        if price_matrix.ndim != 2:
            raise ValueError(
                f"Expected price matrix with shape [timesteps, features], "
                f"received {price_matrix.shape}"
            )

        column_map = {col.lower(): idx for idx, col in enumerate(column_order)}
        active_targets = [
            target for target in self._target_columns if target in column_map
        ]

        feature_dim = price_matrix.shape[1]
        model = self._ensure_model(feature_dim)

        t_steps = price_matrix.shape[0]
        if not active_targets:
            return np.zeros((t_steps, 0), dtype=np.float32), []

        column_blocks: List[np.ndarray] = []
        column_names: List[str] = []

        for column_name in active_targets:
            column_index = column_map[column_name]

            cached = self._load_from_cache(
                price_matrix,
                column_order,
                column_name,
                symbol_prefix,
            )
            if cached is not None:
                col_features, cached_names = cached
                column_blocks.append(col_features)
                column_names.extend(cached_names)
                continue

            if model is None:
                col_features, col_dim = self._compute_statistical_forecasts(
                    price_matrix, column_index
                )
            else:
                model.series_feature_index = column_index
                col_features, col_dim = self._compute_toto_forecasts(price_matrix, model)

            column_blocks.append(col_features)

            prefix = symbol_prefix.lower() if symbol_prefix else "toto"
            if col_dim == 2 * self.options.horizon:
                column_names.extend(
                    [
                        f"{prefix}_{column_name}_toto_mean_t+{step+1}"
                        for step in range(self.options.horizon)
                    ]
                    + [
                        f"{prefix}_{column_name}_toto_std_t+{step+1}"
                        for step in range(self.options.horizon)
                    ]
                )
            else:
                column_names.extend(
                    [
                        f"{prefix}_{column_name}_toto_emb_{idx+1}"
                        for idx in range(col_dim)
                    ]
                )

            self._save_to_cache(
                price_matrix,
                column_order,
                column_name,
                symbol_prefix,
                col_features,
                column_names[-col_features.shape[1]:] if column_names else column_names,
            )

        features = np.concatenate(column_blocks, axis=1) if column_blocks else np.zeros((t_steps, 0), dtype=np.float32)

        return features, column_names

    def _cache_key(
        self,
        price_matrix: np.ndarray,
        column_order: Sequence[str],
        column_name: str,
        symbol_prefix: Optional[str],
    ) -> str:
        """Stable hash for the current feature request."""
        payload = {
            "columns": [col.lower() for col in column_order],
            "target": column_name.lower(),
            "symbol": (symbol_prefix or "").lower(),
            "horizon": self.options.horizon,
            "context_length": self.options.context_length,
            "num_samples": self.options.num_samples,
            "use_toto": self.options.use_toto,
        }
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        hasher.update(np.ascontiguousarray(price_matrix).view(np.uint8))
        return hasher.hexdigest()

    def _cache_path(
        self,
        price_matrix: np.ndarray,
        column_order: Sequence[str],
        column_name: str,
        symbol_prefix: Optional[str],
    ) -> Optional[Path]:
        if not (self.options.enable_cache and self._cache_dir):
            return None
        key = self._cache_key(price_matrix, column_order, column_name, symbol_prefix)
        return self._cache_dir / f"{key}.npz"

    def _load_from_cache(
        self,
        price_matrix: np.ndarray,
        column_order: Sequence[str],
        column_name: str,
        symbol_prefix: Optional[str],
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        cache_path = self._cache_path(price_matrix, column_order, column_name, symbol_prefix)
        if cache_path is None or not cache_path.exists():
            return None
        try:
            with np.load(cache_path, allow_pickle=True) as cached:
                features = cached["features"]
                names = cached["column_names"].tolist()
            return features, names
        except Exception:
            return None

    def _save_to_cache(
        self,
        price_matrix: np.ndarray,
        column_order: Sequence[str],
        column_name: str,
        symbol_prefix: Optional[str],
        features: np.ndarray,
        names: Sequence[str],
    ) -> None:
        cache_path = self._cache_path(price_matrix, column_order, column_name, symbol_prefix)
        if cache_path is None:
            return
        try:
            np.savez_compressed(
                cache_path,
                features=features,
                column_names=np.asarray(list(names), dtype=object),
            )
        except Exception:
            pass

    def _compute_toto_forecasts(
        self,
        price_matrix: np.ndarray,
        model: TotoEmbeddingModel,
    ) -> np.ndarray:
        """Use the Toto forecaster (or its fallbacks) to derive forecast stats."""
        context = self.options.context_length
        indices = list(range(context, price_matrix.shape[0]))
        column_dim = getattr(model, "backbone_dim", 2 * self.options.horizon)
        features = np.zeros((price_matrix.shape[0], column_dim), dtype=np.float32)

        if not indices:
            return features, column_dim

        device = torch.device(
            model.toto_device if torch.cuda.is_available() else "cpu"
        )
        batch_size = 16

        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                windows = [
                    price_matrix[i - context : i, :].astype(np.float32)
                    for i in batch_indices
                ]
                price_tensor = torch.from_numpy(np.stack(windows, axis=0)).to(device)
                stats = model._toto_forecast_stats(price_tensor)
                stats_np = stats.cpu().numpy().astype(np.float32)
                if stats_np.shape[1] != column_dim:
                    column_dim = stats_np.shape[1]
                    if features.shape[1] != column_dim:
                        expanded = np.zeros((features.shape[0], column_dim), dtype=np.float32)
                        copy_width = min(features.shape[1], column_dim)
                        if copy_width > 0:
                            expanded[:, :copy_width] = features[:, :copy_width]
                        features = expanded
                features[batch_indices, : stats_np.shape[1]] = stats_np

        return features, column_dim

    def _compute_statistical_forecasts(
        self,
        price_matrix: np.ndarray,
        column_index: int,
    ) -> np.ndarray:
        """
        Lightweight fallback that mimics Toto outputs using rolling statistics.

        The fallback uses exponentially weighted moving averages to estimate
        forward returns and volatility so gradient flow remains intact.
        """
        series = price_matrix[:, column_index]
        per_column_dim = 2 * self.options.horizon
        out = np.zeros((price_matrix.shape[0], per_column_dim), dtype=np.float32)

        context = self.options.context_length
        horizon = self.options.horizon

        if series.ndim != 1:
            series = np.asarray(series).reshape(-1)

        log_prices = np.log(np.clip(series, a_min=1e-6, a_max=None))
        returns = np.diff(log_prices, prepend=log_prices[0])

        for idx in range(context, len(series)):
            window = returns[idx - context : idx]
            if window.size == 0:
                continue
            mean_ret = np.mean(window)
            vol = np.std(window) + 1e-6
            horizon_means = mean_ret * np.arange(1, horizon + 1, dtype=np.float32)
            horizon_stds = np.sqrt(np.arange(1, horizon + 1, dtype=np.float32)) * vol
            out[idx, :horizon] = horizon_means
            out[idx, horizon:] = horizon_stds

        return out, per_column_dim


def append_toto_columns(
    dataframe,
    feature_matrix: np.ndarray,
    column_names: Optional[Sequence[str]] = None,
):
    """
    Attach Toto forecast features to a pandas DataFrame.

    Mutates the DataFrame in-place for convenience.
    """
    if feature_matrix.size == 0:
        return

    if column_names is None:
        horizon = feature_matrix.shape[1] // 2
        column_names = [
            f"toto_mean_t+{step+1}" for step in range(horizon)
        ] + [
            f"toto_std_t+{step+1}" for step in range(horizon)
        ]

    if len(column_names) != feature_matrix.shape[1]:
        raise ValueError(
            "Provided column names do not match Toto feature dimensionality."
        )

    for name, column in zip(column_names, feature_matrix.T):
        dataframe[name] = column
