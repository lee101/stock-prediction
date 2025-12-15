"""V2 Data module - adapts V1 data infrastructure for unified training/inference.

Most of the heavy lifting is done by the existing neuraldailytraining.data module.
This wrapper provides V2-specific configuration and ensures consistency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from V1 - reuse the battle-tested data infrastructure
from neuraldailytraining.data import (
    DEFAULT_FEATURES,
    DailyDataModule,
    DailySymbolDataset,
    FeatureNormalizer,
    MultiSymbolDataset,
    SymbolFrameBuilder,
)

from neuraldailyv2.config import DailyDatasetConfigV2


class DailyDataModuleV2:
    """
    V2 data module wrapper around V1 infrastructure.

    Key additions:
    - V2-specific configuration
    - Explicit feature column tracking for checkpoint compatibility
    """

    def __init__(
        self,
        config: DailyDatasetConfigV2,
        *,
        feature_columns: Optional[List[str]] = None,
    ):
        self.config = config
        self._feature_columns = feature_columns or list(DEFAULT_FEATURES)
        self._v1_module: Optional[DailyDataModule] = None
        self._prepared = False

    @property
    def feature_columns(self) -> List[str]:
        """Get feature columns used by this data module."""
        return self._feature_columns

    @property
    def input_dim(self) -> int:
        """Number of input features."""
        return len(self._feature_columns)

    @property
    def normalizer(self) -> Optional[FeatureNormalizer]:
        """Get feature normalizer after prepare() is called."""
        if self._v1_module is None:
            return None
        return self._v1_module.normalizer

    @property
    def symbols(self) -> Tuple[str, ...]:
        """Get configured symbols."""
        return self.config.symbols

    def prepare(self) -> None:
        """Prepare datasets and normalizer."""
        if self._prepared:
            return

        # Convert V2 config to V1-compatible dict
        v1_kwargs = {
            "symbols": list(self.config.symbols),
            "data_root": self.config.data_root,
            "forecast_cache_dir": self.config.forecast_cache_dir,
            "sequence_length": self.config.sequence_length,
            "val_fraction": self.config.val_fraction,
            "min_history_days": self.config.min_history_days,
            "require_forecasts": self.config.require_forecasts,
            "forecast_fill_strategy": self.config.forecast_fill_strategy,
            "forecast_cache_writeback": self.config.forecast_cache_writeback,
            "feature_columns": self._feature_columns,
            "validation_days": self.config.validation_days,
            "symbol_dropout_rate": self.config.symbol_dropout_rate,
            "crypto_only": self.config.crypto_only,
            "include_weekly_features": self.config.include_weekly_features,
            "grouping_strategy": self.config.grouping_strategy,
            "correlation_min_corr": self.config.correlation_min_corr,
            "correlation_max_group_size": self.config.correlation_max_group_size,
            "correlation_window_days": self.config.correlation_window_days,
            "correlation_min_overlap": self.config.correlation_min_overlap,
            "split_crypto_groups": self.config.split_crypto_groups,
        }

        # Add optional fields
        if self.config.start_date:
            v1_kwargs["start_date"] = self.config.start_date
        if self.config.end_date:
            v1_kwargs["end_date"] = self.config.end_date
        if self.config.exclude_symbols:
            v1_kwargs["exclude_symbols"] = self.config.exclude_symbols
        if self.config.exclude_symbols_file:
            v1_kwargs["exclude_symbols_file"] = self.config.exclude_symbols_file

        # Create V1 module
        from neuraldailytraining.config import DailyDatasetConfig as V1Config
        v1_config = V1Config(**v1_kwargs)
        self._v1_module = DailyDataModule(v1_config)
        self._v1_module._prepare()
        self._prepared = True

        # Update feature columns from V1 module (in case they were auto-detected)
        self._feature_columns = self._v1_module.feature_columns

    def get_train_dataset(self) -> MultiSymbolDataset:
        """Get training dataset."""
        if not self._prepared:
            self.prepare()
        return self._v1_module.train_dataset

    def get_val_dataset(self) -> MultiSymbolDataset:
        """Get validation dataset."""
        if not self._prepared:
            self.prepare()
        return self._v1_module.val_dataset

    def get_train_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.get_train_dataset(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_val_loader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.get_val_dataset(),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_symbol_group_ids(self) -> Dict[str, int]:
        """Get mapping of symbol -> group ID for cross-attention."""
        if not self._prepared:
            self.prepare()
        return self._v1_module.symbol_to_group_id


class SymbolFrameBuilderV2:
    """
    V2 wrapper around V1 SymbolFrameBuilder.

    Ensures consistent feature engineering between training and inference.
    """

    def __init__(
        self,
        data_root: Path,
        forecast_cache_dir: Path,
        *,
        feature_columns: Optional[List[str]] = None,
        require_forecasts: bool = False,
        forecast_fill_strategy: str = "persistence",
        forecast_cache_writeback: bool = True,
        include_weekly_features: bool = True,
    ):
        # Build V1 config to pass to SymbolFrameBuilder
        from neuraldailytraining.config import DailyDatasetConfig as V1Config
        v1_config = V1Config(
            symbols=(),  # Not used by builder
            data_root=data_root,
            forecast_cache_dir=forecast_cache_dir,
            require_forecasts=require_forecasts,
            forecast_fill_strategy=forecast_fill_strategy,
            forecast_cache_writeback=forecast_cache_writeback,
            include_weekly_features=include_weekly_features,
        )
        feature_cols = feature_columns or list(DEFAULT_FEATURES)
        self._builder = SymbolFrameBuilder(v1_config, feature_cols)
        self.feature_columns = list(self._builder.feature_columns)

    def build(self, symbol: str, *, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Build feature frame for a symbol.

        Args:
            symbol: Stock/crypto symbol
            as_of: Optional cutoff date for backtesting

        Returns:
            DataFrame with all features and price columns
        """
        frame = self._builder.build(symbol)

        if as_of is not None and not frame.empty:
            frame = frame[frame["date"] <= as_of].copy()

        return frame

    def build_batch(
        self,
        symbols: List[str],
        sequence_length: int,
        normalizer: FeatureNormalizer,
        *,
        as_of: Optional[pd.Timestamp] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Build a batch of normalized feature tensors for inference.

        Args:
            symbols: List of symbols to include
            sequence_length: Number of timesteps per symbol
            normalizer: Fitted feature normalizer
            as_of: Optional cutoff date

        Returns:
            Dict with batched tensors or None if no valid data
        """
        feature_list = []
        high_list = []
        low_list = []
        close_list = []
        ref_close_list = []
        chronos_high_list = []
        chronos_low_list = []
        asset_class_list = []
        symbol_indices = []

        for idx, symbol in enumerate(symbols):
            frame = self.build(symbol, as_of=as_of)
            if frame.empty or len(frame) < sequence_length:
                continue

            # Take last sequence_length rows
            window = frame.iloc[-sequence_length:].reset_index(drop=True)

            # Extract feature matrix
            feat_cols = [c for c in self.feature_columns if c in window.columns]
            features = window[feat_cols].values

            # Normalize
            features = normalizer.transform(features)

            # Extract price columns
            high = window["high"].values
            low = window["low"].values
            close = window["close"].values
            ref_close = window.get("reference_close", window["close"]).values
            chronos_high = window.get("chronos_high", window["high"]).values
            chronos_low = window.get("chronos_low", window["low"]).values

            # Determine asset class
            is_crypto = symbol.upper().endswith("USD") or "-USD" in symbol.upper()
            asset_class = 1.0 if is_crypto else 0.0

            feature_list.append(torch.tensor(features, dtype=torch.float32))
            high_list.append(torch.tensor(high, dtype=torch.float32))
            low_list.append(torch.tensor(low, dtype=torch.float32))
            close_list.append(torch.tensor(close, dtype=torch.float32))
            ref_close_list.append(torch.tensor(ref_close, dtype=torch.float32))
            chronos_high_list.append(torch.tensor(chronos_high, dtype=torch.float32))
            chronos_low_list.append(torch.tensor(chronos_low, dtype=torch.float32))
            asset_class_list.append(asset_class)
            symbol_indices.append(idx)

        if not feature_list:
            return None

        return {
            "features": torch.stack(feature_list),
            "high": torch.stack(high_list),
            "low": torch.stack(low_list),
            "close": torch.stack(close_list),
            "reference_close": torch.stack(ref_close_list),
            "chronos_high": torch.stack(chronos_high_list),
            "chronos_low": torch.stack(chronos_low_list),
            "asset_class": torch.tensor(asset_class_list, dtype=torch.float32),
            "symbol_indices": symbol_indices,
        }
