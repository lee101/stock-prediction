"""V3 Timed: Data module with lookahead support for trade episode simulation.

Key V3 additions:
- lookahead_days: Provides future OHLC data for simulation
- Trade episode simulation needs to see N days into the future
- Data samples include history + lookahead windows
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Import from V1 - reuse the battle-tested data infrastructure
from neuraldailytraining.data import (
    DEFAULT_FEATURES,
    DailyDataModule,
    DailySymbolDataset,
    FeatureNormalizer,
    MultiSymbolDataset,
    SymbolFrameBuilder,
)

from neuraldailyv3timed.config import DailyDatasetConfigV3


class MultiSymbolDatasetV3(Dataset):
    """
    V3 Dataset that provides lookahead data for trade episode simulation.

    Each sample includes:
    - features: (seq_len, num_features) - normalized features up to decision point
    - future_highs: (lookahead_days + 1,) - highs for simulation
    - future_lows: (lookahead_days + 1,) - lows for simulation
    - future_closes: (lookahead_days + 1,) - closes for simulation
    - asset_class: scalar - 0 for stocks, 1 for crypto
    - reference_close, chronos_high, chronos_low: (seq_len,) - for price decoding
    """

    def __init__(
        self,
        symbol_datasets: List[DailySymbolDataset],
        sequence_length: int,
        lookahead_days: int = 10,
        *,
        permutation_rate: float = 0.0,
    ):
        self.symbol_datasets = symbol_datasets
        self.sequence_length = sequence_length
        self.lookahead_days = lookahead_days
        self.permutation_rate = permutation_rate
        self.total_length = sequence_length + lookahead_days

        # Build index: (dataset_idx, start_idx) for valid windows
        self._indices: List[Tuple[int, int]] = []
        for ds_idx, ds in enumerate(symbol_datasets):
            # Need total_length consecutive rows
            max_start = len(ds) - self.total_length
            for start in range(max(0, max_start)):
                self._indices.append((ds_idx, start))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_idx, start_idx = self._indices[idx]
        ds = self.symbol_datasets[ds_idx]

        # Get window: history + lookahead
        end_idx = start_idx + self.total_length

        # Extract features for history portion (up to decision point)
        history_end = start_idx + self.sequence_length
        features = ds.features[start_idx:history_end]

        # Extract OHLC for history
        high = ds.high[start_idx:history_end]
        low = ds.low[start_idx:history_end]
        close = ds.close[start_idx:history_end]
        reference_close = ds.reference_close[start_idx:history_end]
        chronos_high = ds.chronos_high[start_idx:history_end]
        chronos_low = ds.chronos_low[start_idx:history_end]

        # Extract OHLC for lookahead (day 0 = last history day, days 1-N = future)
        # Day 0 is the entry day, days 1 to lookahead_days are the hold period
        lookahead_start = history_end - 1  # Include the decision day
        lookahead_end = lookahead_start + self.lookahead_days + 1
        future_highs = ds.high[lookahead_start:lookahead_end]
        future_lows = ds.low[lookahead_start:lookahead_end]
        future_closes = ds.close[lookahead_start:lookahead_end]

        # Asset class (stored as asset_flag in V1)
        asset_class = torch.tensor(ds.asset_flag, dtype=torch.float32)

        # Group ID for cross-attention
        group_id = torch.tensor(getattr(ds, 'group_id', 0), dtype=torch.long)

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "high": torch.tensor(high, dtype=torch.float32),
            "low": torch.tensor(low, dtype=torch.float32),
            "close": torch.tensor(close, dtype=torch.float32),
            "reference_close": torch.tensor(reference_close, dtype=torch.float32),
            "chronos_high": torch.tensor(chronos_high, dtype=torch.float32),
            "chronos_low": torch.tensor(chronos_low, dtype=torch.float32),
            "future_highs": torch.tensor(future_highs, dtype=torch.float32),
            "future_lows": torch.tensor(future_lows, dtype=torch.float32),
            "future_closes": torch.tensor(future_closes, dtype=torch.float32),
            "asset_class": asset_class,
            "group_id": group_id,
        }


class DailyDataModuleV3:
    """
    V3 data module with lookahead support for trade episode simulation.

    Key additions over V2:
    - lookahead_days parameter for trade episode simulation
    - Samples include future OHLC data for simulation
    """

    def __init__(
        self,
        config: DailyDatasetConfigV3,
        *,
        feature_columns: Optional[List[str]] = None,
    ):
        self.config = config
        self._feature_columns = feature_columns or list(DEFAULT_FEATURES)
        self._v1_module: Optional[DailyDataModule] = None
        self._train_dataset: Optional[MultiSymbolDatasetV3] = None
        self._val_dataset: Optional[MultiSymbolDatasetV3] = None
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

        # Convert V3 config to V1-compatible dict
        v1_kwargs = {
            "symbols": list(self.config.symbols),
            "data_root": self.config.data_root,
            "forecast_cache_dir": self.config.forecast_cache_dir,
            "sequence_length": self.config.sequence_length + self.config.lookahead_days,  # Total needed
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

        # Create V3 datasets with lookahead
        self._train_dataset = MultiSymbolDatasetV3(
            self._v1_module.train_dataset.datasets,
            sequence_length=self.config.sequence_length,
            lookahead_days=self.config.lookahead_days,
        )
        self._val_dataset = MultiSymbolDatasetV3(
            self._v1_module.val_dataset.datasets,
            sequence_length=self.config.sequence_length,
            lookahead_days=self.config.lookahead_days,
        )

    def get_train_dataset(self) -> MultiSymbolDatasetV3:
        """Get training dataset."""
        if not self._prepared:
            self.prepare()
        return self._train_dataset

    def get_val_dataset(self) -> MultiSymbolDatasetV3:
        """Get validation dataset."""
        if not self._prepared:
            self.prepare()
        return self._val_dataset

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


class SymbolFrameBuilderV3:
    """
    V3 wrapper around V1 SymbolFrameBuilder.

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
