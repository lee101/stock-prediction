"""V5 Data module - Portfolio-centric data loading.

Key additions:
- Returns per-asset returns for portfolio simulation
- Groups assets together for batch training
- Supports chronos forecasts as auxiliary features
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuraldailytraining.data import DEFAULT_FEATURES, FeatureNormalizer
from neuraldailyv3timed.data import DailyDataModuleV3, MultiSymbolDatasetV3
from neuraldailyv5.config import DailyDatasetConfigV5


class MultiAssetDatasetV5(Dataset):
    """
    V5 Dataset that provides multi-asset data for portfolio training.

    Each sample includes:
    - features: (num_assets, seq_len, num_features) - features per asset
    - daily_returns: (num_assets, lookahead_days) - returns for simulation
    - asset_class: (num_assets,) - 0 for stocks, 1 for crypto
    - reference_close: (num_assets,) - last close price per asset
    - chronos_high/low: (num_assets,) - forecast bounds
    """

    def __init__(
        self,
        symbol_datasets: List,  # List of DailySymbolDataset
        sequence_length: int,
        lookahead_days: int = 20,
        symbols: Tuple[str, ...] = None,
    ):
        self.symbol_datasets = symbol_datasets
        self.sequence_length = sequence_length
        self.lookahead_days = lookahead_days
        self.total_length = sequence_length + lookahead_days
        self.symbols = symbols or tuple(ds.symbol for ds in symbol_datasets)
        self.num_assets = len(symbol_datasets)

        # Build index: find dates where all assets have data
        self._build_aligned_index()

    def _build_aligned_index(self):
        """Build index of dates where all assets have sufficient data."""
        # Get valid ranges for each asset
        asset_indices = []
        for ds in self.symbol_datasets:
            max_start = len(ds) - self.total_length
            if max_start > 0:
                asset_indices.append(set(range(max_start)))
            else:
                asset_indices.append(set())

        # Find common valid indices (simplified: use min length)
        if not asset_indices:
            self._indices = []
            return

        min_len = min(len(ds) for ds in self.symbol_datasets)
        max_samples = max(0, min_len - self.total_length)
        self._indices = list(range(max_samples))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = self._indices[idx]

        all_features = []
        all_returns = []
        all_asset_class = []
        all_ref_close = []
        all_chronos_high = []
        all_chronos_low = []

        for ds in self.symbol_datasets:
            # Get window
            history_end = start_idx + self.sequence_length
            lookahead_end = history_end + self.lookahead_days

            # Features for history
            features = ds.features[start_idx:history_end]

            # Prices for history and lookahead
            close_history = ds.close[start_idx:history_end]
            close_lookahead = ds.close[history_end:lookahead_end]

            # Compute daily returns for lookahead period
            # Return[t] = (close[t] - close[t-1]) / close[t-1]
            all_close = ds.close[history_end - 1:lookahead_end]  # Include last history day
            returns = (all_close[1:] - all_close[:-1]) / (all_close[:-1] + 1e-8)

            # Pad if needed
            if len(returns) < self.lookahead_days:
                pad_size = self.lookahead_days - len(returns)
                returns = torch.cat([
                    torch.tensor(returns, dtype=torch.float32),
                    torch.zeros(pad_size, dtype=torch.float32)
                ])
            else:
                returns = torch.tensor(returns[:self.lookahead_days], dtype=torch.float32)

            # Reference prices
            ref_close = ds.reference_close[history_end - 1]  # Last history day
            chronos_high = ds.chronos_high[history_end - 1]
            chronos_low = ds.chronos_low[history_end - 1]

            # Asset class
            asset_class = 1.0 if ds.asset_flag else 0.0

            all_features.append(torch.tensor(features, dtype=torch.float32))
            all_returns.append(returns)
            all_asset_class.append(asset_class)
            all_ref_close.append(ref_close)
            all_chronos_high.append(chronos_high)
            all_chronos_low.append(chronos_low)

        return {
            "features": torch.stack(all_features),  # (num_assets, seq_len, num_features)
            "daily_returns": torch.stack(all_returns),  # (num_assets, lookahead_days)
            "asset_class": torch.tensor(all_asset_class, dtype=torch.float32),  # (num_assets,)
            "reference_close": torch.tensor(all_ref_close, dtype=torch.float32),  # (num_assets,)
            "chronos_high": torch.tensor(all_chronos_high, dtype=torch.float32),  # (num_assets,)
            "chronos_low": torch.tensor(all_chronos_low, dtype=torch.float32),  # (num_assets,)
        }


class DailyDataModuleV5:
    """
    V5 Data module for portfolio training.

    Wraps V3/V4 infrastructure but provides multi-asset batches.
    """

    def __init__(self, config: DailyDatasetConfigV5):
        self.config = config
        self.symbols = config.symbols
        self.num_assets = len(config.symbols)

        # Convert to V3 config
        from neuraldailyv3timed.config import DailyDatasetConfigV3

        v3_config = DailyDatasetConfigV3(
            symbols=config.symbols,
            data_root=config.data_root,
            forecast_cache_dir=config.forecast_cache_dir,
            sequence_length=config.sequence_length,
            lookahead_days=config.lookahead_days,
            val_fraction=config.val_fraction,
            min_history_days=config.min_history_days,
            require_forecasts=config.require_forecasts,
            forecast_fill_strategy=config.forecast_fill_strategy,
            forecast_cache_writeback=config.forecast_cache_writeback,
            feature_columns=config.feature_columns,
            start_date=config.start_date,
            end_date=config.end_date,
            validation_days=config.validation_days,
            symbol_dropout_rate=config.symbol_dropout_rate,
            exclude_symbols=config.exclude_symbols,
            exclude_symbols_file=config.exclude_symbols_file,
            crypto_only=config.crypto_only,
            include_weekly_features=config.include_weekly_features,
            grouping_strategy=config.grouping_strategy,
            correlation_min_corr=config.correlation_min_corr,
            correlation_max_group_size=config.correlation_max_group_size,
            correlation_window_days=config.correlation_window_days,
            correlation_min_overlap=config.correlation_min_overlap,
            split_crypto_groups=config.split_crypto_groups,
        )

        self._v3_module = DailyDataModuleV3(v3_config)
        self._train_dataset: Optional[MultiAssetDatasetV5] = None
        self._val_dataset: Optional[MultiAssetDatasetV5] = None
        self._prepared = False

    @property
    def feature_columns(self) -> List[str]:
        return self._v3_module.feature_columns

    @property
    def normalizer(self) -> Optional[FeatureNormalizer]:
        return self._v3_module.normalizer

    def setup(self) -> None:
        """Prepare data module."""
        if not self._prepared:
            self._v3_module.prepare()
            self._prepared = True

            # Create V5 multi-asset datasets
            # Note: We need to align datasets by date
            train_symbol_datasets = self._v3_module._v1_module.train_dataset.datasets
            val_symbol_datasets = self._v3_module._v1_module.val_dataset.datasets

            self._train_dataset = MultiAssetDatasetV5(
                train_symbol_datasets,
                sequence_length=self.config.sequence_length,
                lookahead_days=self.config.lookahead_days,
                symbols=self.config.symbols,
            )
            self._val_dataset = MultiAssetDatasetV5(
                val_symbol_datasets,
                sequence_length=self.config.sequence_length,
                lookahead_days=self.config.lookahead_days,
                symbols=self.config.symbols,
            )

    def train_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        """Get training dataloader."""
        self.setup()
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    def val_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        """Get validation dataloader."""
        self.setup()
        return DataLoader(
            self._val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_asset_class_tensor(self, device: torch.device = None) -> torch.Tensor:
        """Get asset class tensor for the configured symbols."""
        asset_classes = []
        for symbol in self.symbols:
            is_crypto = symbol.upper().endswith("USD") or "-USD" in symbol.upper()
            asset_classes.append(1.0 if is_crypto else 0.0)
        tensor = torch.tensor(asset_classes, dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
