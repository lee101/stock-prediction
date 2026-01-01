"""V4 Data module - reuses V3 data loading with lookahead.

Uses the V3 data infrastructure which already supports lookahead windows.
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
from neuraldailyv4.config import DailyDatasetConfigV4


class DailyDataModuleV4:
    """
    V4 Data module wrapping V3's lookahead-aware data infrastructure.

    Simply converts V4 config to V3 config and reuses V3's data loading.
    """

    def __init__(self, config: DailyDatasetConfigV4):
        self.config = config

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

    def train_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        """Get training dataloader."""
        self.setup()
        return self._v3_module.get_train_loader(batch_size, shuffle=True, num_workers=num_workers)

    def val_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        """Get validation dataloader."""
        self.setup()
        return self._v3_module.get_val_loader(batch_size, num_workers=num_workers)
