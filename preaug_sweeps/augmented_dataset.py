"""
Augmented dataset wrapper that applies pre-augmentation before training.

This wraps the existing KronosMultiTickerDataset to apply transformations.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch

from kronostraining.data_utils import (
    ALL_FEATURES,
    TIME_FEATURES,
    list_symbol_files,
    load_symbol_dataframe,
)

# Handle both direct execution and module import
try:
    from .augmentations import BaseAugmentation
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from augmentations import BaseAugmentation


class AugmentedDatasetBuilder:
    """
    Builds training data with pre-augmentation applied.

    This modifies the CSV data on disk in a temporary directory
    so that existing training pipelines can use augmented data.
    """

    def __init__(
        self,
        source_dir: str,
        augmentation: BaseAugmentation,
        target_symbols: Optional[List[str]] = None
    ):
        """
        Args:
            source_dir: Original training data directory
            augmentation: Pre-augmentation strategy to apply
            target_symbols: Optional list of symbols to process (None = all)
        """
        self.source_dir = Path(source_dir)
        self.augmentation = augmentation
        self.target_symbols = target_symbols
        self.augmented_dir: Optional[Path] = None

    def create_augmented_dataset(self, output_dir: str) -> Path:
        """
        Create augmented dataset in output_dir.

        Returns:
            Path to augmented dataset directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Process each symbol
        for symbol, csv_path in list_symbol_files(self.source_dir):
            if self.target_symbols and symbol not in self.target_symbols:
                continue

            # Load original dataframe
            df = load_symbol_dataframe(csv_path)

            # Apply augmentation to features only (not timestamps or time features)
            df_aug = self._augment_symbol_data(df)

            # Save augmented data
            output_file = output_path / f"{symbol}.csv"
            self._save_augmented_csv(df_aug, output_file)

        self.augmented_dir = output_path
        return output_path

    def _augment_symbol_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply augmentation to a single symbol's dataframe."""
        # Separate timestamps and time features (these don't get augmented)
        timestamps = df["timestamps"].copy()
        time_cols = {col: df[col].copy() for col in TIME_FEATURES if col in df.columns}

        # Apply augmentation to the feature columns
        df_features = df[list(ALL_FEATURES)].copy()
        df_aug_features = self.augmentation.transform_dataframe(df_features)

        # Reconstruct full dataframe
        df_aug = pd.DataFrame()
        df_aug["timestamps"] = timestamps

        # Add augmented features
        for col in ALL_FEATURES:
            if col in df_aug_features.columns:
                df_aug[col] = df_aug_features[col]

        # Add time features (unchanged)
        for col, values in time_cols.items():
            df_aug[col] = values

        return df_aug

    def _save_augmented_csv(self, df: pd.DataFrame, output_file: Path) -> None:
        """Save augmented dataframe to CSV."""
        # Keep same format as original
        df.to_csv(output_file, index=False)

    def cleanup(self) -> None:
        """Remove temporary augmented dataset."""
        if self.augmented_dir and self.augmented_dir.exists():
            import shutil
            shutil.rmtree(self.augmented_dir)


class AugmentationAwarePredictor:
    """
    Wrapper for making predictions with augmented models.

    Handles inverse transformation of predictions back to original scale.
    """

    def __init__(
        self,
        base_predictor,
        augmentation: BaseAugmentation,
        original_data_dir: str
    ):
        """
        Args:
            base_predictor: KronosPredictor or similar
            augmentation: The augmentation used during training
            original_data_dir: Path to original (non-augmented) data
        """
        self.base_predictor = base_predictor
        self.augmentation = augmentation
        self.original_data_dir = Path(original_data_dir)

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str,
        **predict_kwargs
    ) -> pd.DataFrame:
        """
        Make predictions and inverse transform to original scale.

        Args:
            df: Context dataframe (in original scale)
            symbol: Symbol name
            **predict_kwargs: Additional args for base_predictor.predict()

        Returns:
            Predictions in original scale
        """
        # Transform context to augmented scale
        df_features = df[list(ALL_FEATURES)].copy()
        df_aug = self.augmentation.transform_dataframe(df_features)

        # Get predictions in augmented space
        # Note: The exact API depends on your predictor
        # This is a template - adjust as needed
        pred_aug = self.base_predictor.predict(df_aug, **predict_kwargs)

        # Inverse transform predictions to original scale
        if isinstance(pred_aug, pd.DataFrame):
            pred_array = pred_aug[list(ALL_FEATURES)].values
        else:
            pred_array = pred_aug

        pred_original = self.augmentation.inverse_transform_predictions(
            pred_array,
            context=df_features,
            columns=df_features.columns,
        )

        # Convert back to dataframe format
        if isinstance(pred_aug, pd.DataFrame):
            pred_df = pred_aug.copy()
            for i, col in enumerate(ALL_FEATURES):
                if i < pred_original.shape[1]:
                    pred_df[col] = pred_original[:, i]
            return pred_df
        else:
            return pred_original
