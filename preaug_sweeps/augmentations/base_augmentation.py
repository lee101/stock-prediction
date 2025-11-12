"""
Base class for pre-augmentation strategies.

Pre-augmentation applies transformations to the training data BEFORE model training,
with the goal of improving MAE by presenting the data in a more learnable form.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


class BaseAugmentation(ABC):
    """Base class for all pre-augmentation strategies."""

    def __init__(self, **kwargs):
        """Initialize augmentation with configuration."""
        self.config = kwargs
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def name(self) -> str:
        """Return the name of this augmentation strategy."""
        pass

    @abstractmethod
    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a dataframe before training.

        Args:
            df: DataFrame with columns [timestamps, open, high, low, close, volume, amount, ...]

        Returns:
            Transformed DataFrame with same structure
        """
        pass

    @abstractmethod
    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        context: pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Reverse the transformation on model predictions.

        Args:
            predictions: Model predictions (N, num_features)
            context: Original context dataframe used for prediction
            columns: Optional explicit column order for ``predictions``. When
                omitted, implementations may infer it from ``context``.

        Returns:
            Predictions in original scale
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for this augmentation."""
        return {
            "name": self.name(),
            "params": self.config,
            "metadata": self.metadata
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config})"
