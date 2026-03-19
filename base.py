"""Base class for pre-augmentation strategies."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence
import numpy as np
import pandas as pd


class BaseAugmentation(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        context: pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        pass

    def get_config(self) -> Dict[str, Any]:
        return {"name": self.name(), "params": self.config, "metadata": self.metadata}
