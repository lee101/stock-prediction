"""Data processing utilities for backtesting."""

from typing import Optional, Union

import numpy as np
import pandas as pd


def mean_if_exists(df: pd.DataFrame, column: Optional[str]) -> Optional[float]:
    """Calculate mean of a column if it exists and has valid data.

    Args:
        df: DataFrame to query
        column: Column name to calculate mean for

    Returns:
        Mean value or None if column doesn't exist or has no valid data
    """
    if not column or column not in df.columns:
        return None
    series = df[column]
    if series.empty:
        return None
    value = float(series.mean())
    if np.isnan(value):
        return None
    return value


def to_numpy_array(values: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Convert various return series formats to numpy array.

    Args:
        values: Returns as numpy array or pandas Series

    Returns:
        Returns as numpy array
    """
    if isinstance(values, pd.Series):
        array = values.to_numpy(dtype=float)
    else:
        array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    return array


def normalize_series(series: pd.Series, coerce_numeric_fn) -> pd.Series:
    """Normalize a pandas Series by coercing all values to numeric.

    Args:
        series: Series to normalize
        coerce_numeric_fn: Function to coerce values to numeric (signature: value, default, prefer)

    Returns:
        Normalized series with numeric values
    """
    return series.apply(lambda value: coerce_numeric_fn(value, default=0.0, prefer="mean"))
