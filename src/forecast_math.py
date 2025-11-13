"""Utility math helpers for converting forecast formats."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, Sequence[float]]


def absolute_prices_to_pct_returns(abs_predictions: ArrayLike, last_price: float) -> torch.Tensor:
    """Convert absolute price forecasts into percentage returns relative to the prior close."""

    pct_changes = []
    prev_price = float(last_price)
    values = abs_predictions.tolist() if isinstance(abs_predictions, np.ndarray) else list(abs_predictions)
    for future_price in values:
        price = float(future_price)
        pct_change = 0.0 if prev_price == 0.0 else (price - prev_price) / prev_price
        pct_changes.append(pct_change)
        prev_price = price
    return torch.tensor(pct_changes, dtype=torch.float32)


__all__ = ["absolute_prices_to_pct_returns"]
