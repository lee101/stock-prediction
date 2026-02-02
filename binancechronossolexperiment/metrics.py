from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def annualized_return(
    equity_curve: Iterable[float] | pd.Series,
    *,
    periods_per_year: int = 24 * 365,
) -> float:
    """Compute annualized return from an equity curve."""

    series = pd.Series(equity_curve, dtype=float)
    if series.size < 2:
        return 0.0
    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0.0:
        return 0.0
    periods = max(1, series.size - 1)
    return float((end / start) ** (periods_per_year / periods) - 1.0)


__all__ = ["annualized_return"]
