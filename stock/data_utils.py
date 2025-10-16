from __future__ import annotations

import math
import numbers
from decimal import Decimal
from typing import Any, Literal, Optional

import numpy as np

try:  # Pandas is optional at runtime for certain unit tests.
    import pandas as pd

    _HAS_PANDAS = True
except Exception:  # pragma: no cover - pandas missing in minimal envs.
    pd = None  # type: ignore[assignment]
    _HAS_PANDAS = False

PreferStrategy = Literal["first", "last", "mean"]


def _nan_guard(value: float, default: float) -> float:
    if math.isnan(value):
        return float(default)
    return value


def _extract_from_ndarray(array: np.ndarray, prefer: PreferStrategy) -> Optional[float]:
    if array.size == 0:
        return None
    try:
        flattened = np.asarray(array, dtype="float64").reshape(-1)
    except (TypeError, ValueError):
        return None
    if prefer == "mean":
        with np.errstate(all="ignore"):
            candidate = float(np.nanmean(flattened))
        if math.isnan(candidate):
            return None
        return candidate

    iterator = flattened if prefer == "first" else flattened[::-1]
    for candidate in iterator:
        if not math.isnan(candidate):
            return float(candidate)
    return None


def _extract_from_series(series: "pd.Series[Any]", prefer: PreferStrategy) -> Optional[float]:
    if series.empty:
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    if prefer == "mean":
        try:
            return float(valid.astype("float64").mean())
        except (TypeError, ValueError):
            return None
    index = 0 if prefer == "first" else -1
    try:
        return float(valid.astype("float64").iloc[index])
    except (TypeError, ValueError):
        return None


def _extract_from_dataframe(frame: "pd.DataFrame", prefer: PreferStrategy) -> Optional[float]:
    if frame.empty:
        return None
    numeric = frame.select_dtypes(include=["number"])
    if numeric.empty:
        return None
    return _extract_from_ndarray(numeric.to_numpy(), prefer)


def coerce_numeric(
    value: Any,
    default: float = 0.0,
    *,
    prefer: PreferStrategy = "last",
) -> float:
    """Coerce scalars, numpy arrays, or pandas objects to a finite float.

    Parameters
    ----------
    value:
        Input value that may be numeric, numpy-based, or pandas-based.
    default:
        Fallback when the input cannot be coerced or resolves to NaN.
    prefer:
        Strategy used when the input contains multiple values. Options:
        - ``"last"`` (default): take the last finite observation.
        - ``"first"``: take the first finite observation.
        - ``"mean"``: compute the mean of all numeric values.
    """

    if value is None:
        return float(default)

    if isinstance(value, bool):
        return float(int(value))

    if isinstance(value, numbers.Real):
        return _nan_guard(float(value), default)

    if isinstance(value, Decimal):
        return _nan_guard(float(value), default)

    if isinstance(value, np.ndarray):
        candidate = _extract_from_ndarray(value, prefer)
        if candidate is None:
            return float(default)
        return candidate

    if _HAS_PANDAS:
        if isinstance(value, pd.Series):
            candidate = _extract_from_series(value, prefer)
            if candidate is None:
                return float(default)
            return candidate
        if isinstance(value, pd.Index):
            candidate = _extract_from_series(value.to_series(index=False), prefer)
            if candidate is None:
                return float(default)
            return candidate
        if isinstance(value, pd.DataFrame):
            candidate = _extract_from_dataframe(value, prefer)
            if candidate is None:
                return float(default)
            return candidate

    if hasattr(value, "item"):
        try:
            return coerce_numeric(value.item(), default=default, prefer=prefer)
        except (TypeError, ValueError):
            pass

    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    return _nan_guard(coerced, default)


def ensure_lower_bound(
    value: Any,
    lower_bound: float,
    *,
    default: float = 0.0,
    prefer: PreferStrategy = "last",
) -> float:
    """Clamp ``value`` to ``lower_bound`` with robust numeric coercion."""

    candidate = coerce_numeric(value, default=default, prefer=prefer)
    minimum = coerce_numeric(lower_bound, default=lower_bound, prefer=prefer)
    if math.isnan(minimum):
        raise ValueError("lower_bound resolves to NaN")
    if candidate < minimum:
        return minimum
    return candidate


def ensure_range(
    value: Any,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    default: float = 0.0,
    prefer: PreferStrategy = "last",
) -> float:
    """Clamp ``value`` within ``[minimum, maximum]`` while handling non-scalars."""

    candidate = coerce_numeric(value, default=default, prefer=prefer)
    if minimum is not None:
        min_value = coerce_numeric(minimum, default=minimum, prefer=prefer)
        if math.isnan(min_value):
            raise ValueError("minimum resolves to NaN")
        if candidate < min_value:
            candidate = min_value
    if maximum is not None:
        max_value = coerce_numeric(maximum, default=maximum, prefer=prefer)
        if math.isnan(max_value):
            raise ValueError("maximum resolves to NaN")
        if candidate > max_value:
            candidate = max_value
    return candidate


def safe_divide(
    numerator: Any,
    denominator: Any,
    *,
    default: float = 0.0,
    prefer: PreferStrategy = "last",
    epsilon: float = 1e-12,
) -> float:
    """Robust divide helper that avoids propagating NaNs or ZeroDivision."""

    denom = coerce_numeric(denominator, default=0.0, prefer=prefer)
    if math.isnan(denom) or abs(denom) <= epsilon:
        return float(default)
    numer = coerce_numeric(numerator, default=default, prefer=prefer)
    return numer / denom
