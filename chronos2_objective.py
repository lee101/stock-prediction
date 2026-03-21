from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChronosObjectiveMetrics:
    """Composite forecast metrics for Chronos2 tuning."""

    price_mae: float
    price_rmse: float
    pct_return_mae: float
    pct_return_mae_smoothness: float
    direction_accuracy: float
    objective: float
    n_samples: int


def _as_1d_array(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence.")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def compute_error_smoothness(abs_errors: Sequence[float] | np.ndarray) -> float:
    """Return a smoothness penalty for an absolute-error series."""

    errors = _as_1d_array(abs_errors, name="abs_errors")
    if errors.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(errors))))


def compute_chronos_objective(
    *,
    actual_close: Sequence[float] | np.ndarray,
    predicted_close: Sequence[float] | np.ndarray,
    reference_close: float,
    smoothness_weight: float = 0.0,
    direction_bonus: float = 0.0,
) -> ChronosObjectiveMetrics:
    """Compute MAE/smoothness/direction metrics and a single optimization objective."""

    actual = _as_1d_array(actual_close, name="actual_close")
    predicted = _as_1d_array(predicted_close, name="predicted_close")
    if actual.size != predicted.size:
        raise ValueError("actual_close and predicted_close must have identical length.")

    finite_mask = np.isfinite(actual) & np.isfinite(predicted)
    if not finite_mask.any():
        raise ValueError("actual_close/predicted_close contain no finite overlapping values.")

    actual = actual[finite_mask]
    predicted = predicted[finite_mask]

    abs_price_error = np.abs(predicted - actual)
    price_mae = float(np.mean(abs_price_error))
    price_rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))

    ref = float(reference_close)
    if not np.isfinite(ref) or abs(ref) < 1e-9:
        raise ValueError("reference_close must be finite and non-zero.")

    actual_returns = (actual - ref) / ref
    predicted_returns = (predicted - ref) / ref
    abs_return_error = np.abs(predicted_returns - actual_returns)
    pct_return_mae = float(np.mean(abs_return_error))
    pct_return_mae_smoothness = compute_error_smoothness(abs_return_error)

    actual_direction = np.sign(actual_returns)
    predicted_direction = np.sign(predicted_returns)
    direction_accuracy = float(np.mean(actual_direction == predicted_direction))

    objective = (
        pct_return_mae
        + float(smoothness_weight) * pct_return_mae_smoothness
        - float(direction_bonus) * direction_accuracy
    )

    return ChronosObjectiveMetrics(
        price_mae=price_mae,
        price_rmse=price_rmse,
        pct_return_mae=pct_return_mae,
        pct_return_mae_smoothness=pct_return_mae_smoothness,
        direction_accuracy=direction_accuracy,
        objective=float(objective),
        n_samples=int(actual.size),
    )


def build_correlation_matrix(
    close_history: Mapping[str, pd.Series],
    *,
    lookback: int | None = None,
    min_periods: int = 48,
) -> pd.DataFrame:
    """Build a symbol correlation matrix from close-price history."""

    aligned: dict[str, pd.Series] = {}
    for symbol, series in close_history.items():
        if not isinstance(series, pd.Series):
            continue
        s = pd.to_numeric(series, errors="coerce")
        s = s.dropna()
        if s.empty:
            continue
        aligned[str(symbol).upper()] = s.astype(float)

    if not aligned:
        return pd.DataFrame()

    # Use an outer join so symbols with shorter or staggered histories do not
    # erase the overlap for the rest of the universe. Pairwise correlation will
    # still honor `min_periods` on the overlapping timestamps for each pair.
    frame = pd.concat(aligned, axis=1, join="outer").sort_index()
    if frame.empty:
        return pd.DataFrame()
    if lookback is not None and lookback > 0 and len(frame) > lookback:
        frame = frame.iloc[-lookback:]

    returns = frame.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if returns.empty:
        return pd.DataFrame(index=frame.columns, columns=frame.columns, dtype=float)
    return returns.corr(min_periods=max(2, int(min_periods)))


def select_correlation_cohort(
    *,
    symbol: str,
    corr_matrix: pd.DataFrame,
    max_size: int,
    min_abs_corr: float = 0.25,
    include_negative: bool = False,
) -> tuple[str, ...]:
    """Pick up to ``max_size`` most correlated peers for a symbol."""

    symbol_key = str(symbol).upper()
    if max_size <= 0:
        return ()
    if corr_matrix.empty or symbol_key not in corr_matrix.index:
        return ()

    row = corr_matrix.loc[symbol_key].drop(labels=[symbol_key], errors="ignore")
    row = row.dropna()
    if not include_negative:
        row = row[row >= 0.0]
    row = row[row.abs() >= float(min_abs_corr)]
    if row.empty:
        return ()
    ranked = row.reindex(row.abs().sort_values(ascending=False).index)
    return tuple(str(col).upper() for col in ranked.index[: int(max_size)])


def build_correlation_cohort_map(
    corr_matrix: pd.DataFrame,
    *,
    max_size: int,
    min_abs_corr: float = 0.25,
    include_negative: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Build symbol -> correlated peers map from a correlation matrix."""

    if corr_matrix.empty:
        return {}
    cohorts: dict[str, tuple[str, ...]] = {}
    for symbol in corr_matrix.index:
        symbol_key = str(symbol).upper()
        cohorts[symbol_key] = select_correlation_cohort(
            symbol=symbol_key,
            corr_matrix=corr_matrix,
            max_size=max_size,
            min_abs_corr=min_abs_corr,
            include_negative=include_negative,
        )
    return cohorts


__all__ = [
    "ChronosObjectiveMetrics",
    "build_correlation_cohort_map",
    "build_correlation_matrix",
    "compute_chronos_objective",
    "compute_error_smoothness",
    "select_correlation_cohort",
]
