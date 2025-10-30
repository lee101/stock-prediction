from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array of samples, received shape {arr.shape!r}")
    return arr


@dataclass(frozen=True)
class ForecastReturnSet:
    """
    Represents a collection of Monte Carlo samples for the next rebalancing
    period's returns across the trading universe.

    The `samples` matrix has shape (num_paths, num_assets) with each entry
    expressing a simple (not log) return for the upcoming trading horizon.
    """

    universe: Tuple[str, ...]
    samples: np.ndarray

    def __post_init__(self) -> None:
        samples = _ensure_2d(self.samples)
        object.__setattr__(self, "samples", samples)
        if samples.shape[1] != len(self.universe):
            raise ValueError(
                f"Sample dimension mismatch: expected {len(self.universe)} columns, "
                f"received {samples.shape[1]}."
            )

    @property
    def sample_count(self) -> int:
        return int(self.samples.shape[0])

    def mean(self) -> np.ndarray:
        return np.mean(self.samples, axis=0)

    def covariance(self, *, ddof: int = 1) -> np.ndarray:
        if self.sample_count <= 1:
            raise ValueError("Cannot compute covariance with fewer than two samples.")
        return np.cov(self.samples, rowvar=False, ddof=ddof)


def shrink_covariance(matrix: np.ndarray, shrinkage: float = 0.0) -> np.ndarray:
    """
    Apply linear shrinkage towards a scaled identity target.

    Parameters
    ----------
    matrix:
        Positive semi-definite covariance matrix.
    shrinkage:
        Blend factor in [0, 1]. 0 leaves the matrix untouched; 1 replaces it
        with a scaled identity matrix that preserves the average variance.
    """
    cov = np.asarray(matrix, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    shrink = float(np.clip(shrinkage, 0.0, 1.0))
    if shrink == 0.0:
        return cov
    n = cov.shape[0]
    avg_var = float(np.trace(cov) / n) if n else 0.0
    target = np.eye(n, dtype=float) * avg_var
    return (1.0 - shrink) * cov + shrink * target


def ensure_common_universe(
    sets: Sequence[ForecastReturnSet],
) -> Tuple[Tuple[str, ...], Sequence[ForecastReturnSet]]:
    """
    Validate that all forecast sets share a consistent universe ordering.
    """
    if not sets:
        raise ValueError("At least one forecast return set is required.")
    reference = sets[0].universe
    for forecast in sets[1:]:
        if forecast.universe != reference:
            raise ValueError("All forecast sets must share the same universe ordering.")
    return reference, sets


def combine_forecast_sets(
    sets: Sequence[ForecastReturnSet],
    *,
    weights: Optional[Iterable[float]] = None,
    shrinkage: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse multiple forecast distributions into a single prior mean/covariance estimate.

    Combination is performed via law of total expectation / law of total variance,
    ensuring that the resulting covariance captures between-model dispersion in
    addition to each model's own uncertainty.
    """
    universe, sets = ensure_common_universe(sets)
    n = len(universe)

    if weights is None:
        raw_weights = np.ones(len(sets), dtype=float)
    else:
        raw_weights = np.asarray(list(weights), dtype=float)
        if raw_weights.shape != (len(sets),):
            raise ValueError("Weights must align with the number of forecast sets.")
    if np.any(raw_weights < 0):
        raise ValueError("Forecast weights must be non-negative.")
    if not np.any(raw_weights > 0):
        raise ValueError("At least one forecast weight must be positive.")

    weights_norm = raw_weights / raw_weights.sum()
    means = [forecast.mean() for forecast in sets]
    covs = [forecast.covariance() for forecast in sets]

    mu_prior = np.zeros(n, dtype=float)
    second_moment = np.zeros((n, n), dtype=float)

    for weight, mean_vec, cov_mat in zip(weights_norm, means, covs):
        mu_prior += weight * mean_vec
        second_moment += weight * (cov_mat + np.outer(mean_vec, mean_vec))

    cov_prior = second_moment - np.outer(mu_prior, mu_prior)
    cov_prior = (cov_prior + cov_prior.T) * 0.5  # ensure symmetry
    cov_prior = shrink_covariance(cov_prior, shrinkage=shrinkage)
    return mu_prior, cov_prior


def annualise_returns(mu: np.ndarray, *, periods_per_year: int = 252) -> np.ndarray:
    """Convert per-period simple returns into annualised equivalents."""
    mu = np.asarray(mu, dtype=float)
    return (1.0 + mu) ** periods_per_year - 1.0


def annualise_covariance(
    cov: np.ndarray,
    *,
    periods_per_year: int = 252,
) -> np.ndarray:
    """
    Convert per-period covariance into annualised covariance under the assumption
    of identical, independent increments.
    """
    cov = np.asarray(cov, dtype=float)
    return cov * periods_per_year

