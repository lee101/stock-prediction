"""
Sample aggregation utilities shared across Toto inference pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..dependency_injection import register_observer, resolve_numpy

np = resolve_numpy()


def _refresh_numpy(module):
    global np
    np = module


register_observer("numpy", _refresh_numpy)

_DEFAULT_METHODS = {
    "mean",
    "median",
    "p10",
    "p90",
}


def aggregate_with_spec(samples: Iterable[float] | np.ndarray, method: str) -> np.ndarray:
    """
    Aggregate Toto sample trajectories according to ``method``.

    Parameters
    ----------
    samples:
        Sample matrix shaped ``(num_samples, horizon)`` or anything that can be
        coerced into that layout.
    method:
        Aggregation spec string. Supported forms:

        * ``mean`` / ``median`` / ``p10`` / ``p90``
        * ``trimmed_mean_<fraction>`` (fraction in [0, 50], accepts percentages)
        * ``lower_trimmed_mean_<fraction>``
        * ``upper_trimmed_mean_<fraction>``
        * ``quantile_<fraction>``
        * ``mean_minus_std_<scale>``
        * ``mean_plus_std_<scale>``
        * ``mean_quantile_mix_<fraction>_<weight>`` (weight ∈ [0, 1])
        * ``quantile_plus_std_<fraction>_<scale>``

    Returns
    -------
    np.ndarray
        Aggregated horizon shaped ``(prediction_length,)``.
    """
    matrix = _ensure_matrix(samples)
    method = (method or "mean").strip().lower()

    if method in _DEFAULT_METHODS:
        if method == "mean":
            return matrix.mean(axis=0, dtype=np.float64)
        if method == "median":
            return np.median(matrix, axis=0)
        if method == "p10":
            return np.quantile(matrix, 0.10, axis=0)
        if method == "p90":
            return np.quantile(matrix, 0.90, axis=0)

    if method.startswith("trimmed_mean_"):
        fraction = _parse_fraction(method.split("_")[-1])
        return _trimmed_mean(matrix, fraction)

    if method.startswith("lower_trimmed_mean_"):
        fraction = _parse_fraction(method.split("_")[-1])
        sorted_matrix = np.sort(matrix, axis=0)
        total = sorted_matrix.shape[0]
        cutoff = max(1, int(total * (1.0 - fraction)))
        return sorted_matrix[:cutoff].mean(axis=0, dtype=np.float64)

    if method.startswith("upper_trimmed_mean_"):
        fraction = _parse_fraction(method.split("_")[-1])
        sorted_matrix = np.sort(matrix, axis=0)
        total = sorted_matrix.shape[0]
        start = min(total - 1, int(total * fraction))
        return sorted_matrix[start:].mean(axis=0, dtype=np.float64)

    if method.startswith("quantile_"):
        quantile = _parse_fraction(method.split("_")[-1])
        return np.quantile(matrix, quantile, axis=0)

    if method.startswith("mean_minus_std_"):
        factor = _parse_float(method.split("_")[-1], "mean_minus_std")
        mean = matrix.mean(axis=0, dtype=np.float64)
        std = matrix.std(axis=0, dtype=np.float64)
        return mean - factor * std

    if method.startswith("mean_plus_std_"):
        factor = _parse_float(method.split("_")[-1], "mean_plus_std")
        mean = matrix.mean(axis=0, dtype=np.float64)
        std = matrix.std(axis=0, dtype=np.float64)
        return mean + factor * std

    if method.startswith("mean_quantile_mix_"):
        parts = method.split("_")
        if len(parts) < 5:
            raise ValueError(f"Invalid mean_quantile_mix specifier: '{method}'")
        quantile = _parse_fraction(parts[-2])
        mean_weight = np.clip(_parse_float(parts[-1], "mean_quantile_mix"), 0.0, 1.0)
        mean_val = matrix.mean(axis=0, dtype=np.float64)
        quant_val = np.quantile(matrix, quantile, axis=0)
        return mean_weight * mean_val + (1.0 - mean_weight) * quant_val

    if method.startswith("quantile_plus_std_"):
        parts = method.split("_")
        if len(parts) < 5:
            raise ValueError(f"Invalid quantile_plus_std specifier: '{method}'")
        quantile = _parse_fraction(parts[-2])
        factor = _parse_float(parts[-1], "quantile_plus_std")
        return aggregate_quantile_plus_std(matrix, quantile, factor)

    raise ValueError(f"Unknown aggregation method '{method}'")


def aggregate_quantile_plus_std(
    samples: Iterable[float] | np.ndarray,
    quantile: float,
    std_scale: float,
) -> np.ndarray:
    """
    Aggregate samples by taking a quantile and adding a scaled standard deviation.
    """
    matrix = _ensure_matrix(samples)
    quantile = _validate_fraction(quantile, "quantile")
    std_scale = float(std_scale)
    quant_val = np.quantile(matrix, quantile, axis=0)
    std = matrix.std(axis=0, dtype=np.float64)
    return quant_val + std_scale * std


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ensure_matrix(samples: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("Samples must contain at least one element.")

    arr = np.squeeze(arr)

    if arr.ndim == 1:
        return arr.reshape(-1, 1)

    if arr.ndim == 2:
        # Ensure samples dimension is axis 0.
        if arr.shape[0] < arr.shape[1]:
            return arr.T.copy()
        return arr.copy()

    # Remove singleton dimensions and retry.
    squeeze_axes = [idx for idx, size in enumerate(arr.shape) if size == 1]
    if squeeze_axes:
        arr = np.squeeze(arr, axis=tuple(squeeze_axes))
        return _ensure_matrix(arr)

    raise ValueError(f"Unrecognised sample tensor shape: {arr.shape}")


def _trimmed_mean(matrix: np.ndarray, fraction: float) -> np.ndarray:
    fraction = _validate_fraction(fraction, "trimmed mean")
    if not 0.0 <= fraction < 0.5:
        raise ValueError("Trimmed mean fraction must lie in [0, 0.5).")

    sorted_matrix = np.sort(matrix, axis=0)
    total = sorted_matrix.shape[0]
    trim = int(total * fraction)

    if trim == 0 or trim * 2 >= total:
        return sorted_matrix.mean(axis=0, dtype=np.float64)

    return sorted_matrix[trim : total - trim].mean(axis=0, dtype=np.float64)


def _parse_fraction(token: str) -> float:
    return _validate_fraction(_parse_float(token, "fraction"), "fraction")


def _validate_fraction(value: float, name: str) -> float:
    if value > 1.0:
        value /= 100.0
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]; received {value}.")
    return float(value)


def _parse_float(token: str, context: str) -> float:
    try:
        return float(token)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid {context} parameter '{token}'.") from exc
