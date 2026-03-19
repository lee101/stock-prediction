"""
Sample aggregation utilities shared across Toto inference pipelines.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from numpy import ndarray as NDArray
else:  # pragma: no cover - typing fallback
    NDArray = Any


def _optional_import(module_name: str) -> ModuleType | None:
    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        return None


np: ModuleType | None = _optional_import("numpy")


def setup_toto_aggregation_imports(
    *,
    numpy_module: ModuleType | None = None,
    **_: Any,
) -> None:
    global np
    if numpy_module is not None:
        np = numpy_module


def _require_numpy() -> ModuleType:
    global np
    if np is not None:
        return np
    try:
        module = import_module("numpy")
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is unavailable. Call setup_toto_aggregation_imports before use.") from exc
    np = module
    return module


_DEFAULT_METHODS = {
    "mean",
    "median",
    "p10",
    "p90",
}


def aggregate_with_spec(samples: Iterable[float] | NDArray, method: str) -> NDArray:
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
    numpy_mod = _require_numpy()
    matrix = _ensure_matrix(samples)
    method = (method or "mean").strip().lower()

    if method in _DEFAULT_METHODS:
        if method == "mean":
            return matrix.mean(axis=0, dtype=numpy_mod.float64)
        if method == "median":
            return numpy_mod.median(matrix, axis=0)
        if method == "p10":
            return numpy_mod.quantile(matrix, 0.10, axis=0)
        if method == "p90":
            return numpy_mod.quantile(matrix, 0.90, axis=0)

    if method.startswith("trimmed_mean_"):
        fraction = _parse_fraction(method.split("_")[-1])
        return _trimmed_mean(matrix, fraction)

    if method.startswith("lower_trimmed_mean_"):
        fraction = _parse_fraction(method.split("_")[-1])
        sorted_matrix = numpy_mod.sort(matrix, axis=0)
        total = sorted_matrix.shape[0]
        cutoff = max(1, int(total * (1.0 - fraction)))
        return sorted_matrix[:cutoff].mean(axis=0, dtype=numpy_mod.float64)

    if method.startswith("upper_trimmed_mean_"):
        fraction = _parse_fraction(method.split("_")[-1])
        sorted_matrix = numpy_mod.sort(matrix, axis=0)
        total = sorted_matrix.shape[0]
        start = min(total - 1, int(total * fraction))
        return sorted_matrix[start:].mean(axis=0, dtype=numpy_mod.float64)

    if method.startswith("quantile_"):
        quantile = _parse_fraction(method.split("_")[-1])
        return numpy_mod.quantile(matrix, quantile, axis=0)

    if method.startswith("mean_minus_std_"):
        factor = _parse_float(method.split("_")[-1], "mean_minus_std")
        mean = matrix.mean(axis=0, dtype=numpy_mod.float64)
        std = matrix.std(axis=0, dtype=numpy_mod.float64)
        return mean - factor * std

    if method.startswith("mean_plus_std_"):
        factor = _parse_float(method.split("_")[-1], "mean_plus_std")
        mean = matrix.mean(axis=0, dtype=numpy_mod.float64)
        std = matrix.std(axis=0, dtype=numpy_mod.float64)
        return mean + factor * std

    if method.startswith("mean_quantile_mix_"):
        parts = method.split("_")
        if len(parts) < 5:
            raise ValueError(f"Invalid mean_quantile_mix specifier: '{method}'")
        quantile = _parse_fraction(parts[-2])
        mean_weight = numpy_mod.clip(_parse_float(parts[-1], "mean_quantile_mix"), 0.0, 1.0)
        mean_val = matrix.mean(axis=0, dtype=numpy_mod.float64)
        quant_val = numpy_mod.quantile(matrix, quantile, axis=0)
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
    samples: Iterable[float] | NDArray,
    quantile: float,
    std_scale: float,
) -> NDArray:
    """
    Aggregate samples by taking a quantile and adding a scaled standard deviation.
    """
    numpy_mod = _require_numpy()
    matrix = _ensure_matrix(samples)
    quantile = _validate_fraction(quantile, "quantile")
    std_scale = float(std_scale)
    quant_val = numpy_mod.quantile(matrix, quantile, axis=0)
    std = matrix.std(axis=0, dtype=numpy_mod.float64)
    return quant_val + std_scale * std


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ensure_matrix(samples: Iterable[float] | NDArray) -> NDArray:
    numpy_mod = _require_numpy()
    arr = numpy_mod.asarray(samples, dtype=numpy_mod.float64)
    if arr.ndim == 0:
        raise ValueError("Samples must contain at least one element.")

    arr = numpy_mod.squeeze(arr)

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
        arr = numpy_mod.squeeze(arr, axis=tuple(squeeze_axes))
        return _ensure_matrix(arr)

    raise ValueError(f"Unrecognised sample tensor shape: {arr.shape}")


def _trimmed_mean(matrix: NDArray, fraction: float) -> NDArray:
    numpy_mod = _require_numpy()
    fraction = _validate_fraction(fraction, "trimmed mean")
    if not 0.0 <= fraction < 0.5:
        raise ValueError("Trimmed mean fraction must lie in [0, 0.5).")

    sorted_matrix = numpy_mod.sort(matrix, axis=0)
    total = sorted_matrix.shape[0]
    trim = int(total * fraction)

    if trim == 0 or trim * 2 >= total:
        return sorted_matrix.mean(axis=0, dtype=numpy_mod.float64)

    return sorted_matrix[trim : total - trim].mean(axis=0, dtype=numpy_mod.float64)


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
