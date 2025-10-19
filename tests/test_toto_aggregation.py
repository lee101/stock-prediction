from __future__ import annotations

import numpy as np
import pytest

from src.models.toto_aggregation import (
    aggregate_quantile_plus_std,
    aggregate_with_spec,
)


@pytest.fixture
def sample_matrix() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.normal(loc=100.0, scale=2.5, size=(128, 3))


def test_aggregate_with_spec_mean(sample_matrix: np.ndarray) -> None:
    expected = sample_matrix.mean(axis=0, dtype=np.float64)
    result = aggregate_with_spec(sample_matrix, "mean")
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_aggregate_with_spec_trimmed_mean(sample_matrix: np.ndarray) -> None:
    trimmed = aggregate_with_spec(sample_matrix, "trimmed_mean_10")
    manual = aggregate_with_spec(sample_matrix, "trimmed_mean_0.1")  # check parsing paths align
    np.testing.assert_allclose(trimmed, manual, rtol=1e-12, atol=1e-12)


def test_aggregate_quantile_plus_std(sample_matrix: np.ndarray) -> None:
    quant = aggregate_with_spec(sample_matrix, "quantile_0.25")
    result = aggregate_quantile_plus_std(sample_matrix, 0.25, 0.5)
    manual = quant + 0.5 * sample_matrix.std(axis=0, dtype=np.float64)
    np.testing.assert_allclose(result, manual, rtol=1e-12, atol=1e-12)


def test_aggregate_with_spec_invalid(sample_matrix: np.ndarray) -> None:
    with pytest.raises(ValueError):
        aggregate_with_spec(sample_matrix, "unknown_method")

