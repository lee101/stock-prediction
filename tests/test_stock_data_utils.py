import math

import numpy as np
import pandas as pd
import pytest

from stock.data_utils import coerce_numeric, ensure_lower_bound, ensure_range, safe_divide


def test_coerce_numeric_prefers_last_valid_numpy():
    data = np.array([np.nan, 1.25, 2.75])
    result = coerce_numeric(data)
    assert math.isclose(result, 2.75)


def test_coerce_numeric_series_drops_nan():
    series = pd.Series([np.nan, np.nan, 4.5])
    assert math.isclose(coerce_numeric(series), 4.5)


def test_coerce_numeric_series_mean_strategy():
    series = pd.Series([1.0, 3.0, 5.0])
    result = coerce_numeric(series, prefer="mean")
    assert math.isclose(result, 3.0)


@pytest.mark.parametrize(
    "value,lower,expected",
    [
        (-1.0, 0.0, 0.0),
        (5.0, 0.0, 5.0),
        (np.float64(-2.5), 1.5, 1.5),
    ],
)
def test_ensure_lower_bound_clamps(value, lower, expected):
    assert math.isclose(ensure_lower_bound(value, lower), expected)


def test_ensure_range_handles_bounds():
    assert ensure_range(-5, minimum=-2, maximum=2) == -2
    assert ensure_range(5, minimum=-2, maximum=2) == 2
    assert ensure_range(1, minimum=-2, maximum=2) == 1


@pytest.mark.parametrize(
    "numerator,denominator,expected",
    [
        (10.0, 2.0, 5.0),
        (10.0, 0.0, 0.0),
        (np.array([1.0, 2.0]), np.array([0.0, 0.0]), 0.0),
    ],
)
def test_safe_divide_handles_zero(numerator, denominator, expected):
    assert math.isclose(safe_divide(numerator, denominator), expected)
