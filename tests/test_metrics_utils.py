import numpy as np
import pytest

from kronostraining.metrics_utils import compute_mae_percent


def test_compute_mae_percent_basic():
    actual = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    mae = 2.0
    assert compute_mae_percent(mae, actual) == pytest.approx(10.0)


def test_compute_mae_percent_handles_negative_actuals():
    actual = np.array([-5.0, 5.0, -15.0, 15.0], dtype=np.float64)
    mae = 1.5
    mean_abs = np.mean(np.abs(actual))
    expected = (mae / mean_abs) * 100.0
    assert compute_mae_percent(mae, actual) == pytest.approx(expected)


def test_compute_mae_percent_zero_scale_returns_inf():
    actual = np.zeros(4, dtype=np.float64)
    mae = 0.25
    result = compute_mae_percent(mae, actual)
    assert np.isinf(result) and result > 0


def test_compute_mae_percent_zero_mae_zero_scale():
    actual = np.zeros(3, dtype=np.float64)
    mae = 0.0
    assert compute_mae_percent(mae, actual) == 0.0


def test_compute_mae_percent_validates_inputs():
    actual = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        compute_mae_percent(-0.1, actual)

    with pytest.raises(ValueError):
        compute_mae_percent(0.1, np.array([]))
