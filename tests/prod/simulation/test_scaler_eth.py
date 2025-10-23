import numpy as np

from backtest_test3_inline import calibrate_signal


def test_eth_calibration_small_delta_stability():
    """Regression test: tiny normalized ETH deltas should not explode after calibration."""
    predictions = np.array([-0.010, 0.000, 0.003, 0.006, 0.0025], dtype=float)
    actual_returns = np.array([-0.008, 0.001, 0.002, 0.005, 0.0018], dtype=float)
    slope, intercept = calibrate_signal(predictions, actual_returns)
    raw_delta = 0.005098  # ~0.51% normalized signal from ETH incident
    calibrated_delta = slope * raw_delta + intercept
    assert abs(calibrated_delta) < 0.02, (
        "Calibrated ETH move deviated more than 2%, indicating scaler instability"
    )
