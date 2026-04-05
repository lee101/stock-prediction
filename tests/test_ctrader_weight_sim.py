from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from ctrader.market_sim_ffi import load_library, simulate_target_weights


REPO = Path("/home/lee/code/stock")
CTRADER_DIR = REPO / "ctrader"


def test_weight_sim_ffi_buy_and_hold():
    subprocess.run(
        ["make", "libmarket_sim.so"],
        cwd=CTRADER_DIR,
        check=True,
        capture_output=True,
        text=True,
    )

    lib = load_library(CTRADER_DIR / "libmarket_sim.so")
    close = np.array([[100.0], [110.0], [121.0]], dtype=np.float64)
    weights = np.array([[1.0], [1.0], [0.0]], dtype=np.float64)

    result, equity_curve = simulate_target_weights(
        close,
        weights,
        {
            "initial_cash": 10_000.0,
            "max_gross_leverage": 1.0,
            "fee_rate": 0.0,
            "borrow_rate_per_period": 0.0,
            "periods_per_year": 2.0,
            "can_short": 0,
        },
        library=lib,
    )

    assert result.total_return == pytest.approx(0.21)
    assert result.final_equity == pytest.approx(12100.0)
    assert result.annualized_return == pytest.approx(0.21)
    assert result.max_drawdown == pytest.approx(0.0)
    assert np.allclose(equity_curve, np.array([10000.0, 11000.0, 12100.0]))
