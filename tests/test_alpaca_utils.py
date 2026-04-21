"""Unit tests for ``src/alpaca_utils`` — leverage + financing helpers.

These utilities are production-critical: they translate the brokerage's
6.5%/yr financing charge into a per-step penalty and clamp end-of-day
exposure so margin calls never fire. Keep them bit-for-bit deterministic.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.alpaca_utils import (
    ANNUAL_MARGIN_RATE,
    BASE_GROSS_EXPOSURE,
    MAX_GROSS_EXPOSURE,
    TRADING_DAYS_PER_YEAR,
    annual_to_daily_rate,
    clamp_end_of_day_weights,
    leverage_penalty,
)


def test_annual_to_daily_rate_default() -> None:
    r = annual_to_daily_rate(0.0504)
    assert r == pytest.approx(0.0504 / TRADING_DAYS_PER_YEAR)


def test_annual_to_daily_rate_custom_trading_days() -> None:
    assert annual_to_daily_rate(0.1, trading_days=100) == pytest.approx(0.001)


def test_annual_to_daily_rate_zero_or_negative_days_clamped_to_one() -> None:
    # trading_days clamp guards against div-by-zero
    assert annual_to_daily_rate(0.1, trading_days=0) == pytest.approx(0.1)
    assert annual_to_daily_rate(0.1, trading_days=-5) == pytest.approx(0.1)


def test_leverage_penalty_no_excess_is_zero() -> None:
    # gross at or below baseline: no financing charge
    assert leverage_penalty(1.0) == 0.0
    assert leverage_penalty(0.5) == 0.0


def test_leverage_penalty_uses_default_annual_rate() -> None:
    expected_daily = ANNUAL_MARGIN_RATE / TRADING_DAYS_PER_YEAR
    # 1.5× gross on 1.0 baseline → 0.5× excess
    assert leverage_penalty(1.5) == pytest.approx(0.5 * expected_daily)


def test_leverage_penalty_accepts_explicit_daily_rate() -> None:
    assert leverage_penalty(2.0, daily_rate=0.0001) == pytest.approx(0.0001)
    assert leverage_penalty(3.0, daily_rate=0.0002) == pytest.approx(0.0004)


def test_leverage_penalty_respects_custom_base() -> None:
    assert leverage_penalty(3.0, base_exposure=2.0, daily_rate=0.0001) == pytest.approx(0.0001)


def test_clamp_end_of_day_weights_under_cap_is_noop() -> None:
    w = np.array([0.4, -0.3, 0.2], dtype=np.float32)
    clamped, turnover = clamp_end_of_day_weights(w)
    np.testing.assert_allclose(clamped, w)
    assert turnover == 0.0
    # must not alias the input
    assert clamped is not w


def test_clamp_end_of_day_weights_above_cap_scaled_down() -> None:
    # gross = 3.0, cap = 2.0 → scale = 2/3
    w = np.array([1.5, -1.0, 0.5], dtype=np.float32)
    clamped, turnover = clamp_end_of_day_weights(w, max_gross=2.0)
    assert float(np.sum(np.abs(clamped))) == pytest.approx(2.0, abs=1e-5)
    # turnover is sum |w - scaled|  = gross - max_gross (sign-preserving scale)
    assert turnover == pytest.approx(1.0, abs=1e-5)


def test_clamp_end_of_day_weights_cap_cannot_go_below_one() -> None:
    # max_gross=0.5 is clamped to 1.0 internally
    w = np.array([0.6, 0.3], dtype=np.float32)  # gross 0.9 ≤ 1.0
    clamped, turnover = clamp_end_of_day_weights(w, max_gross=0.5)
    np.testing.assert_allclose(clamped, w)
    assert turnover == 0.0


def test_clamp_end_of_day_weights_preserves_signs() -> None:
    w = np.array([2.0, -2.0, 1.0], dtype=np.float32)
    clamped, _ = clamp_end_of_day_weights(w, max_gross=MAX_GROSS_EXPOSURE)
    # signs match
    assert np.sign(clamped).tolist() == np.sign(w).tolist()
    assert float(np.sum(np.abs(clamped))) == pytest.approx(MAX_GROSS_EXPOSURE, abs=1e-5)


def test_clamp_end_of_day_weights_empty_array() -> None:
    w = np.array([], dtype=np.float32)
    clamped, turnover = clamp_end_of_day_weights(w)
    assert clamped.size == 0
    assert turnover == 0.0


def test_leverage_penalty_base_default_is_one() -> None:
    # confirm BASE_GROSS_EXPOSURE exported constant matches behaviour
    assert BASE_GROSS_EXPOSURE == 1.0
    # no excess at gross = base
    assert leverage_penalty(BASE_GROSS_EXPOSURE) == 0.0
