import math

import pytest

from src.binan.margin_interest_rates import (
    BINANCE_MARGIN_INTEREST_RATES_PCT,
    HOURS_PER_YEAR,
    compute_compound_margin_interest,
    compute_simple_margin_interest,
    get_binance_margin_interest_rate_pct,
)


def test_get_rate_normalizes_asset_and_tier() -> None:
    rate = get_binance_margin_interest_rate_pct(" usdt ", tier="vip1")
    assert rate.hourly_pct == pytest.approx(0.000441)
    assert rate.yearly_pct == pytest.approx(3.86)

    rate2 = get_binance_margin_interest_rate_pct("USDT", tier="standard")
    assert rate2.hourly_pct == pytest.approx(0.00044504)
    assert rate2.yearly_pct == pytest.approx(3.90)


def test_get_rate_rejects_unknown_asset() -> None:
    with pytest.raises(KeyError):
        get_binance_margin_interest_rate_pct("NOTACOIN", tier="VIP1")


def test_get_rate_rejects_unknown_tier() -> None:
    with pytest.raises(ValueError):
        get_binance_margin_interest_rate_pct("USDT", tier="vip2")


def test_table_yearly_percent_matches_hourly_percent_within_rounding() -> None:
    # Binance UI rounds yearly %; accept a small rounding mismatch.
    for asset, tiers in BINANCE_MARGIN_INTEREST_RATES_PCT.items():
        for tier, rate in tiers.items():
            implied = rate.hourly_pct * HOURS_PER_YEAR
            assert implied == pytest.approx(rate.yearly_pct, abs=0.05), (asset, tier, implied, rate.yearly_pct)


def test_simple_interest_matches_expected_usdt_vip1() -> None:
    interest = compute_simple_margin_interest(1000.0, 24.0, borrowed_asset="USDT", tier="VIP1")
    # 1000 * (0.000441% / 100) * 24
    assert interest == pytest.approx(0.10584, rel=0, abs=1e-10)


def test_compound_interest_is_close_to_simple_for_small_rates_short_horizon() -> None:
    principal = 1_000_000.0
    hours = 24.0
    simple = compute_simple_margin_interest(principal, hours, borrowed_asset="USDT", tier="VIP1")
    compound = compute_compound_margin_interest(principal, hours, borrowed_asset="USDT", tier="VIP1")
    assert compound >= simple
    # Relative gap is ~ (hourly_rate * hours) / 2 for small rates (2nd order term).
    assert compound == pytest.approx(simple, rel=1e-4)


@pytest.mark.parametrize("principal,hours", [(0.0, 24.0), (100.0, 0.0)])
def test_interest_zero_cases(principal: float, hours: float) -> None:
    assert compute_simple_margin_interest(principal, hours, borrowed_asset="USDT") == 0.0
    assert compute_compound_margin_interest(principal, hours, borrowed_asset="USDT") == 0.0


@pytest.mark.parametrize("principal,hours", [(-1.0, 1.0), (1.0, -1.0), (math.nan, 1.0), (1.0, math.inf)])
def test_interest_rejects_non_finite_or_negative(principal: float, hours: float) -> None:
    with pytest.raises(ValueError):
        compute_simple_margin_interest(principal, hours, borrowed_asset="USDT")
    with pytest.raises(ValueError):
        compute_compound_margin_interest(principal, hours, borrowed_asset="USDT")

