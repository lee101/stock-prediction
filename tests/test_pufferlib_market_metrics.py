from __future__ import annotations

import pytest

from pufferlib_market.metrics import annualize_total_return, stock_market_hours_per_year


def test_annualize_total_return_matches_closed_form() -> None:
    annualized = annualize_total_return(0.21, periods=2, periods_per_year=252.0)
    expected = (1.21 ** (252.0 / 2.0)) - 1.0
    assert annualized == pytest.approx(expected)


def test_annualize_total_return_handles_bankrupt_path() -> None:
    assert annualize_total_return(-1.0, periods=10, periods_per_year=252.0) == -1.0


def test_stock_market_hours_per_year_uses_trading_day_hours() -> None:
    assert stock_market_hours_per_year() == pytest.approx(1638.0)
