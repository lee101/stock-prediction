from __future__ import annotations

import pytest

from src.allocation_utils import allocation_base_usd, allocation_usd_for_symbol


class _Account:
    def __init__(self, *, cash: float, buying_power: float, equity: float):
        self.cash = cash
        self.buying_power = buying_power
        self.equity = equity


def test_allocation_base_usd_crypto_prefers_cash():
    acct = _Account(cash=123.0, buying_power=999.0, equity=555.0)
    assert allocation_base_usd(acct, symbol="BTCUSD", prefer_cash_for_crypto=True) == 123.0


def test_allocation_base_usd_crypto_falls_back_to_equity_when_cash_missing():
    acct = _Account(cash=0.0, buying_power=999.0, equity=555.0)
    assert allocation_base_usd(acct, symbol="BTCUSD", prefer_cash_for_crypto=True) == 555.0


def test_allocation_base_usd_stock_prefers_buying_power_then_equity():
    acct = _Account(cash=123.0, buying_power=999.0, equity=555.0)
    assert allocation_base_usd(acct, symbol="NVDA", prefer_cash_for_crypto=True) == 999.0
    acct2 = _Account(cash=123.0, buying_power=0.0, equity=555.0)
    assert allocation_base_usd(acct2, symbol="NVDA", prefer_cash_for_crypto=True) == 555.0


def test_allocation_usd_for_symbol_allocation_usd_overrides_pct_and_mode():
    acct = _Account(cash=100.0, buying_power=1000.0, equity=500.0)
    assert (
        allocation_usd_for_symbol(
            acct,
            symbol="BTCUSD",
            allocation_usd=42.0,
            allocation_pct=0.5,
            allocation_mode="portfolio",
            symbols_count=10,
        )
        == 42.0
    )


def test_allocation_usd_for_symbol_pct_per_symbol_vs_portfolio():
    acct = _Account(cash=1000.0, buying_power=0.0, equity=1000.0)
    per_symbol = allocation_usd_for_symbol(
        acct,
        symbol="BTCUSD",
        allocation_usd=None,
        allocation_pct=0.5,
        allocation_mode="per_symbol",
        symbols_count=4,
    )
    portfolio = allocation_usd_for_symbol(
        acct,
        symbol="BTCUSD",
        allocation_usd=None,
        allocation_pct=0.5,
        allocation_mode="portfolio",
        symbols_count=4,
    )
    assert per_symbol == 500.0
    assert portfolio == 125.0


def test_allocation_usd_for_symbol_rejects_negative_pct():
    acct = _Account(cash=100.0, buying_power=1000.0, equity=500.0)
    with pytest.raises(ValueError):
        allocation_usd_for_symbol(
            acct,
            symbol="NVDA",
            allocation_usd=None,
            allocation_pct=-0.1,
        )

