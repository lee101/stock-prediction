from __future__ import annotations

from types import SimpleNamespace

import pytest

import trade_daily_stock_prod as daily_stock


class _FakeClient:
    def __init__(self, positions: list[object], *, portfolio_value: float = 10_000.0, buying_power: float = 10_000.0):
        self._positions = positions
        self._account = SimpleNamespace(portfolio_value=portfolio_value, buying_power=buying_power)

    def get_all_positions(self):
        return list(self._positions)

    def get_account(self):
        return self._account


def test_effective_signal_allocation_pct_scales_base_allocation() -> None:
    signal = SimpleNamespace(allocation_pct=0.4)

    effective = daily_stock.effective_signal_allocation_pct(signal, base_allocation_pct=25.0)

    assert effective == pytest.approx(10.0)


def test_effective_signal_allocation_pct_defaults_to_base_when_missing() -> None:
    signal = SimpleNamespace()

    effective = daily_stock.effective_signal_allocation_pct(signal, base_allocation_pct=25.0)

    assert effective == pytest.approx(25.0)


def test_execute_signal_scales_open_size_by_signal_allocation(monkeypatch) -> None:
    captured: dict[str, float] = {}

    def _fake_compute_target_qty(*, account, price: float, allocation_pct: float) -> float:
        del account, price
        captured["allocation_pct"] = allocation_pct
        return 1.0

    def _fake_submit_limit_order(client, *, symbol: str, qty: float, side: str, limit_price: float):
        del client, limit_price
        return SimpleNamespace(id=f"{symbol}-{side}-{qty}")

    monkeypatch.setattr(daily_stock, "compute_target_qty", _fake_compute_target_qty)
    monkeypatch.setattr(daily_stock, "submit_limit_order", _fake_submit_limit_order)

    client = _FakeClient([])
    state = daily_stock.StrategyState(active_symbol=None, active_qty=0.0)
    signal = SimpleNamespace(symbol="MSFT", direction="long", action="long_MSFT", allocation_pct=0.4)

    changed = daily_stock.execute_signal(
        signal,
        client=client,
        quotes={"MSFT": 50.0},
        state=state,
        symbols=["MSFT"],
        allocation_pct=25.0,
        dry_run=False,
    )

    assert changed is True
    assert captured["allocation_pct"] == pytest.approx(10.0)
