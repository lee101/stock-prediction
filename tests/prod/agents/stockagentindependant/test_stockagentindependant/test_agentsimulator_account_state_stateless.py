from types import SimpleNamespace
from datetime import timezone

import pytest

from stockagentindependant.agentsimulator import account_state
from stockagentindependant.agentsimulator.data_models import AccountPosition


def test_stateless_account_snapshot_handles_missing_positions(monkeypatch) -> None:
    account = SimpleNamespace(equity="2500", cash="1250", buying_power="4000")
    valid_position = SimpleNamespace(
        symbol="msft",
        qty="2",
        side="long",
        market_value="300",
        avg_entry_price="120",
        unrealized_pl="5",
        unrealized_plpc="0.04",
    )
    invalid_position = SimpleNamespace(symbol="oops", qty=None, side="long", market_value="0", avg_entry_price="0")

    monkeypatch.setattr(account_state.alpaca_wrapper, "get_account", lambda: account)
    monkeypatch.setattr(
        account_state.alpaca_wrapper,
        "get_all_positions",
        lambda: [valid_position, invalid_position],
    )

    def fake_from_alpaca(cls, position_obj):
        if getattr(position_obj, "symbol", "") == "oops":
            raise ValueError("bad position")
        return cls(
            symbol=str(position_obj.symbol).upper(),
            quantity=float(position_obj.qty),
            side=str(position_obj.side),
            market_value=float(position_obj.market_value),
            avg_entry_price=float(position_obj.avg_entry_price),
            unrealized_pl=float(getattr(position_obj, "unrealized_pl", 0.0)),
            unrealized_plpc=float(getattr(position_obj, "unrealized_plpc", 0.0)),
        )

    monkeypatch.setattr(AccountPosition, "from_alpaca", classmethod(fake_from_alpaca))

    snapshot = account_state.get_account_snapshot()
    assert snapshot.equity == 2500.0
    assert snapshot.cash == 1250.0
    assert snapshot.buying_power == 4000.0
    assert len(snapshot.positions) == 1
    assert snapshot.positions[0].symbol == "MSFT"
    assert snapshot.timestamp.tzinfo is timezone.utc


def test_stateless_account_snapshot_raises_when_account_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        account_state.alpaca_wrapper,
        "get_account",
        lambda: (_ for _ in ()).throw(RuntimeError("alpaca down")),
    )
    with pytest.raises(RuntimeError, match="alpaca down"):
        account_state.get_account_snapshot()
