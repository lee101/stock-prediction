from types import SimpleNamespace
from datetime import timezone

import pytest

from stockagent.agentsimulator import account_state
from stockagent.agentsimulator.data_models import AccountPosition


def test_get_account_snapshot_filters_bad_positions(monkeypatch) -> None:
    account = SimpleNamespace(equity="1500", cash="700", buying_power="2000")
    good_position = SimpleNamespace(
        symbol="aapl",
        qty="5",
        side="long",
        market_value="750",
        avg_entry_price="100",
        unrealized_pl="5",
        unrealized_plpc="0.02",
    )
    bad_position = SimpleNamespace(symbol="bad", qty="?", side="long", market_value="0", avg_entry_price="0")

    monkeypatch.setattr(account_state.alpaca_wrapper, "get_account", lambda: account)
    monkeypatch.setattr(
        account_state.alpaca_wrapper,
        "get_all_positions",
        lambda: [good_position, bad_position],
    )

    def fake_from_alpaca(cls, position_obj):
        if getattr(position_obj, "symbol", "").lower() == "bad":
            raise ValueError("malformed position")
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
    assert snapshot.equity == 1500.0
    assert snapshot.cash == 700.0
    assert snapshot.buying_power == 2000.0
    assert snapshot.positions and snapshot.positions[0].symbol == "AAPL"
    assert snapshot.positions[0].quantity == 5.0
    assert snapshot.timestamp.tzinfo is timezone.utc


def test_get_account_snapshot_propagates_account_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        account_state.alpaca_wrapper,
        "get_account",
        lambda: (_ for _ in ()).throw(RuntimeError("api down")),
    )

    with pytest.raises(RuntimeError, match="api down"):
        account_state.get_account_snapshot()
