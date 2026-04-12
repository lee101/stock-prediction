from __future__ import annotations

from dataclasses import dataclass

import pytest

from trade_stock_wide.live import (
    WideLiveExecutionConfig,
    prepare_live_entry,
    submit_live_entry_orders,
)
from trade_stock_wide.types import WideCandidate, WideOrder


def _order(
    *,
    symbol: str = "AAPL",
    entry_price: float = 99.0,
    take_profit_price: float = 103.0,
    reserved_notional: float = 5_000.0,
) -> WideOrder:
    candidate = WideCandidate(
        symbol=symbol,
        strategy="maxdiff",
        forecasted_pnl=0.05,
        avg_return=0.02,
        last_close=100.0,
        entry_price=entry_price,
        take_profit_price=take_profit_price,
        predicted_high=take_profit_price,
        predicted_low=entry_price,
        realized_close=101.0,
        realized_high=103.0,
        realized_low=98.0,
        score=0.05,
        day_index=0,
        session_date="2026-04-10",
    )
    return WideOrder(
        rank=1,
        candidate=candidate,
        reserved_notional=reserved_notional,
        reserved_fraction_of_equity=0.5,
    )


@dataclass
class _StubClient:
    claimed: int = 0
    heartbeats: int = 0
    submitted: list[dict[str, object]] | None = None

    def __post_init__(self) -> None:
        if self.submitted is None:
            self.submitted = []

    def claim_writer(self, *, ttl_seconds: int | None = None):
        self.claimed += 1
        return {"account": "paper", "session_id": "test", "expires_at": "later"}

    def heartbeat_writer(self, *, ttl_seconds: int | None = None):
        self.heartbeats += 1
        return {"account": "paper", "session_id": "test", "expires_at": "later"}

    def submit_limit_order(self, **kwargs):
        assert self.submitted is not None
        self.submitted.append(kwargs)
        return {
            "order": {
                "symbol": kwargs["symbol"],
                "side": kwargs["side"],
                "qty": kwargs["qty"],
                "limit_price": kwargs["limit_price"],
                "metadata": kwargs["metadata"],
            },
            "quote": None,
            "filled": False,
        }

    def refresh_prices(self, *, symbols=None):
        raise NotImplementedError

    def get_account(self):
        raise NotImplementedError

    def get_orders(self, *, include_history: bool = False):
        raise NotImplementedError

    def close(self) -> None:
        return None


def test_prepare_live_entry_embeds_exit_target_metadata():
    prepared = prepare_live_entry(_order())

    assert prepared.qty == pytest.approx(5_000.0 / 99.0)
    assert prepared.metadata["planned_entry_price"] == pytest.approx(99.0)
    assert prepared.metadata["planned_take_profit_price"] == pytest.approx(103.0)
    assert prepared.metadata["price_relationship_validated"] is True


def test_prepare_live_entry_rejects_flipped_prices():
    with pytest.raises(ValueError, match="take_profit_price"):
        prepare_live_entry(_order(entry_price=100.0, take_profit_price=99.0))


def test_submit_live_entry_orders_claims_writer_and_uses_live_ack():
    client = _StubClient()
    submitted = submit_live_entry_orders(
        [_order(symbol="AAPL"), _order(symbol="MSFT", entry_price=49.0, take_profit_price=52.0)],
        client=client,
        config=WideLiveExecutionConfig(execution_mode="live", writer_ttl_seconds=30),
    )

    assert len(submitted) == 2
    assert client.claimed == 1
    assert client.heartbeats == 2
    assert client.submitted is not None
    assert [payload["symbol"] for payload in client.submitted] == ["AAPL", "MSFT"]
    assert all(payload["side"] == "buy" for payload in client.submitted)
    assert all(payload["live_ack"] == "LIVE" for payload in client.submitted)
    assert all(payload["allow_loss_exit"] is False for payload in client.submitted)


def test_submit_live_entry_orders_blocks_server_submit_on_invalid_prices():
    client = _StubClient()

    with pytest.raises(ValueError, match="invalid long price plan"):
        submit_live_entry_orders(
            [_order(entry_price=101.0, take_profit_price=100.0)],
            client=client,
            config=WideLiveExecutionConfig(execution_mode="paper"),
        )

    assert client.claimed == 1
    assert client.submitted == []
