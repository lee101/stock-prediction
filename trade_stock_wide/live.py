from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.order_validation import normalize_positive_finite_float
from src.trading_server.client import (
    ExecutionMode,
    TradingServerClientLike,
    TradingServerOrderSubmitResponse,
)

from .planner import validate_long_price_levels
from .types import WideOrder


@dataclass(frozen=True)
class WideLiveExecutionConfig:
    execution_mode: ExecutionMode = "paper"
    writer_ttl_seconds: int | None = None
    require_writer_heartbeat: bool = True
    min_order_notional: float = 1.0


@dataclass(frozen=True)
class PreparedWideLiveOrder:
    symbol: str
    qty: float
    entry_price: float
    take_profit_price: float
    metadata: dict[str, object]


@dataclass(frozen=True)
class SubmittedWideLiveOrder:
    prepared: PreparedWideLiveOrder
    response: TradingServerOrderSubmitResponse


def prepare_live_entry(
    order: WideOrder,
    *,
    min_order_notional: float = 1.0,
) -> PreparedWideLiveOrder:
    candidate = order.candidate
    validate_long_price_levels(
        symbol=candidate.symbol,
        entry_price=candidate.entry_price,
        take_profit_price=candidate.take_profit_price,
    )
    reserved_notional = normalize_positive_finite_float(
        order.reserved_notional,
        field_name="reserved_notional",
    )
    normalized_entry = normalize_positive_finite_float(
        candidate.entry_price,
        field_name="entry_price",
    )
    normalized_take_profit = normalize_positive_finite_float(
        candidate.take_profit_price,
        field_name="take_profit_price",
    )
    qty = reserved_notional / normalized_entry
    normalized_qty = normalize_positive_finite_float(qty, field_name="qty")
    if reserved_notional < float(min_order_notional):
        raise ValueError(
            f"{candidate.symbol}: reserved_notional {reserved_notional:.6f} is below "
            f"min_order_notional {float(min_order_notional):.6f}"
        )
    metadata = {
        "strategy": candidate.strategy,
        "rank": int(order.rank),
        "reserved_notional": reserved_notional,
        "reserved_fraction_of_equity": float(order.reserved_fraction_of_equity),
        "planned_entry_price": normalized_entry,
        "planned_take_profit_price": normalized_take_profit,
        "expected_return_pct": float(candidate.expected_return_pct),
        "session_date": candidate.session_date,
        "forecasted_pnl": float(candidate.forecasted_pnl),
        "avg_return": float(candidate.avg_return),
        "price_relationship_validated": True,
    }
    return PreparedWideLiveOrder(
        symbol=candidate.symbol,
        qty=normalized_qty,
        entry_price=normalized_entry,
        take_profit_price=normalized_take_profit,
        metadata=metadata,
    )


def submit_live_entry_orders(
    orders: Sequence[WideOrder],
    *,
    client: TradingServerClientLike,
    config: WideLiveExecutionConfig | None = None,
) -> list[SubmittedWideLiveOrder]:
    live_config = config or WideLiveExecutionConfig()
    client.claim_writer(ttl_seconds=live_config.writer_ttl_seconds)
    submitted: list[SubmittedWideLiveOrder] = []
    for order in orders:
        if live_config.require_writer_heartbeat:
            client.heartbeat_writer(ttl_seconds=live_config.writer_ttl_seconds)
        prepared = prepare_live_entry(
            order,
            min_order_notional=live_config.min_order_notional,
        )
        validate_long_price_levels(
            symbol=prepared.symbol,
            entry_price=prepared.entry_price,
            take_profit_price=prepared.take_profit_price,
        )
        response = client.submit_limit_order(
            symbol=prepared.symbol,
            side="buy",
            qty=prepared.qty,
            limit_price=prepared.entry_price,
            allow_loss_exit=False,
            force_exit_reason=None,
            live_ack="LIVE" if live_config.execution_mode == "live" else None,
            metadata=prepared.metadata,
        )
        submitted.append(SubmittedWideLiveOrder(prepared=prepared, response=response))
    return submitted
