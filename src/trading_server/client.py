from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Literal, NotRequired, Protocol, TypedDict, cast

import requests

from .settings import TradingServerSettings


TradingMode = Literal["paper", "live"]
ExecutionMode = Literal["paper", "live"]
OrderSide = Literal["buy", "sell"]


class TradingServerWriterLeaseResponse(TypedDict):
    account: str
    session_id: str
    expires_at: str
    bot_id: NotRequired[str]
    mode: NotRequired[TradingMode]


class TradingServerPublicWriterClaim(TypedDict):
    claimed_at: str | None
    expires_at: str | None
    active: bool


class TradingServerQuotePayload(TypedDict):
    symbol: str
    bid_price: float
    ask_price: float
    last_price: float
    as_of: str


class TradingServerPositionPayload(TypedDict, total=False):
    qty: float
    avg_entry_price: float
    current_price: float
    opened_at: str | None
    last_buy_at: str | None
    realized_pnl: float


class TradingServerOrderPayload(TypedDict, total=False):
    id: str
    symbol: str
    side: OrderSide
    qty: float
    limit_price: float
    status: str
    execution_mode: ExecutionMode
    allow_loss_exit: bool
    force_exit_reason: str | None
    metadata: dict[str, Any]
    created_at: str | None
    updated_at: str | None
    filled_at: str | None
    fill_price: float | None
    broker_response: dict[str, Any]


class TradingServerOrderSubmitResponse(TypedDict):
    order: TradingServerOrderPayload
    quote: TradingServerQuotePayload | None
    filled: bool


class TradingServerRefreshAccountResult(TypedDict):
    account: str
    refreshed_symbols: list[str]
    unavailable_symbols: list[str]
    filled_orders: list[str]


class TradingServerRefreshResponse(TypedDict):
    accounts: list[TradingServerRefreshAccountResult]


class TradingServerAccountSnapshot(TypedDict):
    account: str
    mode: TradingMode
    cash: float
    equity: NotRequired[float]
    buying_power: NotRequired[float]
    realized_pnl: float
    positions: dict[str, TradingServerPositionPayload]
    open_orders: list[TradingServerOrderPayload]
    writer_claim: TradingServerPublicWriterClaim | None
    updated_at: str | None


class TradingServerOrdersResponse(TypedDict):
    account: str
    open_orders: list[TradingServerOrderPayload]
    order_history: NotRequired[list[TradingServerOrderPayload]]


@dataclass(frozen=True)
class EngineWriterLeaseRequest:
    account: str
    bot_id: str
    session_id: str
    ttl_seconds: int


@dataclass(frozen=True)
class EngineOrderRequest:
    account: str
    bot_id: str
    session_id: str
    symbol: str
    side: str
    qty: float
    limit_price: float
    execution_mode: str
    allow_loss_exit: bool
    force_exit_reason: str | None
    live_ack: str | None
    metadata: dict[str, Any]


class InProcessTradingServerEngineLike(Protocol):
    def claim_writer(self, request: EngineWriterLeaseRequest) -> TradingServerWriterLeaseResponse: ...

    def heartbeat_writer(self, request: EngineWriterLeaseRequest) -> TradingServerWriterLeaseResponse: ...

    def refresh_prices(
        self,
        *,
        account: str | None = None,
        symbols: Iterable[str] | None = None,
    ) -> TradingServerRefreshResponse: ...

    def get_account_snapshot(self, account: str) -> TradingServerAccountSnapshot: ...

    def get_orders(self, account: str, *, include_history: bool = False) -> TradingServerOrdersResponse: ...

    def submit_order(self, request: EngineOrderRequest) -> TradingServerOrderSubmitResponse: ...


def resolve_writer_ttl_seconds(ttl_seconds: int | None = None) -> int:
    return (
        TradingServerSettings.from_env().writer_ttl_seconds
        if ttl_seconds is None
        else int(ttl_seconds)
    )


class TradingServerClientLike(Protocol):
    account: str
    bot_id: str
    session_id: str

    def claim_writer(self, *, ttl_seconds: int | None = None) -> TradingServerWriterLeaseResponse: ...

    def heartbeat_writer(self, *, ttl_seconds: int | None = None) -> TradingServerWriterLeaseResponse: ...

    def submit_limit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        allow_loss_exit: bool = False,
        force_exit_reason: str | None = None,
        live_ack: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TradingServerOrderSubmitResponse: ...

    def refresh_prices(self, *, symbols: Iterable[str] | None = None) -> TradingServerRefreshResponse: ...

    def get_account(self) -> TradingServerAccountSnapshot: ...

    def get_orders(self, *, include_history: bool = False) -> TradingServerOrdersResponse: ...


class TradingServerClient:
    """Thin client for the repo trading server.

    The server owns account state and writer ownership. Trading code should claim
    a writer lease once at startup, then include the same session on all
    mutating requests.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        account: str,
        bot_id: str,
        session_id: str | None = None,
        execution_mode: str = "paper",
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("TRADING_SERVER_URL", "http://127.0.0.1:8050")).rstrip("/")
        self.account = str(account).strip()
        self.bot_id = str(bot_id).strip()
        self.session_id = str(session_id or uuid.uuid4())
        self.execution_mode = str(execution_mode).strip().lower() or "paper"
        self.timeout = float(timeout)

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def claim_writer(self, *, ttl_seconds: int | None = None) -> TradingServerWriterLeaseResponse:
        resolved_ttl_seconds = resolve_writer_ttl_seconds(ttl_seconds)
        return cast(
            TradingServerWriterLeaseResponse,
            self._post(
                "/api/v1/writer/claim",
                {
                    "account": self.account,
                    "bot_id": self.bot_id,
                    "session_id": self.session_id,
                    "ttl_seconds": resolved_ttl_seconds,
                },
            ),
        )

    def heartbeat_writer(self, *, ttl_seconds: int | None = None) -> TradingServerWriterLeaseResponse:
        resolved_ttl_seconds = resolve_writer_ttl_seconds(ttl_seconds)
        return cast(
            TradingServerWriterLeaseResponse,
            self._post(
                "/api/v1/writer/heartbeat",
                {
                    "account": self.account,
                    "bot_id": self.bot_id,
                    "session_id": self.session_id,
                    "ttl_seconds": resolved_ttl_seconds,
                },
            ),
        )

    def submit_limit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        allow_loss_exit: bool = False,
        force_exit_reason: str | None = None,
        live_ack: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TradingServerOrderSubmitResponse:
        payload = {
            "account": self.account,
            "bot_id": self.bot_id,
            "session_id": self.session_id,
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "limit_price": float(limit_price),
            "execution_mode": self.execution_mode,
            "allow_loss_exit": bool(allow_loss_exit),
            "force_exit_reason": force_exit_reason,
            "live_ack": live_ack,
            "metadata": metadata or {},
        }
        return cast(TradingServerOrderSubmitResponse, self._post("/api/v1/orders", payload))

    def refresh_prices(self, *, symbols: Iterable[str] | None = None) -> TradingServerRefreshResponse:
        payload: dict[str, Any] = {"account": self.account}
        if symbols is not None:
            payload["symbols"] = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
        return cast(TradingServerRefreshResponse, self._post("/api/v1/prices/refresh", payload))

    def get_account(self) -> TradingServerAccountSnapshot:
        return cast(TradingServerAccountSnapshot, self._get(f"/api/v1/account/{self.account}"))

    def get_orders(self, *, include_history: bool = False) -> TradingServerOrdersResponse:
        return cast(
            TradingServerOrdersResponse,
            self._get(f"/api/v1/orders/{self.account}", include_history=str(bool(include_history)).lower()),
        )


class InMemoryTradingServerClient:
    """Adapter for exercising the trading server engine in process."""

    def __init__(
        self,
        *,
        engine: InProcessTradingServerEngineLike,
        account: str,
        bot_id: str,
        execution_mode: str = "paper",
        session_id: str = "daily-stock-backtest",
    ) -> None:
        self.engine = engine
        self.account = str(account).strip()
        self.bot_id = str(bot_id).strip()
        self.execution_mode = str(execution_mode).strip().lower() or "paper"
        self.session_id = str(session_id).strip()

    def claim_writer(self, *, ttl_seconds: int | None = None) -> TradingServerWriterLeaseResponse:
        return self.engine.claim_writer(
            EngineWriterLeaseRequest(
                account=self.account,
                bot_id=self.bot_id,
                session_id=self.session_id,
                ttl_seconds=resolve_writer_ttl_seconds(ttl_seconds),
            )
        )

    def heartbeat_writer(self, *, ttl_seconds: int | None = None) -> TradingServerWriterLeaseResponse:
        return self.engine.heartbeat_writer(
            EngineWriterLeaseRequest(
                account=self.account,
                bot_id=self.bot_id,
                session_id=self.session_id,
                ttl_seconds=resolve_writer_ttl_seconds(ttl_seconds),
            )
        )

    def refresh_prices(self, *, symbols: Iterable[str] | None = None) -> TradingServerRefreshResponse:
        return self.engine.refresh_prices(account=self.account, symbols=list(symbols or []))

    def get_account(self) -> TradingServerAccountSnapshot:
        return self.engine.get_account_snapshot(self.account)

    def get_orders(self, *, include_history: bool = False) -> TradingServerOrdersResponse:
        return self.engine.get_orders(self.account, include_history=include_history)

    def submit_limit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        allow_loss_exit: bool = False,
        force_exit_reason: str | None = None,
        live_ack: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TradingServerOrderSubmitResponse:
        return self.engine.submit_order(
            EngineOrderRequest(
                account=self.account,
                bot_id=self.bot_id,
                session_id=self.session_id,
                symbol=symbol,
                side=side,
                qty=qty,
                limit_price=limit_price,
                execution_mode=self.execution_mode,
                allow_loss_exit=allow_loss_exit,
                force_exit_reason=force_exit_reason,
                live_ack=live_ack,
                metadata=metadata or {},
            )
        )
