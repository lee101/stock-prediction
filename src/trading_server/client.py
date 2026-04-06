from __future__ import annotations

import uuid
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any, Iterable, Literal, NotRequired, Protocol, TypedDict, cast
from urllib.parse import urlparse

import requests

from src.json_types import JsonObject
from src.order_validation import (
    MAX_ORDER_METADATA_BYTES,
    MAX_ORDER_METADATA_ITEMS,
    normalize_order_metadata,
    normalize_positive_finite_float,
)
from src.server_http_auth import format_bearer_auth_header, normalize_auth_token
from .settings import (
    TradingServerSettings,
    resolve_trading_server_auth_token,
    resolve_trading_server_base_url,
)


TradingMode = Literal["paper", "live"]
ExecutionMode = Literal["paper", "live"]
OrderSide = Literal["buy", "sell"]
OrderStatus = Literal["open", "filled", "submitted"]
TradingServerBaseUrlTransport = Literal["http", "https", "missing", "unsupported"]
TradingServerBaseUrlScope = Literal["loopback", "remote", "invalid"]
TradingServerBaseUrlSecurity = Literal["https", "loopback_http", "insecure_remote_http", "invalid"]
TRADING_SERVER_ERROR_BODY_MAX_CHARS = 300


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
    status: OrderStatus
    execution_mode: ExecutionMode
    allow_loss_exit: bool
    force_exit_reason: str | None
    metadata: JsonObject
    created_at: str | None
    updated_at: str | None
    filled_at: str | None
    fill_price: float | None
    broker_response: JsonObject


class TradingServerOrderSubmitResponse(TypedDict):
    order: TradingServerOrderPayload
    quote: TradingServerQuotePayload | None
    filled: bool


class TradingServerRefreshAccountResult(TypedDict):
    account: str
    refreshed_symbols: list[str]
    unavailable_symbols: list[str]
    quote_error_symbols: NotRequired[list[str]]
    unavailable_reasons: NotRequired[dict[str, str]]
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


class TradingServerBaseUrlDetails(TypedDict):
    host: str | None
    transport: TradingServerBaseUrlTransport
    scope: TradingServerBaseUrlScope
    security: TradingServerBaseUrlSecurity


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
    side: OrderSide
    qty: float
    limit_price: float
    execution_mode: ExecutionMode
    allow_loss_exit: bool
    force_exit_reason: str | None
    live_ack: str | None
    metadata: JsonObject


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


def _is_loopback_hostname(hostname: str | None) -> bool:
    normalized = str(hostname or "").strip().strip("[]").lower()
    if not normalized:
        return False
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _normalize_execution_mode(value: str) -> ExecutionMode:
    normalized = str(value).strip().lower() or "paper"
    if normalized in {"paper", "live"}:
        return cast(ExecutionMode, normalized)
    raise ValueError(f"unsupported trading server execution mode: {value!r}")


def _normalize_order_side(value: str) -> OrderSide:
    normalized = str(value).strip().lower()
    if normalized in {"buy", "sell"}:
        return cast(OrderSide, normalized)
    raise ValueError(f"unsupported trading server order side: {value!r}")

def _normalize_order_metadata(value: JsonObject | None) -> JsonObject:
    return cast(
        JsonObject,
        normalize_order_metadata(
            value,
            metadata_label="trading server metadata",
            max_items=MAX_ORDER_METADATA_ITEMS,
            max_bytes=MAX_ORDER_METADATA_BYTES,
        ),
    )


def describe_trading_server_base_url(base_url: str) -> TradingServerBaseUrlDetails:
    parsed = urlparse(str(base_url).strip())
    scheme = parsed.scheme.strip().lower()
    host = parsed.hostname.strip().lower() if parsed.hostname else None
    if not scheme or not parsed.netloc:
        return {
            "host": host,
            "transport": "missing" if not scheme else "unsupported",
            "scope": "invalid",
            "security": "invalid",
        }
    if scheme == "https":
        return {
            "host": host,
            "transport": "https",
            "scope": "loopback" if _is_loopback_hostname(parsed.hostname) else "remote",
            "security": "https",
        }
    if scheme != "http":
        return {
            "host": host,
            "transport": "unsupported",
            "scope": "invalid",
            "security": "invalid",
        }
    if _is_loopback_hostname(parsed.hostname):
        return {
            "host": host,
            "transport": "http",
            "scope": "loopback",
            "security": "loopback_http",
        }
    return {
        "host": host,
        "transport": "http",
        "scope": "remote",
        "security": "insecure_remote_http",
    }


def is_secure_or_loopback_trading_server_url(base_url: str) -> bool:
    return describe_trading_server_base_url(base_url)["security"] in {"https", "loopback_http"}


def validate_trading_server_base_url(*, base_url: str, execution_mode: str) -> None:
    normalized_mode = str(execution_mode).strip().lower() or "paper"
    if normalized_mode != "live":
        return
    if is_secure_or_loopback_trading_server_url(base_url):
        return
    raise ValueError(
        "live trading_server requires an https URL unless the server targets loopback "
        f"(got {base_url!r})"
    )


def _response_body_excerpt(
    response: requests.Response | None,
    *,
    max_chars: int = TRADING_SERVER_ERROR_BODY_MAX_CHARS,
) -> str | None:
    if response is None:
        return None
    try:
        text = str(response.text or "").strip()
    except Exception:
        return None
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _raise_response_contract_error(
    *,
    method: Literal["GET", "POST"],
    path: str,
    url: str,
    response: requests.Response,
    detail: str,
) -> None:
    message_parts = [
        f"trading server {method} {path} returned unexpected JSON payload",
        f"url={url}",
        f"status={response.status_code}",
        detail,
    ]
    body_excerpt = _response_body_excerpt(response)
    if body_excerpt is not None:
        message_parts.append(f"body={body_excerpt}")
    raise RuntimeError(", ".join(message_parts))


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
        side: OrderSide,
        qty: float,
        limit_price: float,
        allow_loss_exit: bool = False,
        force_exit_reason: str | None = None,
        live_ack: str | None = None,
        metadata: JsonObject | None = None,
    ) -> TradingServerOrderSubmitResponse: ...

    def refresh_prices(self, *, symbols: Iterable[str] | None = None) -> TradingServerRefreshResponse: ...

    def get_account(self) -> TradingServerAccountSnapshot: ...

    def get_orders(self, *, include_history: bool = False) -> TradingServerOrdersResponse: ...

    def close(self) -> None: ...


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
        execution_mode: ExecutionMode = "paper",
        timeout: float = 10.0,
        session: requests.Session | None = None,
        auth_token: str | None = None,
    ) -> None:
        self.base_url = resolve_trading_server_base_url(base_url)
        self.account = str(account).strip()
        self.bot_id = str(bot_id).strip()
        self.session_id = str(session_id or uuid.uuid4())
        self.execution_mode = _normalize_execution_mode(execution_mode)
        self.timeout = float(timeout)
        self.auth_token = normalize_auth_token(resolve_trading_server_auth_token(auth_token))
        self._owns_session = session is None
        self._session = session or requests.Session()
        validate_trading_server_base_url(
            base_url=self.base_url,
            execution_mode=self.execution_mode,
        )

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> TradingServerClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        return self._request_json("GET", path, params=params)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json("POST", path, payload=payload)

    def _request_json(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        response: requests.Response | None = None
        request_kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self.auth_token is not None:
            request_kwargs["headers"] = {"Authorization": format_bearer_auth_header(self.auth_token)}
        try:
            if method == "GET":
                response = self._session.get(
                    url,
                    params=params,
                    **request_kwargs,
                )
            else:
                response = self._session.post(
                    url,
                    json=payload,
                    **request_kwargs,
                )
            response.raise_for_status()
        except requests.RequestException as exc:
            response = cast(requests.Response | None, getattr(exc, "response", response))
            message_parts = [
                f"trading server {method} {path} failed",
                f"url={url}",
                f"error={type(exc).__name__}: {exc}",
            ]
            if response is not None:
                message_parts.append(f"status={response.status_code}")
            body_excerpt = _response_body_excerpt(response)
            if body_excerpt is not None:
                message_parts.append(f"body={body_excerpt}")
            raise RuntimeError(", ".join(message_parts)) from exc
        try:
            decoded = response.json()
        except ValueError as exc:
            message_parts = [
                f"trading server {method} {path} returned invalid JSON",
                f"url={url}",
                f"status={response.status_code}",
                f"error={type(exc).__name__}: {exc}",
            ]
            body_excerpt = _response_body_excerpt(response)
            if body_excerpt is not None:
                message_parts.append(f"body={body_excerpt}")
            raise RuntimeError(", ".join(message_parts)) from exc
        if not isinstance(decoded, dict):
            _raise_response_contract_error(
                method=method,
                path=path,
                url=url,
                response=response,
                detail=f"type={type(decoded).__name__}",
            )
        return cast(dict[str, Any], decoded)

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
        side: OrderSide,
        qty: float,
        limit_price: float,
        allow_loss_exit: bool = False,
        force_exit_reason: str | None = None,
        live_ack: str | None = None,
        metadata: JsonObject | None = None,
    ) -> TradingServerOrderSubmitResponse:
        normalized_side = _normalize_order_side(side)
        normalized_qty = normalize_positive_finite_float(qty, field_name="qty")
        normalized_limit_price = normalize_positive_finite_float(limit_price, field_name="limit_price")
        normalized_metadata = _normalize_order_metadata(metadata)
        payload = {
            "account": self.account,
            "bot_id": self.bot_id,
            "session_id": self.session_id,
            "symbol": symbol,
            "side": normalized_side,
            "qty": normalized_qty,
            "limit_price": normalized_limit_price,
            "execution_mode": self.execution_mode,
            "allow_loss_exit": bool(allow_loss_exit),
            "force_exit_reason": force_exit_reason,
            "live_ack": live_ack,
            "metadata": normalized_metadata,
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
        execution_mode: ExecutionMode = "paper",
        session_id: str = "daily-stock-backtest",
    ) -> None:
        self.engine = engine
        self.account = str(account).strip()
        self.bot_id = str(bot_id).strip()
        self.execution_mode = _normalize_execution_mode(execution_mode)
        self.session_id = str(session_id).strip()

    def close(self) -> None:
        return None

    def __enter__(self) -> InMemoryTradingServerClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

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
        side: OrderSide,
        qty: float,
        limit_price: float,
        allow_loss_exit: bool = False,
        force_exit_reason: str | None = None,
        live_ack: str | None = None,
        metadata: JsonObject | None = None,
    ) -> TradingServerOrderSubmitResponse:
        normalized_side = _normalize_order_side(side)
        normalized_qty = normalize_positive_finite_float(qty, field_name="qty")
        normalized_limit_price = normalize_positive_finite_float(limit_price, field_name="limit_price")
        normalized_metadata = _normalize_order_metadata(metadata)
        return self.engine.submit_order(
            EngineOrderRequest(
                account=self.account,
                bot_id=self.bot_id,
                session_id=self.session_id,
                symbol=symbol,
                side=normalized_side,
                qty=normalized_qty,
                limit_price=normalized_limit_price,
                execution_mode=self.execution_mode,
                allow_loss_exit=allow_loss_exit,
                force_exit_reason=force_exit_reason,
                live_ack=live_ack,
                metadata=normalized_metadata,
            )
        )
