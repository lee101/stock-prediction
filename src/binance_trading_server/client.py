from __future__ import annotations

import uuid
from ipaddress import ip_address
from typing import Any, Iterable, Literal, TypedDict, cast
from urllib.parse import urlparse

import requests

from src.server_http_auth import format_bearer_auth_header, normalize_auth_token
from .settings import BinanceTradingServerSettings
from .settings import resolve_auth_token, resolve_binance_trading_server_base_url


ExecutionMode = Literal["paper", "live"]
OrderSide = Literal["buy", "sell"]
BinanceTradingServerBaseUrlTransport = Literal["http", "https", "missing", "unsupported"]
BinanceTradingServerBaseUrlScope = Literal["loopback", "remote", "invalid"]
BinanceTradingServerBaseUrlSecurity = Literal["https", "loopback_http", "insecure_remote_http", "invalid"]
BINANCE_TRADING_SERVER_ERROR_BODY_MAX_CHARS = 300


class BinanceTradingServerBaseUrlDetails(TypedDict):
    host: str | None
    transport: BinanceTradingServerBaseUrlTransport
    scope: BinanceTradingServerBaseUrlScope
    security: BinanceTradingServerBaseUrlSecurity


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
    raise ValueError(f"unsupported binance trading server execution mode: {value!r}")


def _normalize_order_side(value: str) -> OrderSide:
    normalized = str(value).strip().lower()
    if normalized in {"buy", "sell"}:
        return cast(OrderSide, normalized)
    raise ValueError(f"unsupported binance trading server order side: {value!r}")


def describe_binance_trading_server_base_url(base_url: str) -> BinanceTradingServerBaseUrlDetails:
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


def validate_binance_trading_server_base_url(*, base_url: str, execution_mode: str) -> None:
    normalized_mode = str(execution_mode).strip().lower() or "paper"
    if normalized_mode != "live":
        return
    if describe_binance_trading_server_base_url(base_url)["security"] in {"https", "loopback_http"}:
        return
    raise ValueError(
        "live binance trading_server requires an https URL unless the server targets loopback "
        f"(got {base_url!r})"
    )


def _response_body_excerpt(
    response: requests.Response | None,
    *,
    max_chars: int = BINANCE_TRADING_SERVER_ERROR_BODY_MAX_CHARS,
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


class BinanceTradingServerClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        account: str,
        bot_id: str,
        session_id: str | None = None,
        execution_mode: str = "paper",
        timeout: float = 10.0,
        auth_token: str | None = None,
    ) -> None:
        self.base_url = resolve_binance_trading_server_base_url(base_url)
        self.account = str(account).strip()
        self.bot_id = str(bot_id).strip()
        self.session_id = str(session_id or uuid.uuid4())
        self.execution_mode = _normalize_execution_mode(execution_mode)
        self.timeout = float(timeout)
        self.auth_token = normalize_auth_token(resolve_auth_token(auth_token))
        validate_binance_trading_server_base_url(
            base_url=self.base_url,
            execution_mode=self.execution_mode,
        )

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
                response = requests.get(url, params=params, **request_kwargs)
            else:
                response = requests.post(url, json=payload, **request_kwargs)
            response.raise_for_status()
        except requests.RequestException as exc:
            response = cast(requests.Response | None, getattr(exc, "response", response))
            message_parts = [
                f"binance trading server {method} {path} failed",
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
                f"binance trading server {method} {path} returned invalid JSON",
                f"url={url}",
                f"status={response.status_code}",
                f"error={type(exc).__name__}: {exc}",
            ]
            body_excerpt = _response_body_excerpt(response)
            if body_excerpt is not None:
                message_parts.append(f"body={body_excerpt}")
            raise RuntimeError(", ".join(message_parts)) from exc
        if not isinstance(decoded, dict):
            message_parts = [
                f"binance trading server {method} {path} returned unexpected JSON payload",
                f"url={url}",
                f"status={response.status_code}",
                f"type={type(decoded).__name__}",
            ]
            body_excerpt = _response_body_excerpt(response)
            if body_excerpt is not None:
                message_parts.append(f"body={body_excerpt}")
            raise RuntimeError(", ".join(message_parts))
        return cast(dict[str, Any], decoded)

    def claim_writer(self, *, ttl_seconds: int | None = None) -> dict[str, Any]:
        resolved = BinanceTradingServerSettings.from_env().writer_ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        return self._post("/api/v1/writer/claim", {
            "account": self.account, "bot_id": self.bot_id,
            "session_id": self.session_id, "ttl_seconds": resolved,
        })

    def heartbeat_writer(self, *, ttl_seconds: int | None = None) -> dict[str, Any]:
        resolved = BinanceTradingServerSettings.from_env().writer_ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        return self._post("/api/v1/writer/heartbeat", {
            "account": self.account, "bot_id": self.bot_id,
            "session_id": self.session_id, "ttl_seconds": resolved,
        })

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
    ) -> dict[str, Any]:
        normalized_side = _normalize_order_side(side)
        return self._post("/api/v1/orders", {
            "account": self.account, "bot_id": self.bot_id,
            "session_id": self.session_id, "symbol": symbol,
            "side": normalized_side, "qty": float(qty), "limit_price": float(limit_price),
            "execution_mode": self.execution_mode,
            "allow_loss_exit": bool(allow_loss_exit),
            "force_exit_reason": force_exit_reason,
            "live_ack": live_ack, "metadata": metadata or {},
        })

    def refresh_prices(self, *, symbols: Iterable[str] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"account": self.account}
        if symbols is not None:
            payload["symbols"] = [str(s).strip().upper() for s in symbols if str(s).strip()]
        return self._post("/api/v1/prices/refresh", payload)

    def get_account(self) -> dict[str, Any]:
        return self._get(f"/api/v1/account/{self.account}")

    def get_orders(self, *, include_history: bool = False) -> dict[str, Any]:
        return self._get(f"/api/v1/orders/{self.account}", include_history=str(bool(include_history)).lower())
