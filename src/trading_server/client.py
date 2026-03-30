from __future__ import annotations

import os
import uuid
from typing import Any, Iterable

import requests

from .settings import TradingServerSettings


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

    def claim_writer(self, *, ttl_seconds: int | None = None) -> dict[str, Any]:
        resolved_ttl_seconds = (
            TradingServerSettings.from_env().writer_ttl_seconds
            if ttl_seconds is None
            else int(ttl_seconds)
        )
        return self._post(
            "/api/v1/writer/claim",
            {
                "account": self.account,
                "bot_id": self.bot_id,
                "session_id": self.session_id,
                "ttl_seconds": resolved_ttl_seconds,
            },
        )

    def heartbeat_writer(self, *, ttl_seconds: int | None = None) -> dict[str, Any]:
        resolved_ttl_seconds = (
            TradingServerSettings.from_env().writer_ttl_seconds
            if ttl_seconds is None
            else int(ttl_seconds)
        )
        return self._post(
            "/api/v1/writer/heartbeat",
            {
                "account": self.account,
                "bot_id": self.bot_id,
                "session_id": self.session_id,
                "ttl_seconds": resolved_ttl_seconds,
            },
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
    ) -> dict[str, Any]:
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
        return self._post("/api/v1/orders", payload)

    def refresh_prices(self, *, symbols: Iterable[str] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"account": self.account}
        if symbols is not None:
            payload["symbols"] = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
        return self._post("/api/v1/prices/refresh", payload)

    def get_account(self) -> dict[str, Any]:
        return self._get(f"/api/v1/account/{self.account}")

    def get_orders(self, *, include_history: bool = False) -> dict[str, Any]:
        return self._get(f"/api/v1/orders/{self.account}", include_history=str(bool(include_history)).lower())
