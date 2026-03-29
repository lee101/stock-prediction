"""HTTP client for the TradingServer REST API.

Provides :class:`TradingServerClient` – a thin wrapper around ``requests``
that talks to a running :class:`~src.trading_server.server.TradingServerEngine`
via HTTP.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


class TradingServerClient:
    """REST client for the TradingServer.

    Args:
        base_url:       Base URL of the trading server (e.g. ``"http://localhost:8080"``).
                        If ``None``, defaults to ``"http://localhost:8080"``.
        account:        Logical account name (e.g. ``"paper_daily_sortino"``).
        bot_id:         Bot identifier used for writer-lock claims.
        session_id:     Stable session identifier for idempotent operations.
        execution_mode: ``"paper"`` or ``"live"``.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        account: str,
        bot_id: str,
        session_id: Optional[str] = None,
        execution_mode: str = "paper",
    ) -> None:
        self.base_url = (base_url or "http://localhost:8080").rstrip("/")
        self.account = account
        self.bot_id = bot_id
        self.session_id = session_id or f"{bot_id}-session"
        self.execution_mode = execution_mode

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, body: dict) -> dict:
        import requests  # lazy import so the module is importable without requests

        url = f"{self.base_url}{path}"
        response = requests.post(url, json=body, timeout=30)
        response.raise_for_status()
        return response.json()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        import requests

        url = f"{self.base_url}{path}"
        response = requests.get(url, params=params or {}, timeout=30)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    def claim_writer(self, *, ttl_seconds: int = 120) -> dict:
        """Claim exclusive writer access for this bot on the account."""
        return self._post(
            "/writer/claim",
            {
                "account": self.account,
                "bot_id": self.bot_id,
                "session_id": self.session_id,
                "ttl_seconds": ttl_seconds,
            },
        )

    def refresh_prices(self, *, symbols: Optional[Iterable[str]] = None) -> dict:
        """Request the server to refresh cached prices for the given symbols."""
        return self._post(
            "/prices/refresh",
            {
                "account": self.account,
                "symbols": list(symbols or []),
            },
        )

    def get_account(self) -> dict:
        """Return a snapshot of the account state (cash, positions, history)."""
        return self._get("/account/snapshot", {"account": self.account})

    def submit_limit_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        allow_loss_exit: bool = False,
        force_exit_reason: Optional[str] = None,
        live_ack: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Submit a limit order to the trading server."""
        return self._post(
            "/orders/submit",
            {
                "account": self.account,
                "bot_id": self.bot_id,
                "session_id": self.session_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "limit_price": limit_price,
                "execution_mode": self.execution_mode,
                "allow_loss_exit": allow_loss_exit,
                "force_exit_reason": force_exit_reason,
                "live_ack": live_ack,
                "metadata": metadata or {},
            },
        )
