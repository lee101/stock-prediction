"""In-process TradingServerEngine for deterministic paper-trading backtests.

:class:`TradingServerEngine` implements the same interface as the remote HTTP
trading server but runs entirely in memory – no network calls, no persistence
between instantiations (unless *state_dir* is provided).
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


class _AccountState:
    """Mutable state for a single paper-trading account."""

    def __init__(
        self,
        *,
        starting_cash: float,
        allowed_bot_id: str,
        symbols: List[str],
        sell_loss_cooldown_seconds: int = 0,
        min_sell_markup_pct: float = 0.0,
    ) -> None:
        self.cash: float = starting_cash
        self.allowed_bot_id = allowed_bot_id
        self.symbols = [str(s).upper() for s in symbols]
        self.sell_loss_cooldown_seconds = sell_loss_cooldown_seconds
        self.min_sell_markup_pct = min_sell_markup_pct
        # symbol -> {qty, avg_entry_price, opened_at}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        # Current writer session
        self.writer_bot_id: Optional[str] = None
        self.writer_session_id: Optional[str] = None


class TradingServerEngine:
    """In-process trading server engine used for deterministic backtesting.

    The engine supports multiple named accounts, each loaded from the
    *registry_path* JSON file.  State is held entirely in memory; optional
    *state_dir* is reserved for future persistence support.

    Args:
        registry_path: Path to a JSON file that describes accounts.  Format::

            {
              "accounts": {
                "<account_name>": {
                  "mode": "paper",
                  "allowed_bot_id": "<bot_id>",
                  "starting_cash": 10000.0,
                  "symbols": ["AAPL", "MSFT"],
                  "sell_loss_cooldown_seconds": 0,
                  "min_sell_markup_pct": 0.0
                }
              }
            }

        state_dir:      Optional directory for state snapshots (not yet used).
        quote_provider: Callable ``(symbol: str) -> dict | None`` that returns
                        the current quote for a symbol.
        now_fn:         Callable ``() -> datetime`` used to determine the current
                        time (injectable for deterministic backtests).
    """

    def __init__(
        self,
        *,
        registry_path: Path,
        state_dir: Optional[Path] = None,
        quote_provider: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._quote_provider = quote_provider or (lambda symbol: None)
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._accounts: Dict[str, _AccountState] = {}
        self._prices: Dict[str, float] = {}

        registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        for acct_name, cfg in registry.get("accounts", {}).items():
            self._accounts[acct_name] = _AccountState(
                starting_cash=float(cfg.get("starting_cash", 10_000.0)),
                allowed_bot_id=str(cfg.get("allowed_bot_id", "")),
                symbols=cfg.get("symbols", []),
                sell_loss_cooldown_seconds=int(cfg.get("sell_loss_cooldown_seconds", 0)),
                min_sell_markup_pct=float(cfg.get("min_sell_markup_pct", 0.0)),
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _now(self) -> datetime:
        return self._now_fn()

    def _get_account(self, account: str) -> _AccountState:
        state = self._accounts.get(account)
        if state is None:
            raise ValueError(f"Unknown account: {account!r}")
        return state

    def _current_price(self, symbol: str) -> float:
        sym = str(symbol).upper()
        cached = self._prices.get(sym)
        if cached is not None:
            return cached
        quote = self._quote_provider(sym)
        if quote and isinstance(quote, dict):
            for key in ("last_price", "bid_price", "ask_price"):
                v = quote.get(key)
                if v:
                    return float(v)
        return 0.0

    # ------------------------------------------------------------------
    # Engine API (mirroring the InMemoryTradingServerClient adapter)
    # ------------------------------------------------------------------

    def claim_writer(self, request: Any) -> Dict[str, Any]:
        """Claim exclusive write access for a bot on an account."""
        account = str(getattr(request, "account", ""))
        bot_id = str(getattr(request, "bot_id", ""))
        session_id = str(getattr(request, "session_id", ""))
        state = self._get_account(account)
        state.writer_bot_id = bot_id
        state.writer_session_id = session_id
        return {"status": "ok", "account": account, "bot_id": bot_id, "session_id": session_id}

    def refresh_prices(self, *, account: str, symbols: Iterable[str]) -> Dict[str, Any]:
        """Refresh cached prices for the given symbols."""
        updated: Dict[str, float] = {}
        for sym in symbols:
            sym = str(sym).upper()
            quote = self._quote_provider(sym)
            if quote and isinstance(quote, dict):
                for key in ("last_price", "bid_price", "ask_price"):
                    v = quote.get(key)
                    if v:
                        self._prices[sym] = float(v)
                        updated[sym] = float(v)
                        break
        return {"status": "ok", "updated": updated}

    def get_account_snapshot(self, account: str) -> Dict[str, Any]:
        """Return a snapshot of the account: cash, positions, order_history."""
        state = self._get_account(account)
        positions_out: Dict[str, Any] = {}
        for sym, pos in state.positions.items():
            qty = float(pos.get("qty", 0.0))
            if qty == 0.0:
                continue
            avg_entry_price = float(pos.get("avg_entry_price", 0.0))
            current_price = self._current_price(sym) or avg_entry_price
            positions_out[sym] = {
                "qty": qty,
                "avg_entry_price": avg_entry_price,
                "current_price": current_price,
                "opened_at": pos.get("opened_at"),
            }
        equity = state.cash + sum(
            float(p["qty"]) * self._current_price(sym)
            for sym, p in state.positions.items()
        )
        return {
            "account": account,
            "cash": state.cash,
            "equity": equity,
            "positions": positions_out,
            "order_history": list(state.order_history),
        }

    def submit_order(self, request: Any) -> Dict[str, Any]:
        """Execute a limit order immediately (paper mode)."""
        account = str(getattr(request, "account", ""))
        symbol = str(getattr(request, "symbol", "")).upper()
        side = str(getattr(request, "side", "")).lower()
        qty = float(getattr(request, "qty", 0.0))
        limit_price = float(getattr(request, "limit_price", 0.0))
        metadata = getattr(request, "metadata", {}) or {}

        state = self._get_account(account)
        now_iso = self._now().isoformat()
        order_id = str(uuid.uuid4())

        if side == "buy":
            cost = qty * limit_price
            if cost > state.cash:
                # Reduce qty to what cash allows
                qty = max(0.0, state.cash / limit_price)
                cost = qty * limit_price
            if qty > 0:
                state.cash -= cost
                existing = state.positions.get(symbol)
                if existing:
                    total_qty = float(existing["qty"]) + qty
                    existing["avg_entry_price"] = (
                        float(existing["avg_entry_price"]) * float(existing["qty"]) + limit_price * qty
                    ) / total_qty
                    existing["qty"] = total_qty
                else:
                    state.positions[symbol] = {
                        "qty": qty,
                        "avg_entry_price": limit_price,
                        "opened_at": now_iso,
                    }

        elif side == "sell":
            existing = state.positions.get(symbol)
            if existing:
                sell_qty = min(qty, float(existing["qty"]))
                proceeds = sell_qty * limit_price
                state.cash += proceeds
                remaining = float(existing["qty"]) - sell_qty
                if remaining <= 1e-6:
                    del state.positions[symbol]
                else:
                    existing["qty"] = remaining

        order_record = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "limit_price": limit_price,
            "filled_price": limit_price,
            "filled_at": now_iso,
            "metadata": metadata,
        }
        state.order_history.append(order_record)
        logger.debug(
            "TradingServerEngine: %s %s %.4f @ %.4f (account=%s)",
            side.upper(),
            symbol,
            qty,
            limit_price,
            account,
        )
        return {"status": "ok", "order": order_record}
