"""Allocation sizing helpers for live trading and simulations.

These utilities centralize the logic for converting `allocation_usd` / `allocation_pct`
inputs into a per-symbol dollar allocation, with optional handling differences between
crypto (cash-funded) and equities (buying-power-funded).
"""

from __future__ import annotations

import math
from typing import Optional

from src.symbol_utils import is_crypto_symbol


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def allocation_base_usd(
    account: object,
    *,
    symbol: str,
    prefer_cash_for_crypto: bool = True,
) -> float:
    """Return the dollar base used for `allocation_pct` sizing.

    For crypto, orders are typically cash-funded (USD available balance). For equities, Alpaca
    exposes `buying_power` which can include margin. This helper chooses an appropriate base:

    - Crypto: use `account.cash` when `prefer_cash_for_crypto=True`.
    - Equities: prefer `account.buying_power` if present/positive, else fall back to `account.equity`.
    """

    symbol = str(symbol or "").upper()

    if prefer_cash_for_crypto and is_crypto_symbol(symbol):
        cash = _safe_float(getattr(account, "cash", 0.0), default=0.0)
        if cash > 0.0:
            return cash
        return _safe_float(getattr(account, "equity", 0.0), default=0.0)

    buying_power = _safe_float(getattr(account, "buying_power", 0.0), default=0.0)
    if buying_power > 0.0:
        return buying_power
    return _safe_float(getattr(account, "equity", 0.0), default=0.0)


def allocation_usd_for_symbol(
    account: object,
    *,
    symbol: str,
    allocation_usd: Optional[float],
    allocation_pct: Optional[float],
    allocation_mode: str = "per_symbol",
    symbols_count: int = 1,
    prefer_cash_for_crypto: bool = True,
) -> Optional[float]:
    """Compute a per-symbol USD allocation from `allocation_usd`/`allocation_pct`.

    Args:
        account: Alpaca account-like object (supports cash/buying_power/equity attributes).
        symbol: Trading symbol (used to pick cash vs buying_power base).
        allocation_usd: Fixed allocation in USD *per symbol* when provided.
        allocation_pct: Fraction of the chosen base when provided.
        allocation_mode:
            - "per_symbol": treat allocation_pct as per-symbol (legacy behaviour).
            - "portfolio": treat allocation_pct as portfolio-level and divide across symbols_count.
        symbols_count: Number of symbols to divide across when allocation_mode="portfolio".
        prefer_cash_for_crypto: Use `account.cash` as the base for crypto sizing.

    Returns:
        Per-symbol allocation in USD, or None when both allocation_usd and allocation_pct are unset.
    """

    if allocation_usd is not None:
        value = _safe_float(allocation_usd, default=0.0)
        return max(0.0, value)

    if allocation_pct is None:
        return None

    mode = str(allocation_mode or "per_symbol").strip().lower()
    if mode not in {"per_symbol", "portfolio"}:
        raise ValueError(f"allocation_mode must be 'per_symbol' or 'portfolio', got {allocation_mode!r}.")

    pct = _safe_float(allocation_pct, default=float("nan"))
    if not math.isfinite(pct):
        raise ValueError(f"allocation_pct must be finite, got {allocation_pct!r}.")
    if pct < 0.0:
        raise ValueError(f"allocation_pct must be >= 0, got {allocation_pct!r}.")

    base = allocation_base_usd(account, symbol=str(symbol), prefer_cash_for_crypto=prefer_cash_for_crypto)
    allocation = pct * max(0.0, base)
    if mode == "portfolio":
        count = int(symbols_count)
        if count <= 0:
            raise ValueError(f"symbols_count must be >= 1 when allocation_mode='portfolio', got {symbols_count!r}.")
        allocation /= float(count)

    return max(0.0, float(allocation))


__all__ = [
    "allocation_base_usd",
    "allocation_usd_for_symbol",
]

