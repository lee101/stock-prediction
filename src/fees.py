"""
Utilities for asset-specific trading fees.

Prefers metadata from ``hftraining.asset_metadata`` when available; falls back
to basic heuristics and workspace constants otherwise. Returned fee values are
decimal rates (e.g., 0.0005 == 5 bps) suitable for multiplication with notional
turnover.
"""

from __future__ import annotations

from typing import Iterable, List


def _is_crypto_symbol(symbol: str) -> bool:
    s = symbol.upper()
    return s.endswith("USD") or "-USD" in s


def get_fee_for_symbol(symbol: str) -> float:
    """Return the per-side trading fee rate for a symbol.

    Order of precedence:
      1) ``hftraining.asset_metadata.get_trading_fee`` if importable.
      2) Workspace constants from ``stockagent.constants``.
      3) Heuristic: symbols ending in ``USD`` or containing ``-USD`` are crypto.
    """
    try:  # Prefer precise metadata if available
        from hftraining.asset_metadata import get_trading_fee  # type: ignore

        return float(get_trading_fee(symbol))
    except Exception:
        pass

    try:
        from stockagent.constants import TRADING_FEE, CRYPTO_TRADING_FEE  # type: ignore

        return float(CRYPTO_TRADING_FEE if _is_crypto_symbol(symbol) else TRADING_FEE)
    except Exception:
        # Conservative defaults: 5 bps equities, 10 bps crypto (standardized with loss_utils.py)
        return 0.001 if _is_crypto_symbol(symbol) else 0.0005


def get_fees_for_symbols(symbols: Iterable[str]) -> List[float]:
    """Vectorised helper returning fee rates for a sequence of symbols."""
    return [get_fee_for_symbol(sym) for sym in symbols]


__all__ = ["get_fee_for_symbol", "get_fees_for_symbols"]

