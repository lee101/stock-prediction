"""Utilities for asset-specific trading fees.

Prefers metadata from ``hftraining.asset_metadata`` when a symbol record exists,
and otherwise falls back to heuristics + workspace constants. Returned fee values
are decimal rates (e.g., ``0.0005`` == 5 bps) suitable for multiplication with
notional turnover.
"""

from __future__ import annotations

from typing import Iterable, List

from src.symbol_utils import is_crypto_symbol as _is_crypto_symbol_canonical


def _is_crypto_symbol(symbol: str) -> bool:
    # Backwards-compatible alias for fee heuristics; use the canonical detector.
    return bool(_is_crypto_symbol_canonical(symbol))


def _symbol_candidates(symbol: str) -> List[str]:
    raw = str(symbol).strip().upper()
    if not raw:
        return []

    out: List[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        if value and value not in seen:
            seen.add(value)
            out.append(value)

    _push(raw)
    compact = raw.replace("/", "").replace("-", "")
    _push(compact)

    # Add common quote-split forms for metadata stores that use "BASE-QUOTE"/"BASE/QUOTE".
    for quote in ("USD", "USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI"):
        if compact.endswith(quote) and len(compact) > len(quote):
            base = compact[: -len(quote)]
            _push(f"{base}-{quote}")
            _push(f"{base}/{quote}")
            break
    return out


def get_fee_for_symbol(symbol: str) -> float:
    """Return the per-side trading fee rate for a symbol.

    Order of precedence:
      1) ``hftraining.asset_metadata`` when a record exists for the symbol
         (including normalized variants like ``LINKUSD`` -> ``LINK-USD``).
      2) Workspace constants from ``stockagent.constants``.
      3) Heuristic: symbols ending in ``USD`` or containing ``-USD`` are crypto.
    """
    try:  # Prefer precise metadata if available for this symbol.
        from hftraining.asset_metadata import get_asset_record, get_trading_fee  # type: ignore

        for candidate in _symbol_candidates(symbol):
            record = get_asset_record(candidate)
            if record:
                return float(get_trading_fee(candidate))
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
