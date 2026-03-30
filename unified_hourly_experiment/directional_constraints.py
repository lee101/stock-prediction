from __future__ import annotations

from collections.abc import Sequence

from src.trade_directions import DEFAULT_LONG_ONLY_STOCKS, DEFAULT_SHORT_ONLY_STOCKS


_LONG_ONLY = set(DEFAULT_LONG_ONLY_STOCKS)
_SHORT_ONLY = set(DEFAULT_SHORT_ONLY_STOCKS)


def build_directional_constraints(symbols: Sequence[str]) -> dict[str, tuple[float, float]]:
    constraints: dict[str, tuple[float, float]] = {}
    for symbol in symbols:
        if symbol in _LONG_ONLY:
            constraints[symbol] = (1.0, 0.0)
        elif symbol in _SHORT_ONLY:
            constraints[symbol] = (0.0, 1.0)
        else:
            constraints[symbol] = (1.0, 1.0)
    return constraints


__all__ = ["build_directional_constraints"]
