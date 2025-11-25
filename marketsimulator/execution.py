from __future__ import annotations

from typing import Tuple

from src.fixtures import all_crypto_symbols

MEGA_SYMBOLS = {
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "META",
    "GOOG",
    "BTCUSD",
    "ETHUSD",
    "SOLUSD",
}


def classify_liquidity(symbol: str) -> str:
    upper = symbol.upper()
    if upper in MEGA_SYMBOLS or upper.startswith("BTC") or upper.startswith("ETH"):
        return "mega"
    if upper in all_crypto_symbols:
        return "large"
    return "small"


def simulate_fill(
    side: str,
    intended_price: float,
    mid_price: float,
    vol_bps: float,
    notional: float,
    liquidity_tier: str,
) -> Tuple[float, float]:
    """Return (executed_price, slip_bps) applying a conservative slippage model."""
    tier = liquidity_tier.lower()
    if tier == "mega":
        base = 3.0
    elif tier == "large":
        base = 8.0
    else:
        base = 20.0
    size_penalty = min(30.0, 0.5 * (notional / 1_000_000.0))
    vol_penalty = 0.2 * max(vol_bps, 0.0)
    slip_bps = base + size_penalty + vol_penalty
    slip = intended_price * (slip_bps / 1e4)
    side_upper = side.upper()
    if side_upper == "BUY":
        exec_price = intended_price + slip
    else:
        exec_price = max(0.0, intended_price - slip)
    if mid_price > 0:
        # ensure we do not fill inside the spread unrealistically
        if side_upper == "BUY":
            exec_price = max(exec_price, mid_price)
        else:
            exec_price = min(exec_price, mid_price)
    return exec_price, slip_bps
