from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def fetch_1m_klines_batch(symbols: list[str]) -> dict[str, dict[str, float]]:
    try:
        from src.binan.binance_wrapper import get_client
        client = get_client()
    except Exception:
        return {}
    if client is None:
        return {}

    results: dict[str, dict[str, float]] = {}
    for symbol in symbols:
        try:
            klines = client.get_klines(symbol=symbol, interval="1m", limit=2)
            if klines and len(klines) >= 1:
                k = klines[-1]
                results[symbol] = {
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "timestamp": float(k[0]) / 1000.0,
                }
        except Exception:
            continue
    return results


def kline_to_quote(symbol: str, kline: dict[str, float]) -> dict[str, Any]:
    close = kline.get("close", 0.0)
    return {
        "symbol": symbol,
        "bid_price": close,
        "ask_price": close,
        "last_price": close,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
