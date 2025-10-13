from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from .state import get_state


def download_daily_stock_data(cache_key: str, symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    state = get_state()
    if not symbols:
        symbols = state.symbols()
    frames = []
    for symbol in symbols:
        series = state.prices.get(symbol)
        if series is None:
            continue
        frame = series.frame.copy()
        frame["symbol"] = symbol
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def download_exchange_latest_data(client, symbol: str) -> None:
    # No-op in simulation; prices are pulled from the loaded dataset.
    return None


def get_bid(symbol: str) -> Optional[float]:
    state = get_state()
    bid = state.current_bid(symbol)
    if bid is None:
        series = state.prices.get(symbol)
        if series is not None:
            bid = float(series.current_row.get("Close"))
    return bid


def get_ask(symbol: str) -> Optional[float]:
    state = get_state()
    ask = state.current_ask(symbol)
    if ask is None:
        series = state.prices.get(symbol)
        if series is not None:
            ask = float(series.current_row.get("Close"))
    return ask


def fetch_spread(symbol: str) -> float:
    state = get_state()
    bid = state.current_bid(symbol)
    ask = state.current_ask(symbol)
    if bid is None or ask is None:
        return 0.0
    if bid == 0:
        return 0.0
    return ask / bid
