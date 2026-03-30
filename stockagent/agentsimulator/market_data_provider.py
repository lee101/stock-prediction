"""Shared market-data provider protocol for simulator stacks."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class MarketDataProvider(Protocol):
    def get_symbol_bars(self, symbol: str) -> pd.DataFrame: ...
