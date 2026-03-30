"""Shared market-data bundle contract for stockagent variants."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, cast

import pandas as pd

from stock_data_utils import add_ohlc_percent_change


@dataclass
class MarketDataBundle:
    bars: Dict[str, pd.DataFrame]
    lookback_days: int
    as_of: datetime

    def get_symbol_bars(self, symbol: str) -> pd.DataFrame:
        return self.bars.get(symbol.upper(), pd.DataFrame()).copy()

    def trading_days(self) -> List[pd.Timestamp]:
        for df in self.bars.values():
            if not df.empty:
                return list(df.index)
        return []

    def to_payload(self, limit: Optional[int] = None) -> Dict[str, List[Dict[str, float | str]]]:
        payload: Dict[str, List[Dict[str, float | str]]] = {}
        for symbol, df in self.bars.items():
            frame = df.tail(limit) if limit else df
            frame_with_pct = add_ohlc_percent_change(frame)
            payload[symbol] = []
            for _, row in frame_with_pct.iterrows():
                timestamp = cast(pd.Timestamp, row.name)
                payload[symbol].append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "open_pct": float(row["open_pct"]),
                        "high_pct": float(row["high_pct"]),
                        "low_pct": float(row["low_pct"]),
                        "close_pct": float(row["close_pct"]),
                    }
                )
        return payload
