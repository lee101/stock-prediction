"""Utilities for assembling recent OHLC data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, cast

import pandas as pd

from src.fixtures import crypto_symbols
from src.stock_utils import remap_symbols

from ..constants import DEFAULT_SYMBOLS
from .local_market_data import (
    USE_DEFAULT,
    build_market_data_bundle as _build_market_data_bundle,
    normalize_market_symbol as _normalize_market_symbol,
    resolve_local_data_dirs as _shared_resolve_local_data_dirs,
)
from .market_data_bundle import MarketDataBundle

FALLBACK_DATA_DIRS = [
    Path("trainingdata/stockagent/marketdata"),
    Path("stockagent_market_data"),
    Path("trainingdata/marketdata"),
    Path("data"),
    Path("data2"),
]

def fetch_latest_ohlc(
    symbols: Optional[Iterable[str]] = None,
    lookback_days: int = 60,
    as_of: Optional[datetime] = None,
    local_data_dir: Optional[Path] | object = USE_DEFAULT,
    allow_remote_download: bool = False,
    use_fallback_data_dirs: bool | object = USE_DEFAULT,
) -> MarketDataBundle:
    as_of = as_of or datetime.now(timezone.utc)
    start = as_of - timedelta(days=max(lookback_days * 2, 30))

    def _remote_loader(symbol: str) -> pd.DataFrame:
        return _download_remote_bars(symbol, start, as_of)

    return _build_market_data_bundle(
        symbols=symbols,
        default_symbols=DEFAULT_SYMBOLS,
        lookback_days=lookback_days,
        as_of=as_of,
        local_data_dir=local_data_dir,
        fallback_data_dirs=FALLBACK_DATA_DIRS,
        use_fallback_data_dirs=use_fallback_data_dirs,
        remote_loader=_remote_loader if allow_remote_download else None,
    )


def resolve_local_data_dirs(
    *,
    local_data_dir: Optional[Path] | object = USE_DEFAULT,
    use_fallback_data_dirs: bool | object = USE_DEFAULT,
) -> List[Path]:
    return _shared_resolve_local_data_dirs(
        local_data_dir=local_data_dir,
        fallback_data_dirs=FALLBACK_DATA_DIRS,
        use_fallback_data_dirs=use_fallback_data_dirs,
    )


def normalize_market_symbol(raw: str) -> str:
    return _normalize_market_symbol(raw)

def _download_remote_bars(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        from alpaca.data import CryptoBarsRequest, StockBarsRequest, TimeFrame, TimeFrameUnit
        from alpaca.data.enums import Adjustment
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
        from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
    except Exception as exc:
        logger.warning(f"Alpaca dependencies unavailable for {symbol}: {exc}")
        return pd.DataFrame()

    try:
        stock_client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
        crypto_client = CryptoHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
        day_unit = cast(TimeFrameUnit, TimeFrameUnit.Day)
        if symbol in crypto_symbols:
            request = CryptoBarsRequest(
                symbol_or_symbols=remap_symbols(symbol),
                timeframe=TimeFrame(1, day_unit),
                start=start,
                end=end,
            )
            df = crypto_client.get_crypto_bars(request).df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(remap_symbols(symbol), level="symbol")
        else:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, day_unit),
                start=start,
                end=end,
                adjustment=Adjustment.RAW,
            )
            df = stock_client.get_stock_bars(request).df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level="symbol")
        return df
    except Exception as exc:
        logger.warning(f"Failed to download bars for {symbol}: {exc}")
        return pd.DataFrame()
