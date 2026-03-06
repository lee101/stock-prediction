#!/usr/bin/env python3
"""Daemon: download fresh OHLC bars then refresh Chronos2 forecast caches."""
import time
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_hourly_experiment.rebuild_all_caches import BEST_MODELS, build_cache

REFRESH_INTERVAL = 3600
OHLC_DIR = Path("trainingdatahourly/stocks")
STOCK_CACHE_HORIZONS = os.getenv("STOCK_CACHE_HORIZONS", "1,24")
STOCK_CACHE_LOOKBACK_HOURS = int(os.getenv("STOCK_CACHE_LOOKBACK_HOURS", "8000"))


def _refresh_ohlc(symbols):
    """Incrementally download new OHLC bars from Alpaca for each symbol."""
    from alpaca.data import StockBarsRequest, StockHistoricalDataClient, TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
    from alpaca_wrapper import _normalize_bar_frame

    client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    hourly = TimeFrame(1, TimeFrameUnit.Hour)
    now = datetime.now(timezone.utc)

    for symbol in symbols:
        csv_path = OHLC_DIR / f"{symbol}.csv"
        if not csv_path.exists():
            logger.warning("No CSV for {}, skipping OHLC refresh", symbol)
            continue

        try:
            existing = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
            last_ts = existing.index.max()
            if pd.isna(last_ts):
                start = now - timedelta(days=30)
            else:
                start = last_ts.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(hours=1)

            if start >= now:
                logger.debug("{} already up to date", symbol)
                continue

            try:
                req = StockBarsRequest(
                    symbol_or_symbols=symbol, timeframe=hourly,
                    start=start, end=now, adjustment="raw", feed=DataFeed.IEX,
                )
                bars = client.get_stock_bars(req).df
            except Exception:
                req = StockBarsRequest(
                    symbol_or_symbols=symbol, timeframe=hourly,
                    start=start, end=now, adjustment="raw", feed=DataFeed.SIP,
                )
                bars = client.get_stock_bars(req).df

            if bars.empty:
                logger.debug("{} no new bars since {}", symbol, start)
                continue

            new_df = _normalize_bar_frame(symbol, bars)
            if "symbol" in new_df.columns:
                new_df = new_df.drop(columns=["symbol"])
            new_df.index.name = "timestamp"
            if "symbol" in existing.columns:
                new_df["symbol"] = symbol

            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()

            tmp = csv_path.with_suffix(".csv.tmp")
            combined.to_csv(tmp)
            tmp.replace(csv_path)
            logger.info("{} +{} bars (now {} total, last={})",
                        symbol, len(new_df), len(combined),
                        combined.index.max())
            time.sleep(0.3)

        except Exception as e:
            logger.error("OHLC refresh failed for {}: {}", symbol, e)


def main():
    logger.info("Stock cache refresh daemon starting ({} symbols)", len(BEST_MODELS))
    symbols = list(BEST_MODELS.keys())
    while True:
        logger.info("Refreshing OHLC data for {} symbols", len(symbols))
        _refresh_ohlc(symbols)

        logger.info("Rebuilding forecast caches")
        for symbol, model in BEST_MODELS.items():
            try:
                build_cache(
                    symbol,
                    model,
                    horizons=STOCK_CACHE_HORIZONS,
                    lookback_hours=STOCK_CACHE_LOOKBACK_HOURS,
                )
            except Exception as e:
                logger.error("Cache refresh failed for {}: {}", symbol, e)

        logger.info("Refresh complete, sleeping {}s", REFRESH_INTERVAL)
        time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    main()
