#!/usr/bin/env python3
"""Daemon: refresh OHLC + Chronos2 forecast caches for BTC and ETH."""
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from binanceneural.forecasts import ChronosForecastManager, ForecastConfig
from src.models.chronos2_wrapper import Chronos2OHLCWrapper

SYMBOLS = {
    "BTCUSD": "chronos2_finetuned/BTCUSD_lora_percent_change_ctx128_lr5e-5_r16/finetuned-ckpt",
    "ETHUSD": "chronos2_finetuned/ETHUSD_lora_percent_change_ctx128_lr5e-5_r16/finetuned-ckpt",
}
OHLC_DIR = Path("trainingdatahourly/crypto")
CACHE_ROOT = Path("cryptoalpacaexperiment/forecast_cache")
REFRESH_INTERVAL = 3600


def _refresh_ohlc():
    """Download fresh hourly bars from Alpaca crypto."""
    from alpaca.data import CryptoBarsRequest
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD

    client = CryptoHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    hourly = TimeFrame(1, TimeFrameUnit.Hour)
    now = datetime.now(timezone.utc)

    for symbol in SYMBOLS:
        csv_path = OHLC_DIR / f"{symbol}.csv"
        if not csv_path.exists():
            logger.warning("{}: no CSV, skipping", symbol)
            continue
        try:
            existing = pd.read_csv(csv_path, parse_dates=["timestamp"])
            if "timestamp" in existing.columns:
                existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            last_ts = existing["timestamp"].max()
            if pd.isna(last_ts):
                start = now - timedelta(days=30)
            else:
                start = last_ts.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(hours=1)
            if start >= now:
                logger.debug("{}: up to date", symbol)
                continue

            alp_sym = symbol[:-3] + "/" + symbol[-3:]
            req = CryptoBarsRequest(
                symbol_or_symbols=alp_sym, timeframe=hourly,
                start=start, end=now,
            )
            bars = client.get_crypto_bars(req).df
            if bars.empty:
                continue

            bars = bars.reset_index()
            if "symbol" in bars.columns:
                bars = bars.drop(columns=["symbol"])
            rename = {}
            for col in bars.columns:
                rename[col] = col.lower()
            bars = bars.rename(columns=rename)
            bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
            bars["symbol"] = symbol

            keep = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            bars = bars[[c for c in keep if c in bars.columns]]
            combined = pd.concat([existing, bars])
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            tmp = csv_path.with_suffix(".csv.tmp")
            combined.to_csv(tmp, index=False)
            tmp.replace(csv_path)
            logger.info("{}: +{} bars (total={}, last={})", symbol, len(bars), len(combined),
                       combined["timestamp"].max())
        except Exception as e:
            logger.error("{}: OHLC refresh failed: {}", symbol, e)


def _refresh_caches():
    """Rebuild Chronos2 forecast caches for all horizons."""
    for horizon in [1, 6]:
        for symbol, model_path in SYMBOLS.items():
            try:
                cache_dir = CACHE_ROOT / f"h{horizon}"
                wrapper = Chronos2OHLCWrapper.from_pretrained(
                    model_id=str(Path(model_path)),
                    device_map="cuda",
                    default_context_length=24 * 14,
                    default_batch_size=128,
                    quantile_levels=(0.1, 0.5, 0.9),
                )
                cfg = ForecastConfig(
                    symbol=symbol, data_root=OHLC_DIR,
                    context_hours=24 * 14, prediction_horizon_hours=horizon,
                    quantile_levels=(0.1, 0.5, 0.9), batch_size=128, cache_dir=cache_dir,
                )
                manager = ChronosForecastManager(cfg, wrapper_factory=lambda w=wrapper: w)
                manager.ensure_latest(cache_only=False)
                logger.info("h{} {} cache refreshed", horizon, symbol)
            except Exception as e:
                logger.error("h{} {} cache failed: {}", horizon, symbol, e)


def main():
    logger.info("Crypto cache refresh daemon starting")
    while True:
        _refresh_ohlc()
        _refresh_caches()
        logger.info("Refresh complete, sleeping {}s", REFRESH_INTERVAL)
        time.sleep(REFRESH_INTERVAL)


if __name__ == "__main__":
    main()
