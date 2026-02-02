#!/usr/bin/env python3
"""
Download comprehensive hourly stock and crypto data from Alpaca.
Goes back as far as Alpaca allows (typically 5-7 years for stocks, longer for crypto).
"""
import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

from alpaca.data import (
    StockBarsRequest,
    StockHistoricalDataClient,
    CryptoBarsRequest,
    CryptoHistoricalDataClient,
    TimeFrame,
    TimeFrameUnit,
)
from alpaca.data.enums import DataFeed

from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from alpaca_wrapper import (
    DEFAULT_STOCK_SYMBOLS,
    DEFAULT_CRYPTO_SYMBOLS,
)
from src.stock_utils import remap_symbols

# Initialize clients
stock_client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
crypto_client = CryptoHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)

# Alpaca limits - stocks have shorter history for hourly data
DEFAULT_STOCK_HISTORY_YEARS = 7  # Alpaca typically provides ~7 years of hourly stock data
DEFAULT_CRYPTO_HISTORY_YEARS = 10  # Crypto often has more history available

def download_hourly_stock_data(
    symbol: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Download hourly stock bars from Alpaca."""
    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Hour),
            start=start,
            end=end,
            adjustment="raw",
            feed=DataFeed.IEX,
        )
        bars = stock_client.get_stock_bars(request).df

        if bars.empty:
            return pd.DataFrame()

        # Normalize the dataframe
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index(level='symbol', drop=True)

        bars = bars.sort_index()
        bars['symbol'] = symbol

        return bars

    except Exception as e:
        logger.error(f"Failed to download hourly stock data for {symbol}: {e}")
        return pd.DataFrame()


def download_hourly_crypto_data(
    symbol: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Download hourly crypto bars from Alpaca."""
    try:
        # Remap symbol to Alpaca format (e.g., BTCUSD -> BTC/USD)
        remapped = remap_symbols(symbol)
        request = CryptoBarsRequest(
            symbol_or_symbols=remapped,
            timeframe=TimeFrame(1, TimeFrameUnit.Hour),
            start=start,
            end=end,
        )
        bars = crypto_client.get_crypto_bars(request).df

        if bars.empty:
            return pd.DataFrame()

        # Normalize the dataframe
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index(level='symbol', drop=True)

        bars = bars.sort_index()
        bars['symbol'] = symbol

        return bars

    except Exception as e:
        logger.error(f"Failed to download hourly crypto data for {symbol}: {e}")
        return pd.DataFrame()


def download_and_save_symbol(
    symbol: str,
    is_crypto: bool,
    output_dir: Path,
    years_back: int,
    skip_if_exists: bool = True,
) -> dict:
    """Download hourly data for a single symbol and save to CSV."""

    # Determine output path
    asset_dir = output_dir / ("crypto" if is_crypto else "stocks")
    asset_dir.mkdir(parents=True, exist_ok=True)
    output_file = asset_dir / f"{symbol}.csv"

    # Check if already exists and optionally update in place
    existing_df: Optional[pd.DataFrame] = None
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
        except Exception as e:
            logger.warning(f"Error reading existing file for {symbol}: {e}")
            existing_df = None

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=years_back * 365)
    update_mode = False

    if existing_df is not None and not existing_df.empty:
        existing_df.index = pd.to_datetime(existing_df.index, utc=True, errors="coerce")
        existing_df = existing_df[~existing_df.index.isna()]
        latest_ts = pd.to_datetime(existing_df.index.max(), utc=True, errors="coerce") if not existing_df.empty else None
        if latest_ts is not None and not pd.isna(latest_ts):
            if skip_if_exists and latest_ts >= end_dt - timedelta(hours=2):
                logger.info(f"Skipping {symbol} - latest hourly bar already up to date ({latest_ts})")
                return {
                    "symbol": symbol,
                    "status": "skipped",
                    "bars": len(existing_df),
                    "start": str(existing_df.index.min()),
                    "end": str(existing_df.index.max()),
                }
            if skip_if_exists:
                update_mode = True
                start_dt = max(start_dt, latest_ts - timedelta(hours=2))

    logger.info(f"Downloading {symbol} hourly data from {start_dt.date()} to {end_dt.date()}")

    # Download data
    if is_crypto:
        df = download_hourly_crypto_data(symbol, start_dt, end_dt)
    else:
        df = download_hourly_stock_data(symbol, start_dt, end_dt)

    if df.empty:
        if update_mode and existing_df is not None and not existing_df.empty:
            logger.warning(f"No new hourly data retrieved for {symbol}; keeping existing file")
            return {
                "symbol": symbol,
                "status": "no_update",
                "bars": len(existing_df),
                "start": str(existing_df.index.min()),
                "end": str(existing_df.index.max()),
            }
        logger.warning(f"No hourly data retrieved for {symbol}")
        return {
            "symbol": symbol,
            "status": "no_data",
            "bars": 0,
            "error": "Empty dataframe returned",
        }

    if update_mode and existing_df is not None and not existing_df.empty:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]
        combined = pd.concat([existing_df, df], axis=0, sort=False)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        added = max(0, len(combined) - len(existing_df))
        combined.index.name = "timestamp"
        combined.to_csv(output_file)
        logger.success(
            f"Updated {symbol} with {added} new hourly bars (total {len(combined)}) -> {output_file}"
        )
        return {
            "symbol": symbol,
            "status": "updated",
            "bars": len(combined),
            "added_bars": added,
            "start": str(combined.index.min()),
            "end": str(combined.index.max()),
            "file": str(output_file),
        }

    # Save to CSV (full download)
    df.index.name = "timestamp"
    df.to_csv(output_file)

    logger.success(f"Saved {len(df)} hourly bars for {symbol} to {output_file}")

    return {
        "symbol": symbol,
        "status": "ok",
        "bars": len(df),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "file": str(output_file),
    }


def download_all_symbols(
    symbols: Optional[List[str]] = None,
    include_stocks: bool = True,
    include_crypto: bool = True,
    output_dir: Path = Path("trainingdatahourly"),
    stock_years: int = DEFAULT_STOCK_HISTORY_YEARS,
    crypto_years: int = DEFAULT_CRYPTO_HISTORY_YEARS,
    sleep_seconds: float = 0.5,
    skip_if_exists: bool = True,
):
    """Download hourly data for all specified symbols."""

    # Build symbol list
    all_symbols = []
    if symbols:
        all_symbols = [(s, s.endswith("USD")) for s in symbols]
    else:
        if include_stocks:
            all_symbols.extend([(s, False) for s in DEFAULT_STOCK_SYMBOLS])
        if include_crypto:
            all_symbols.extend([(s, True) for s in DEFAULT_CRYPTO_SYMBOLS])

    logger.info(f"Starting download of {len(all_symbols)} symbols to {output_dir}")

    results = []
    for i, (symbol, is_crypto) in enumerate(all_symbols, 1):
        logger.info(f"Processing {i}/{len(all_symbols)}: {symbol}")

        years_back = crypto_years if is_crypto else stock_years
        result = download_and_save_symbol(
            symbol=symbol,
            is_crypto=is_crypto,
            output_dir=output_dir,
            years_back=years_back,
            skip_if_exists=skip_if_exists,
        )
        results.append(result)

        # Rate limiting
        if i < len(all_symbols):
            time.sleep(sleep_seconds)

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_file = output_dir / "download_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Download Complete!")
    logger.info(f"{'='*60}")
    logger.success(f"Total symbols: {len(results)}")
    logger.success(f"Successful: {sum(1 for r in results if r['status'] in {'ok', 'updated'})}")
    logger.success(f"Skipped: {sum(1 for r in results if r['status'] in {'skipped', 'no_update'})}")
    logger.warning(f"No data: {sum(1 for r in results if r['status'] == 'no_data')}")
    logger.info(f"Summary saved to: {summary_file}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download hourly stock and crypto data from Alpaca"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Specific symbols to download (default: all stocks and crypto)",
    )
    parser.add_argument(
        "--only-stocks",
        action="store_true",
        help="Only download stock symbols",
    )
    parser.add_argument(
        "--only-crypto",
        action="store_true",
        help="Only download crypto symbols",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("trainingdatahourly"),
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--stock-years",
        type=int,
        default=DEFAULT_STOCK_HISTORY_YEARS,
        help=f"Years of history to download for stocks (default: {DEFAULT_STOCK_HISTORY_YEARS})",
    )
    parser.add_argument(
        "--crypto-years",
        type=int,
        default=DEFAULT_CRYPTO_HISTORY_YEARS,
        help=f"Years of history to download for crypto (default: {DEFAULT_CRYPTO_HISTORY_YEARS})",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between requests (default: 0.5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.only_stocks and args.only_crypto:
        logger.error("Cannot specify both --only-stocks and --only-crypto")
        sys.exit(1)

    include_stocks = not args.only_crypto
    include_crypto = not args.only_stocks

    download_all_symbols(
        symbols=args.symbols,
        include_stocks=include_stocks,
        include_crypto=include_crypto,
        output_dir=args.output_dir,
        stock_years=args.stock_years,
        crypto_years=args.crypto_years,
        sleep_seconds=args.sleep,
        skip_if_exists=not args.force,
    )


if __name__ == "__main__":
    main()
