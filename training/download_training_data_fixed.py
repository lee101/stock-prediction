#!/usr/bin/env python3
"""
Download diverse stock data for training
Uses the Alpaca API directly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import pandas as pd
import datetime
from loguru import logger
from typing import List, Dict
import json
import time
from alpaca.data import StockBarsRequest, TimeFrame, TimeFrameUnit
from alpaca.data.historical import StockHistoricalDataClient

from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD


# Define diverse stock symbols across different sectors
TRAINING_SYMBOLS = {
    # Tech giants - most liquid
    'tech_mega': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
    
    # Tech growth
    'tech_growth': ['CRM', 'ADBE', 'NFLX', 'PYPL', 'SQ', 'SHOP'],
    
    # Semiconductors
    'semiconductors': ['AMD', 'INTC', 'QCOM', 'AVGO', 'MU'],
    
    # Finance
    'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA'],
    
    # Healthcare
    'healthcare': ['JNJ', 'UNH', 'PFE', 'LLY', 'MRK'],
    
    # Consumer
    'consumer': ['WMT', 'HD', 'PG', 'KO', 'PEP', 'NKE', 'MCD', 'DIS'],
    
    # Energy
    'energy': ['XOM', 'CVX', 'COP'],
    
    # ETFs for broader market exposure  
    'etfs': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'],
}


def download_stock_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start: datetime.datetime,
    end: datetime.datetime
) -> pd.DataFrame:
    """Download stock bars for a single symbol"""
    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start,
            end=end,
            adjustment='raw'
        )
        
        bars = client.get_stock_bars(request)
        
        if bars and bars.df is not None and not bars.df.empty:
            df = bars.df
            
            # If multi-index with symbol, extract it
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level='symbol')
            
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error downloading {symbol}: {e}")
        return pd.DataFrame()


def download_all_training_data(
    output_dir: str = 'trainingdata',
    years_of_history: int = 3,
    sectors: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Download historical data for all training symbols
    
    Args:
        output_dir: Directory to save the data
        years_of_history: Number of years of historical data to download
        sectors: List of sectors to download, None for all
    
    Returns:
        Dictionary mapping symbol to dataframe
    """
    
    # Create output directory
    base_path = Path(__file__).parent.parent
    data_path = base_path / output_dir / 'stocks'
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Get all symbols to download
    if sectors is None:
        sectors = list(TRAINING_SYMBOLS.keys())
    
    all_symbols = []
    for sector in sectors:
        if sector in TRAINING_SYMBOLS:
            all_symbols.extend(TRAINING_SYMBOLS[sector])
    
    # Remove duplicates
    all_symbols = list(set(all_symbols))
    
    logger.info(f"Downloading data for {len(all_symbols)} symbols across {len(sectors)} sectors")
    logger.info(f"Sectors: {sectors}")
    
    # Initialize client
    client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
    
    # Track results
    results = {}
    failed_symbols = []
    
    # Calculate date range
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365 * years_of_history)
    
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Download data for each symbol
    for i, symbol in enumerate(all_symbols, 1):
        try:
            logger.info(f"[{i}/{len(all_symbols)}] Downloading {symbol}...")
            
            # Download data
            df = download_stock_bars(client, symbol, start_date, end_date)
            
            if df is not None and not df.empty:
                # Clean and prepare data
                df = df.copy()
                
                # Ensure columns are lowercase
                df.columns = [col.lower() for col in df.columns]
                
                # Add returns
                df['returns'] = df['close'].pct_change()
                
                # Add simple technical indicators
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean()
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
                
                # Add price change features
                df['high_low_ratio'] = df['high'] / df['low']
                df['close_open_ratio'] = df['close'] / df['open']
                
                # Save to CSV
                file_path = data_path / f"{symbol}_{end_date.strftime('%Y%m%d')}.csv"
                df.to_csv(file_path)
                
                results[symbol] = df
                logger.info(f"  ✓ Saved {len(df)} rows to {file_path}")
            else:
                logger.warning(f"  ⚠ No data received for {symbol}")
                failed_symbols.append(symbol)
            
            # Small delay to avoid rate limiting
            time.sleep(0.2)
            
        except Exception as e:
            logger.error(f"  ✗ Failed to download {symbol}: {e}")
            failed_symbols.append(symbol)
            continue
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Download Summary:")
    logger.info(f"  Successfully downloaded: {len(results)}/{len(all_symbols)} symbols")
    logger.info(f"  Total data points: {sum(len(df) for df in results.values()):,}")
    
    if failed_symbols:
        logger.warning(f"  Failed symbols ({len(failed_symbols)}): {failed_symbols}")
    
    # Save metadata
    metadata = {
        'download_date': datetime.datetime.now().isoformat(),
        'symbols': list(results.keys()),
        'failed_symbols': failed_symbols,
        'sectors': sectors,
        'years_of_history': years_of_history,
        'total_symbols': len(all_symbols),
        'successful_downloads': len(results),
        'data_points': {symbol: len(df) for symbol, df in results.items()}
    }
    
    metadata_path = data_path / 'download_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"  Metadata saved to {metadata_path}")
    
    return results


def create_combined_dataset(data_dir: str = 'trainingdata/stocks') -> pd.DataFrame:
    """
    Combine all downloaded stock data into a single training dataset
    """
    data_path = Path(__file__).parent.parent / data_dir
    
    if not data_path.exists():
        logger.error(f"Data directory {data_path} does not exist")
        return pd.DataFrame()
    
    # Find all CSV files
    csv_files = list(data_path.glob('*.csv'))
    csv_files = [f for f in csv_files if 'metadata' not in f.stem]
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    all_data = []
    
    for file in csv_files:
        # Extract symbol from filename
        symbol = file.stem.split('_')[0]
        
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            df['symbol'] = symbol
            all_data.append(df)
            logger.info(f"  Loaded {symbol}: {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to read {file}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=False)
        combined = combined.sort_index()
        
        logger.info(f"\nCombined dataset: {len(combined):,} rows, {combined['symbol'].nunique()} unique symbols")
        
        # Save combined dataset
        combined_path = data_path.parent / 'combined_training_data.csv'
        combined.to_csv(combined_path)
        logger.info(f"Saved combined dataset to {combined_path}")
        
        # Save as parquet for faster loading
        parquet_path = data_path.parent / 'combined_training_data.parquet'
        combined.to_parquet(parquet_path)
        logger.info(f"Saved parquet version to {parquet_path}")
        
        return combined
    else:
        logger.error("No data to combine")
        return pd.DataFrame()


def main():
    """Main function to download training data"""
    logger.info("="*80)
    logger.info("DOWNLOADING DIVERSE TRAINING DATA")
    logger.info("="*80)
    
    # Start with a smaller subset for testing
    test_sectors = ['tech_mega', 'etfs', 'finance']  # Start with most liquid stocks
    
    logger.info(f"Downloading data for sectors: {test_sectors}")
    
    results = download_all_training_data(
        output_dir='trainingdata',
        years_of_history=2,  # Start with 2 years
        sectors=test_sectors
    )
    
    if results:
        # Create combined dataset
        logger.info("\nCreating combined training dataset...")
        combined = create_combined_dataset()
        
        if not combined.empty:
            logger.info(f"\n✓ Successfully created training dataset with {len(combined):,} samples")
            logger.info(f"  Date range: {combined.index.min()} to {combined.index.max()}")
            logger.info(f"  Symbols: {combined['symbol'].nunique()}")
            
            # Show sample statistics
            logger.info("\nSample statistics:")
            for symbol in list(combined['symbol'].unique())[:5]:
                symbol_data = combined[combined['symbol'] == symbol]
                logger.info(f"  {symbol}: {len(symbol_data)} samples, "
                          f"price range ${symbol_data['close'].min():.2f} - ${symbol_data['close'].max():.2f}")
            
            # Show data quality
            logger.info("\nData quality:")
            logger.info(f"  Missing values: {combined.isnull().sum().sum()}")
            logger.info(f"  Columns: {list(combined.columns)}")
    else:
        logger.error("Failed to download any data")
    
    logger.info("\n" + "="*80)
    logger.info("DATA DOWNLOAD COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()