#!/usr/bin/env python3
"""
Download diverse stock data for training
Uses the existing alpaca data download functionality
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

from data_curate_daily import download_daily_stock_data, download_exchange_historical_data
from alpaca.data.historical import StockHistoricalDataClient  
from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD


# Define diverse stock symbols across different sectors
TRAINING_SYMBOLS = {
    # Tech giants
    'tech_mega': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
    
    # Tech growth
    'tech_growth': ['CRM', 'ADBE', 'NFLX', 'PYPL', 'SQ', 'SHOP', 'SNOW', 'PLTR', 'MSFT'],
    
    # Semiconductors
    'semiconductors': ['AMD', 'INTC', 'QCOM', 'AVGO', 'MU', 'MRVL', 'AMAT', 'LRCX'],
    
    # Finance
    'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'SCHW'],
    
    # Healthcare
    'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'CVS', 'LLY', 'MRK', 'DHR'],
    
    # Consumer
    'consumer': ['WMT', 'HD', 'PG', 'KO', 'PEP', 'NKE', 'MCD', 'DIS', 'SBUX', 'COST'],
    
    # Energy
    'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO'],
    
    # Industrial
    'industrial': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'DE', 'LMT'],
    
    # ETFs for broader market exposure
    'etfs': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'EFA', 'EEM', 'GLD', 'TLT'],
    
    # Crypto (if available)
    'crypto': ['BTCUSD', 'ETHUSD'],
    
    # High volatility stocks for learning extreme patterns
    'volatile': ['GME', 'AMC', 'BBBY', 'SOFI', 'RIVN', 'LCID', 'SPCE'],
}


def download_all_training_data(
    output_dir: str = 'trainingdata',
    years_of_history: int = 4,
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
    
    # Download data for each symbol
    for i, symbol in enumerate(all_symbols, 1):
        try:
            logger.info(f"[{i}/{len(all_symbols)}] Downloading {symbol}...")
            
            # Calculate date range
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=365 * years_of_history)
            
            # Download using existing function
            df = download_exchange_historical_data(client, symbol)
            
            if df is not None and not df.empty:
                # Clean and prepare data
                df = df.copy()
                
                # Ensure we have the columns we need
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_cols):
                    # Add returns
                    df['returns'] = df['close'].pct_change()
                    
                    # Add technical indicators
                    df['sma_20'] = df['close'].rolling(window=20).mean()
                    df['sma_50'] = df['close'].rolling(window=50).mean()
                    df['rsi'] = calculate_rsi(df['close'])
                    
                    # Save to CSV
                    file_path = data_path / f"{symbol}_{end_date.strftime('%Y%m%d')}.csv"
                    df.to_csv(file_path)
                    
                    results[symbol] = df
                    logger.info(f"  ✓ Saved {len(df)} rows to {file_path}")
                else:
                    logger.warning(f"  ⚠ Missing required columns for {symbol}")
                    failed_symbols.append(symbol)
            else:
                logger.warning(f"  ⚠ No data received for {symbol}")
                failed_symbols.append(symbol)
                
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


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


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
    logger.info(f"Found {len(csv_files)} CSV files")
    
    all_data = []
    
    for file in csv_files:
        if 'metadata' in file.stem:
            continue
            
        # Extract symbol from filename
        symbol = file.stem.split('_')[0]
        
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            df['symbol'] = symbol
            all_data.append(df)
        except Exception as e:
            logger.error(f"Failed to read {file}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=False)
        combined = combined.sort_index()
        
        logger.info(f"Combined dataset: {len(combined):,} rows, {combined['symbol'].nunique()} unique symbols")
        
        # Save combined dataset
        combined_path = data_path.parent / 'combined_training_data.csv'
        combined.to_csv(combined_path)
        logger.info(f"Saved combined dataset to {combined_path}")
        
        return combined
    else:
        logger.error("No data to combine")
        return pd.DataFrame()


def main():
    """Main function to download training data"""
    logger.info("="*80)
    logger.info("DOWNLOADING DIVERSE TRAINING DATA")
    logger.info("="*80)
    
    # Download data for specific sectors (or all if None)
    # Start with a smaller subset for testing
    test_sectors = ['tech_mega', 'tech_growth', 'etfs']  # Start with these
    
    logger.info(f"Downloading data for sectors: {test_sectors}")
    
    results = download_all_training_data(
        output_dir='trainingdata',
        years_of_history=3,  # 3 years of data
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
            for symbol in combined['symbol'].unique()[:5]:
                symbol_data = combined[combined['symbol'] == symbol]
                logger.info(f"  {symbol}: {len(symbol_data)} samples, "
                          f"price range ${symbol_data['close'].min():.2f} - ${symbol_data['close'].max():.2f}")
    else:
        logger.error("Failed to download any data")
    
    logger.info("\n" + "="*80)
    logger.info("DATA DOWNLOAD COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
