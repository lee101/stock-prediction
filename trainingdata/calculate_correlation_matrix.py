#!/usr/bin/env python3
"""
Calculate correlation matrix for all tradeable symbols.

This script computes rolling correlation matrices for portfolio risk management,
helping identify concentration risk from correlated positions.

Usage:
    python trainingdata/calculate_correlation_matrix.py [--lookback 60] [--output trainingdata/]

Output:
    - correlation_matrix.pkl (binary, fast loading)
    - correlation_matrix.json (human-readable)
    - correlation_matrix_YYYYMMDD.pkl (dated backup)
"""

import argparse
import json
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# Add parent directory to path to import from project
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpaca_wrapper import (
    DEFAULT_CRYPTO_SYMBOLS,
    DEFAULT_STOCK_SYMBOLS,
    DEFAULT_TRAINING_SYMBOLS,
    data_client,
)
from alpaca.data import StockBarsRequest, TimeFrame
from src.fixtures import crypto_symbols
from src.logging_utils import setup_logging

logger = setup_logging("calculate_correlation_matrix.log")


def fetch_historical_bars(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    timeframe: TimeFrame = TimeFrame.Day,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical price bars for multiple symbols.

    Args:
        symbols: List of trading symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        timeframe: Bar timeframe (default: daily)

    Returns:
        Dict mapping symbol to DataFrame with OHLCV data
    """
    logger.info(f"Fetching historical data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")

    # Split into batches to avoid API limits
    batch_size = 50
    all_data = {}

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        logger.debug(f"Fetching batch {i // batch_size + 1}: {len(batch)} symbols")

        try:
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
            )

            bars = data_client.get_stock_bars(request)

            # Convert to dict of DataFrames
            for symbol in batch:
                if symbol in bars.data:
                    df = bars.df[bars.df.index.get_level_values('symbol') == symbol]
                    if not df.empty:
                        # Reset index to get timestamp as column
                        df = df.reset_index(level='symbol', drop=True)
                        all_data[symbol] = df
                    else:
                        logger.warning(f"No data returned for {symbol}")
                else:
                    logger.warning(f"Symbol {symbol} not in response")

        except Exception as exc:
            logger.error(f"Failed to fetch batch {i // batch_size + 1}: {exc}")
            continue

    logger.info(f"Successfully fetched data for {len(all_data)}/{len(symbols)} symbols")
    return all_data


def calculate_returns(price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate daily returns from price data.

    Args:
        price_data: Dict mapping symbol to DataFrame with OHLCV data

    Returns:
        DataFrame with symbols as columns and daily returns as rows
    """
    logger.info("Calculating daily returns")

    returns_dict = {}
    for symbol, df in price_data.items():
        if 'close' not in df.columns:
            logger.warning(f"No 'close' column for {symbol}, skipping")
            continue

        # Calculate daily returns (pct_change)
        returns = df['close'].pct_change()

        # Drop first row (NaN) and infinite values
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna()

        if len(returns) > 0:
            returns_dict[symbol] = returns
        else:
            logger.warning(f"No valid returns for {symbol}")

    # Combine into single DataFrame
    returns_df = pd.DataFrame(returns_dict)

    logger.info(f"Calculated returns for {len(returns_df.columns)} symbols over {len(returns_df)} days")
    return returns_df


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Calculate correlation matrix with data quality metrics.

    Args:
        returns_df: DataFrame with returns (symbols as columns)

    Returns:
        Tuple of (correlation_matrix, data_quality_dict)
    """
    logger.info("Calculating correlation matrix")

    # Calculate pairwise correlations (using all available pairwise complete observations)
    correlation_matrix = returns_df.corr(method='pearson')

    # Calculate data quality metrics
    total_days = len(returns_df)
    data_quality = {}

    for symbol in returns_df.columns:
        valid_days = returns_df[symbol].notna().sum()
        missing_days = total_days - valid_days
        data_pct = (valid_days / total_days * 100) if total_days > 0 else 0.0

        data_quality[symbol] = {
            "valid_days": int(valid_days),
            "missing_days": int(missing_days),
            "data_pct": round(float(data_pct), 2),
        }

    logger.info(f"Correlation matrix shape: {correlation_matrix.shape}")

    # Log statistics
    corr_values = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
    logger.info(f"Correlation statistics: mean={np.mean(corr_values):.3f}, "
                f"median={np.median(corr_values):.3f}, "
                f"std={np.std(corr_values):.3f}")

    return correlation_matrix, data_quality


def cluster_by_correlation(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.7,
    method: str = 'average',
) -> Dict[str, Dict]:
    """
    Cluster symbols by correlation using hierarchical clustering.

    Args:
        correlation_matrix: Pairwise correlation matrix
        threshold: Correlation threshold for grouping (default: 0.7)
        method: Linkage method for hierarchical clustering

    Returns:
        Dict mapping cluster_id to cluster info (symbols, avg_correlation, etc.)
    """
    logger.info(f"Clustering symbols with threshold={threshold}")

    # Convert correlation to distance: distance = sqrt(2 * (1 - correlation))
    # This ensures distance is 0 when correlation is 1, and increases as correlation decreases
    distance_matrix = np.sqrt(2 * (1 - correlation_matrix.values))

    # Replace any NaN values with max distance
    distance_matrix = np.nan_to_num(distance_matrix, nan=np.sqrt(2))

    # Convert to condensed distance matrix for linkage
    # Extract upper triangle (excluding diagonal)
    condensed_distance = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance, method=method)

    # Convert threshold to distance threshold
    # If correlation threshold is 0.7, distance threshold is sqrt(2 * (1 - 0.7)) = sqrt(0.6) ≈ 0.775
    distance_threshold = np.sqrt(2 * (1 - threshold))

    # Get cluster assignments
    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')

    # Build cluster dict
    clusters = {}
    symbols = correlation_matrix.columns.tolist()

    for cluster_id in np.unique(cluster_labels):
        cluster_symbols = [symbols[i] for i, label in enumerate(cluster_labels) if label == cluster_id]

        if len(cluster_symbols) == 1:
            # Singleton cluster, skip or mark as uncorrelated
            continue

        # Calculate average pairwise correlation within cluster
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_corr_values = []

        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                idx_i, idx_j = cluster_indices[i], cluster_indices[j]
                corr_val = correlation_matrix.iloc[idx_i, idx_j]
                if not np.isnan(corr_val):
                    cluster_corr_values.append(corr_val)

        avg_correlation = np.mean(cluster_corr_values) if cluster_corr_values else 0.0

        # Auto-label based on composition
        label = _generate_cluster_label(cluster_symbols)

        clusters[f"cluster_{cluster_id}"] = {
            "symbols": cluster_symbols,
            "size": len(cluster_symbols),
            "avg_correlation": round(float(avg_correlation), 3),
            "label": label,
        }

    logger.info(f"Identified {len(clusters)} correlation clusters")

    return clusters


def _generate_cluster_label(symbols: List[str]) -> str:
    """Generate a human-readable label for a cluster based on its symbols."""

    # Common patterns
    if all(sym in crypto_symbols for sym in symbols):
        return "Crypto Assets"

    tech_stocks = {'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'AMZN', 'NVDA', 'TSLA', 'AMD', 'INTC'}
    if len(set(symbols) & tech_stocks) / len(symbols) > 0.5:
        return "Tech Mega-Cap"

    finance_stocks = {'JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'BLK', 'SCHW'}
    if len(set(symbols) & finance_stocks) / len(symbols) > 0.5:
        return "Financials"

    energy_stocks = {'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'HAL'}
    if len(set(symbols) & energy_stocks) / len(symbols) > 0.5:
        return "Energy"

    # Default label
    if len(symbols) <= 3:
        return f"Small Group ({', '.join(symbols[:3])})"
    else:
        return f"Mixed Group ({len(symbols)} symbols)"


def save_correlation_data(
    correlation_matrix: pd.DataFrame,
    data_quality: Dict[str, Dict],
    clusters: Dict[str, Dict],
    output_dir: Path,
    lookback_days: int,
) -> None:
    """
    Save correlation matrix and metadata to disk in multiple formats.

    Args:
        correlation_matrix: Correlation matrix DataFrame
        data_quality: Data quality metrics per symbol
        clusters: Cluster assignments and info
        output_dir: Output directory path
        lookback_days: Lookback period in days
    """
    timestamp = datetime.now(timezone.utc)
    date_str = timestamp.strftime("%Y%m%d")

    # Prepare data structure
    data = {
        "timestamp": timestamp.isoformat(),
        "lookback_days": lookback_days,
        "symbols": correlation_matrix.columns.tolist(),
        "correlation_matrix": correlation_matrix.values.tolist(),
        "data_quality": data_quality,
        "clusters": clusters,
        "metadata": {
            "num_symbols": len(correlation_matrix.columns),
            "num_clusters": len(clusters),
            "avg_correlation": float(np.mean(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)])),
        }
    }

    # Save as pickle (fast loading for production)
    pkl_path = output_dir / "correlation_matrix.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved correlation matrix to {pkl_path}")

    # Save as JSON (human-readable)
    json_path = output_dir / "correlation_matrix.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved correlation matrix to {json_path}")

    # Save dated backup
    dated_pkl_path = output_dir / f"correlation_matrix_{date_str}.pkl"
    with open(dated_pkl_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved dated backup to {dated_pkl_path}")

    # Save CSV for easy inspection
    csv_path = output_dir / "correlation_matrix.csv"
    correlation_matrix.to_csv(csv_path)
    logger.info(f"Saved correlation matrix CSV to {csv_path}")


def load_correlation_matrix(input_path: Path) -> Dict:
    """
    Load correlation matrix from pickle file.

    Args:
        input_path: Path to pickle file

    Returns:
        Dict with correlation data
    """
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description="Calculate correlation matrix for trading symbols")
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Lookback period in days (default: 60)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trainingdata",
        help="Output directory (default: trainingdata/)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Correlation threshold for clustering (default: 0.7)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="all",
        choices=["all", "stocks", "crypto"],
        help="Which symbols to include (default: all)"
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select symbols
    if args.symbols == "stocks":
        symbols = DEFAULT_STOCK_SYMBOLS
    elif args.symbols == "crypto":
        symbols = DEFAULT_CRYPTO_SYMBOLS
    else:
        symbols = DEFAULT_TRAINING_SYMBOLS

    logger.info(f"Starting correlation calculation for {len(symbols)} symbols")
    logger.info(f"Lookback: {args.lookback} days, Threshold: {args.threshold}")

    # Calculate date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.lookback)

    # Fetch data
    price_data = fetch_historical_bars(symbols, start_date, end_date)

    if len(price_data) == 0:
        logger.error("No price data fetched, exiting")
        sys.exit(1)

    # Calculate returns
    returns_df = calculate_returns(price_data)

    if returns_df.empty:
        logger.error("No valid returns calculated, exiting")
        sys.exit(1)

    # Calculate correlation matrix
    correlation_matrix, data_quality = calculate_correlation_matrix(returns_df)

    # Cluster symbols
    clusters = cluster_by_correlation(correlation_matrix, threshold=args.threshold)

    # Save results
    save_correlation_data(
        correlation_matrix,
        data_quality,
        clusters,
        output_dir,
        args.lookback,
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CORRELATION MATRIX SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Symbols analyzed: {len(correlation_matrix.columns)}")
    logger.info(f"Clusters identified: {len(clusters)}")
    logger.info(f"Average correlation: {np.mean(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]):.3f}")
    logger.info(f"\nTop clusters:")

    # Sort clusters by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]['size'], reverse=True)
    for i, (cluster_id, cluster_info) in enumerate(sorted_clusters[:5]):
        logger.info(f"  {i+1}. {cluster_info['label']}: {cluster_info['size']} symbols, "
                   f"avg corr={cluster_info['avg_correlation']:.3f}")
        logger.info(f"     Symbols: {', '.join(cluster_info['symbols'][:10])}"
                   f"{'...' if len(cluster_info['symbols']) > 10 else ''}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Output saved to {output_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
