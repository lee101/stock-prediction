"""
Position Sizing Dataset Collection Script

This script collects a comprehensive dataset for position sizing by running
the market simulator over rolling 7-day windows across a full year of data.

It handles:
- Multiple symbols (stocks, crypto, ETFs)
- Different trading calendars (252 days for stocks, 365 for crypto)
- Rolling window simulations (52 weeks of 7-day windows)
- Trade logs, positions, PnL aggregation
- Strategy performance metrics

The collected data can be used to train position sizing models.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import market simulator components
from marketsimulator.environment import activate_simulation
from marketsimulator.runner import simulate_strategy
from marketsimulator.state import SimulationState
from marketsimulator.data_feed import load_price_data


class TradingCalendar:
    """Handle different trading calendars for stocks vs crypto"""

    @staticmethod
    def is_crypto(symbol: str) -> bool:
        """Check if symbol is a cryptocurrency"""
        return '-USD' in symbol.upper()

    @staticmethod
    def get_trading_days_per_year(symbol: str) -> int:
        """Get number of trading days per year for symbol"""
        return 365 if TradingCalendar.is_crypto(symbol) else 252

    @staticmethod
    def get_hourly_bars_per_year(symbol: str) -> int:
        """Get approximate hourly bars per year"""
        if TradingCalendar.is_crypto(symbol):
            return 365 * 24  # 24/7 trading
        else:
            # Stock market: ~6.5 hours per day, 252 days
            return 252 * 7  # Approximate hourly bars


class DatasetCollector:
    """Collects position sizing dataset from market simulations"""

    def __init__(
        self,
        data_dir: str = "trainingdata/train",
        output_dir: str = "strategytraining/datasets",
        window_days: int = 7,
        stride_days: int = 7,
        min_data_points: int = 2000  # Minimum hourly bars needed
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.window_days = window_days
        self.stride_days = stride_days
        self.min_data_points = min_data_points

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for collected data
        self.trades_data = []
        self.window_summaries = []
        self.position_snapshots = []

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from data directory"""
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} not found")
            return []

        symbols = []
        for csv_file in self.data_dir.glob("*.csv"):
            symbol = csv_file.stem
            symbols.append(symbol)

        return sorted(symbols)

    def load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load historical data for a symbol"""
        csv_path = self.data_dir / f"{symbol}.csv"

        if not csv_path.exists():
            print(f"Warning: Data file not found for {symbol}")
            return None

        try:
            df = pd.read_csv(csv_path)

            # Ensure timestamp column exists
            if 'timestamp' not in df.columns:
                print(f"Warning: No timestamp column in {symbol}")
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Verify required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Missing required columns in {symbol}")
                return None

            return df

        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return None

    def calculate_rolling_windows(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[Tuple[int, int]]:
        """
        Calculate rolling window indices for the dataset

        Returns list of (start_idx, end_idx) tuples for each window
        """
        is_crypto = TradingCalendar.is_crypto(symbol)

        # Calculate window size in hours
        if is_crypto:
            # Crypto: 24/7 trading
            hours_per_day = 24
        else:
            # Stocks: approximate hourly bars (market hours only)
            hours_per_day = 7  # ~6.5 hours rounded up

        window_size = self.window_days * hours_per_day
        stride_size = self.stride_days * hours_per_day

        windows = []
        start_idx = 0

        while start_idx + window_size <= len(df):
            end_idx = start_idx + window_size
            windows.append((start_idx, end_idx))
            start_idx += stride_size

        return windows

    def simulate_window(
        self,
        symbol: str,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        window_num: int
    ) -> Dict:
        """
        Run market simulation for a single window

        Returns dictionary with trades, positions, and performance metrics
        """
        window_df = df.iloc[start_idx:end_idx].copy()

        if len(window_df) < 10:  # Minimum viable window
            return None

        # Prepare window metadata
        start_time = window_df['timestamp'].iloc[0]
        end_time = window_df['timestamp'].iloc[-1]

        try:
            with activate_simulation():
                # Run simulation
                # Note: simulate_strategy typically takes symbol and runs internally
                # We'll call it and collect results

                # For now, we'll create a simplified simulation
                # In production, this would call simulate_strategy() properly
                result = self._run_simplified_simulation(
                    symbol, window_df, start_time, end_time
                )

                if result is None:
                    return None

                # Add metadata
                result['window_metadata'] = {
                    'symbol': symbol,
                    'window_num': window_num,
                    'start_time': str(start_time),
                    'end_time': str(end_time),
                    'num_bars': len(window_df),
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'is_crypto': TradingCalendar.is_crypto(symbol)
                }

                return result

        except Exception as e:
            print(f"Error simulating {symbol} window {window_num}: {e}")
            return None

    def _run_simplified_simulation(
        self,
        symbol: str,
        df: pd.DataFrame,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> Optional[Dict]:
        """
        Run a simplified market simulation

        This collects the key metrics we need for position sizing:
        - Individual trades (entry, exit, PnL, duration)
        - Position sizes over time
        - Cumulative PnL
        - Drawdowns
        - Volatility metrics
        """

        # Initialize tracking
        trades = []
        positions = []
        equity_curve = []

        # Starting capital
        initial_capital = 100000.0
        current_capital = initial_capital
        current_position = 0.0
        position_entry_price = 0.0
        position_entry_idx = 0

        # Simple strategy: mean reversion based on recent moves
        # This is just for data collection - real strategies would be more sophisticated

        for idx, row in df.iterrows():
            price = row['Close']
            high = row['High']
            low = row['Low']
            timestamp = row['timestamp']

            # Calculate simple signal (example)
            if idx >= 20:
                recent_prices = df.loc[max(0, idx-20):idx, 'Close']
                mean_price = recent_prices.mean()
                std_price = recent_prices.std()

                z_score = (price - mean_price) / (std_price + 1e-8)

                # Generate trade signals
                if current_position == 0:
                    # Entry logic
                    if z_score < -1.5:  # Oversold
                        # Buy signal
                        position_size = 100  # shares
                        cost = price * position_size * 1.001  # Include fees

                        if cost < current_capital * 0.95:  # Leave buffer
                            current_position = position_size
                            position_entry_price = price
                            position_entry_idx = idx
                            current_capital -= cost

                            positions.append({
                                'timestamp': timestamp,
                                'action': 'open',
                                'position': current_position,
                                'price': price,
                                'capital': current_capital,
                                'equity': current_capital + current_position * price
                            })

                elif current_position > 0:
                    # Exit logic
                    exit_signal = False

                    # Take profit
                    if price > position_entry_price * 1.02:
                        exit_signal = True
                    # Stop loss
                    elif price < position_entry_price * 0.98:
                        exit_signal = True
                    # Mean reversion complete
                    elif z_score > 0.5:
                        exit_signal = True

                    if exit_signal:
                        # Close position
                        proceeds = price * current_position * 0.999  # Include fees
                        pnl = proceeds - (position_entry_price * current_position)
                        pnl_pct = pnl / (position_entry_price * current_position)
                        duration_bars = idx - position_entry_idx

                        current_capital += proceeds

                        trades.append({
                            'entry_timestamp': df.loc[position_entry_idx, 'timestamp'],
                            'exit_timestamp': timestamp,
                            'entry_price': position_entry_price,
                            'exit_price': price,
                            'position_size': current_position,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'duration_bars': duration_bars,
                            'max_drawdown_during': 0.0  # Could calculate
                        })

                        positions.append({
                            'timestamp': timestamp,
                            'action': 'close',
                            'position': 0,
                            'price': price,
                            'capital': current_capital,
                            'equity': current_capital,
                            'pnl': pnl
                        })

                        current_position = 0.0
                        position_entry_price = 0.0

            # Track equity
            equity = current_capital + current_position * price
            equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'position': current_position,
                'price': price
            })

        # Close any remaining position at end
        if current_position > 0:
            final_price = df['Close'].iloc[-1]
            proceeds = final_price * current_position * 0.999
            pnl = proceeds - (position_entry_price * current_position)
            current_capital += proceeds

            trades.append({
                'entry_timestamp': df.loc[position_entry_idx, 'timestamp'],
                'exit_timestamp': df['timestamp'].iloc[-1],
                'entry_price': position_entry_price,
                'exit_price': final_price,
                'position_size': current_position,
                'pnl': pnl,
                'pnl_pct': pnl / (position_entry_price * current_position),
                'duration_bars': len(df) - position_entry_idx,
                'max_drawdown_during': 0.0
            })

        # Calculate summary statistics
        final_equity = current_capital
        total_return = (final_equity - initial_capital) / initial_capital

        equity_series = pd.Series([e['equity'] for e in equity_curve])
        returns = equity_series.pct_change().dropna()

        sharpe = 0.0
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 6.5)  # Annualized

        max_drawdown = 0.0
        if len(equity_series) > 0:
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max
            max_drawdown = drawdown.min()

        return {
            'trades': trades,
            'positions': positions,
            'equity_curve': equity_curve,
            'summary': {
                'initial_capital': initial_capital,
                'final_capital': final_equity,
                'total_return': total_return,
                'num_trades': len(trades),
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': sum(1 for t in trades if t['pnl'] > 0) / max(len(trades), 1),
                'avg_pnl': np.mean([t['pnl'] for t in trades]) if trades else 0.0,
                'avg_duration': np.mean([t['duration_bars'] for t in trades]) if trades else 0.0
            }
        }

    def collect_symbol_data(self, symbol: str) -> Dict:
        """Collect all rolling window data for a single symbol"""

        print(f"\n{'='*80}")
        print(f"Processing {symbol}")
        print(f"{'='*80}")

        # Load data
        df = self.load_symbol_data(symbol)
        if df is None or len(df) < self.min_data_points:
            print(f"Skipping {symbol}: insufficient data")
            return None

        print(f"Loaded {len(df)} data points from {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Calculate windows
        windows = self.calculate_rolling_windows(df, symbol)
        print(f"Generated {len(windows)} rolling windows of {self.window_days} days each")

        if len(windows) == 0:
            print(f"Skipping {symbol}: no valid windows")
            return None

        # Simulate each window
        symbol_trades = []
        symbol_summaries = []
        symbol_positions = []

        for window_num, (start_idx, end_idx) in enumerate(tqdm(windows, desc=f"{symbol} windows")):
            result = self.simulate_window(symbol, df, start_idx, end_idx, window_num)

            if result is None:
                continue

            # Collect data
            metadata = result['window_metadata']

            # Add trades
            for trade in result['trades']:
                trade_record = {**trade, **metadata}
                symbol_trades.append(trade_record)

            # Add summary
            summary_record = {**result['summary'], **metadata}
            symbol_summaries.append(summary_record)

            # Store position snapshots (sample)
            for pos in result['positions']:
                pos_record = {**pos, **metadata}
                symbol_positions.append(pos_record)

        print(f"\nCollected {len(symbol_trades)} trades across {len(windows)} windows")

        # Aggregate to main storage
        self.trades_data.extend(symbol_trades)
        self.window_summaries.extend(symbol_summaries)
        self.position_snapshots.extend(symbol_positions)

        return {
            'symbol': symbol,
            'num_windows': len(windows),
            'num_trades': len(symbol_trades),
            'total_data_points': len(df)
        }

    def collect_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        max_symbols: Optional[int] = None
    ):
        """Collect data for all symbols"""

        if symbols is None:
            symbols = self.get_available_symbols()

        if max_symbols is not None:
            symbols = symbols[:max_symbols]

        print(f"\nCollecting position sizing dataset for {len(symbols)} symbols")
        print(f"Window size: {self.window_days} days")
        print(f"Stride: {self.stride_days} days")
        print(f"Output directory: {self.output_dir}")

        results = []

        for symbol in symbols:
            result = self.collect_symbol_data(symbol)
            if result:
                results.append(result)

        return results

    def save_dataset(self, dataset_name: str = "position_sizing_dataset"):
        """Save collected dataset to disk"""

        print(f"\n{'='*80}")
        print(f"Saving dataset: {dataset_name}")
        print(f"{'='*80}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = self.output_dir / f"{dataset_name}_{timestamp}"

        # Save trades
        trades_df = pd.DataFrame(self.trades_data)
        trades_path = f"{output_base}_trades.parquet"
        trades_df.to_parquet(trades_path, index=False)
        print(f"Saved {len(trades_df)} trades to {trades_path}")

        # Save window summaries
        summaries_df = pd.DataFrame(self.window_summaries)
        summaries_path = f"{output_base}_summaries.parquet"
        summaries_df.to_parquet(summaries_path, index=False)
        print(f"Saved {len(summaries_df)} window summaries to {summaries_path}")

        # Save position snapshots
        positions_df = pd.DataFrame(self.position_snapshots)
        positions_path = f"{output_base}_positions.parquet"
        positions_df.to_parquet(positions_path, index=False)
        print(f"Saved {len(positions_df)} position snapshots to {positions_path}")

        # Save metadata
        metadata = {
            'dataset_name': dataset_name,
            'created_at': timestamp,
            'window_days': self.window_days,
            'stride_days': self.stride_days,
            'num_trades': len(trades_df),
            'num_windows': len(summaries_df),
            'num_position_snapshots': len(positions_df),
            'symbols': summaries_df['symbol'].unique().tolist() if len(summaries_df) > 0 else []
        }

        metadata_path = f"{output_base}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

        print(f"\nDataset collection complete!")
        print(f"Total trades: {len(trades_df)}")
        print(f"Total windows: {len(summaries_df)}")
        print(f"Unique symbols: {len(metadata['symbols'])}")

        return {
            'trades_path': trades_path,
            'summaries_path': summaries_path,
            'positions_path': positions_path,
            'metadata_path': metadata_path
        }


def main():
    """Main entry point"""

    import argparse

    parser = argparse.ArgumentParser(description="Collect position sizing dataset")
    parser.add_argument('--data-dir', default='trainingdata/train',
                       help='Directory containing training data')
    parser.add_argument('--output-dir', default='strategytraining/datasets',
                       help='Output directory for dataset')
    parser.add_argument('--window-days', type=int, default=7,
                       help='Window size in days')
    parser.add_argument('--stride-days', type=int, default=7,
                       help='Stride between windows in days')
    parser.add_argument('--max-symbols', type=int, default=None,
                       help='Maximum number of symbols to process')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Specific symbols to process')
    parser.add_argument('--dataset-name', default='position_sizing_dataset',
                       help='Name for the output dataset')

    args = parser.parse_args()

    # Create collector
    collector = DatasetCollector(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_days=args.window_days,
        stride_days=args.stride_days
    )

    # Collect data
    results = collector.collect_all_symbols(
        symbols=args.symbols,
        max_symbols=args.max_symbols
    )

    # Save dataset
    if len(collector.trades_data) > 0:
        paths = collector.save_dataset(dataset_name=args.dataset_name)
        print(f"\nDataset saved successfully!")
        for key, path in paths.items():
            print(f"  {key}: {path}")
    else:
        print("\nNo data collected. Check your data directory and symbol list.")


if __name__ == '__main__':
    main()
