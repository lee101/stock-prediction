"""
Enhanced Position Sizing Dataset Collection with Multiple Strategies

Collects PnL data for ALL strategies across ALL symbols over rolling windows.
This creates a comprehensive dataset showing which (symbol, strategy) combinations
work best over time - perfect for training position sizing algorithms.

Strategies tracked:
- simple_strategy: Predicted close price changes
- all_signals_strategy: Average of close/high/low predictions
- entry_takeprofit: Profit potential from high prices
- highlow: Range-based trading
- maxdiff: Maximum directional edge
- buy_hold: Buy and hold baseline
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from marketsimulator.environment import activate_simulation
from marketsimulator.backtest_test3_inline import backtest_forecasts


# Available strategies from backtest_test3_inline.py
STRATEGIES = [
    'simple_strategy',
    'all_signals_strategy',
    'entry_takeprofit',
    'highlow',
    'maxdiff',
    'buy_hold',
]


class StrategyPnLCollector:
    """Collects PnL data for all strategies across symbols and time windows"""

    def __init__(
        self,
        data_dir: str = "trainingdata/train",
        output_dir: str = "strategytraining/datasets",
        window_days: int = 7,
        stride_days: int = 7,
        min_data_points: int = 2000
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.window_days = window_days
        self.stride_days = stride_days
        self.min_data_points = min_data_points

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for collected data
        self.strategy_performance = []  # (symbol, strategy, window, pnl, metrics)
        self.window_details = []  # Detailed window-level data
        self.strategy_trades = []  # Individual trade-level data per strategy

    @staticmethod
    def is_crypto(symbol: str) -> bool:
        """Check if symbol is cryptocurrency"""
        return '-USD' in symbol.upper()

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols"""
        if not self.data_dir.exists():
            return []
        return sorted([f.stem for f in self.data_dir.glob("*.csv")])

    def load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load historical data for symbol"""
        csv_path = self.data_dir / f"{symbol}.csv"
        if not csv_path.exists():
            return None

        try:
            df = pd.read_csv(csv_path)

            rename_map = {}
            for col in df.columns:
                key = str(col).strip()
                lower = key.lower()
                if lower == 'timestamp':
                    rename_map[col] = 'timestamp'
                elif lower == 'open':
                    rename_map[col] = 'Open'
                elif lower == 'high':
                    rename_map[col] = 'High'
                elif lower == 'low':
                    rename_map[col] = 'Low'
                elif lower == 'close':
                    rename_map[col] = 'Close'
                elif lower == 'volume':
                    rename_map[col] = 'Volume'
            if rename_map:
                df = df.rename(columns=rename_map)
            if 'timestamp' not in df.columns:
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            required = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required):
                return None

            if 'Volume' not in df.columns:
                df['Volume'] = 0.0

            return df
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return None

    def calculate_rolling_windows(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[Tuple[int, int]]:
        """Calculate rolling window indices"""
        is_crypto = self.is_crypto(symbol)
        hours_per_day = 24 if is_crypto else 7

        window_size = self.window_days * hours_per_day
        stride_size = self.stride_days * hours_per_day

        windows = []
        start_idx = 0

        while start_idx + window_size <= len(df):
            end_idx = start_idx + window_size
            windows.append((start_idx, end_idx))
            start_idx += stride_size

        return windows

    def simulate_strategy_on_window(
        self,
        symbol: str,
        strategy_name: str,
        strategy_returns: pd.Series,
        window_df: pd.DataFrame,
        window_num: int
    ) -> Dict:
        """
        Simulate a single strategy on a window

        Args:
            symbol: Asset symbol
            strategy_name: Name of strategy
            strategy_returns: Series of strategy returns (from backtest_forecasts)
            window_df: Price data for this window
            window_num: Window number

        Returns:
            Dictionary with trades, positions, and summary metrics
        """
        try:
            # Initialize
            initial_capital = 100000.0
            capital = initial_capital
            position = 0.0
            position_entry_price = 0.0
            position_entry_idx = 0

            trades = []
            equity_curve = []

            # Simulate strategy
            for idx, row in window_df.iterrows():
                if idx >= len(strategy_returns):
                    break

                price = row['Close']
                timestamp = row['timestamp']
                signal = strategy_returns.iloc[idx]

                # Calculate current equity
                equity = capital + position * price
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': equity,
                    'position': position,
                    'price': price
                })

                # Trading logic based on signal
                if position == 0:
                    # Entry logic
                    if signal > 0.005:  # Positive signal threshold
                        # Buy
                        position_size = 100  # shares
                        cost = price * position_size * 1.001  # Include fees

                        if cost < capital * 0.95:
                            position = position_size
                            position_entry_price = price
                            position_entry_idx = idx
                            capital -= cost

                elif position > 0:
                    # Exit logic
                    exit_signal = False

                    # Take profit at 2%
                    if price > position_entry_price * 1.02:
                        exit_signal = True
                    # Stop loss at -2%
                    elif price < position_entry_price * 0.98:
                        exit_signal = True
                    # Exit on negative signal
                    elif signal < -0.005:
                        exit_signal = True
                    # Time-based exit after 5 bars
                    elif idx - position_entry_idx >= 5:
                        exit_signal = True

                    if exit_signal:
                        # Close position
                        proceeds = price * position * 0.999
                        pnl = proceeds - (position_entry_price * position)
                        pnl_pct = pnl / (position_entry_price * position)
                        duration = idx - position_entry_idx

                        capital += proceeds

                        trades.append({
                            'entry_timestamp': window_df.iloc[position_entry_idx]['timestamp'],
                            'exit_timestamp': timestamp,
                            'entry_price': position_entry_price,
                            'exit_price': price,
                            'position_size': position,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'duration_bars': duration,
                            'signal_at_entry': float(strategy_returns.iloc[position_entry_idx]),
                            'signal_at_exit': float(signal)
                        })

                        position = 0.0

            # Close any remaining position
            if position > 0:
                final_price = window_df['Close'].iloc[-1]
                proceeds = final_price * position * 0.999
                pnl = proceeds - (position_entry_price * position)
                pnl_pct = pnl / (position_entry_price * position)

                capital += proceeds

                trades.append({
                    'entry_timestamp': window_df.iloc[position_entry_idx]['timestamp'],
                    'exit_timestamp': window_df['timestamp'].iloc[-1],
                    'entry_price': position_entry_price,
                    'exit_price': final_price,
                    'position_size': position,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'duration_bars': len(window_df) - position_entry_idx,
                    'signal_at_entry': float(strategy_returns.iloc[position_entry_idx]),
                    'signal_at_exit': float(strategy_returns.iloc[-1])
                })

            # Calculate summary metrics
            final_equity = capital
            total_return = (final_equity - initial_capital) / initial_capital

            equity_series = pd.Series([e['equity'] for e in equity_curve])
            returns = equity_series.pct_change().dropna()

            sharpe = 0.0
            if len(returns) > 0 and returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252 * 6.5)

            max_drawdown = 0.0
            if len(equity_series) > 0:
                running_max = equity_series.expanding().max()
                drawdown = (equity_series - running_max) / running_max
                max_drawdown = drawdown.min()

            win_rate = sum(1 for t in trades if t['pnl'] > 0) / max(len(trades), 1)
            avg_pnl = np.mean([t['pnl'] for t in trades]) if trades else 0.0
            total_pnl = sum(t['pnl'] for t in trades)

            return {
                'symbol': symbol,
                'strategy': strategy_name,
                'window_num': window_num,
                'trades': trades,
                'equity_curve': equity_curve,
                'summary': {
                    'initial_capital': initial_capital,
                    'final_capital': final_equity,
                    'total_return': total_return,
                    'total_pnl': total_pnl,
                    'num_trades': len(trades),
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'avg_duration': np.mean([t['duration_bars'] for t in trades]) if trades else 0.0
                }
            }

        except Exception as e:
            print(f"Error simulating {strategy_name} on {symbol}: {e}")
            return None

    def process_window(
        self,
        symbol: str,
        df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        window_num: int
    ) -> List[Dict]:
        """
        Process one window: slice forecasts and simulate all strategies

        Args:
            symbol: Asset symbol
            df: Full price dataframe
            forecast_df: Full forecast dataframe (pre-computed)
            start_idx: Window start index
            end_idx: Window end index
            window_num: Window number

        Returns list of results, one per strategy
        """
        window_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        window_forecast_df = forecast_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

        if len(window_df) < 10:
            return []

        start_time = window_df['timestamp'].iloc[0]
        end_time = window_df['timestamp'].iloc[-1]

        try:
            # Simulate each strategy
            results = []

            for strategy in STRATEGIES:
                # Get strategy return column
                return_col = f"{strategy}_return"
                if return_col not in window_forecast_df.columns:
                    continue

                strategy_returns = window_forecast_df[return_col]

                # Simulate this strategy
                result = self.simulate_strategy_on_window(
                    symbol=symbol,
                    strategy_name=strategy,
                    strategy_returns=strategy_returns,
                    window_df=window_df,
                    window_num=window_num
                )

                if result:
                    # Add metadata
                    result['start_time'] = str(start_time)
                    result['end_time'] = str(end_time)
                    result['num_bars'] = len(window_df)
                    result['is_crypto'] = self.is_crypto(symbol)
                    results.append(result)

            return results

        except Exception as e:
            print(f"  Error processing window {window_num} for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def collect_symbol_data(self, symbol: str) -> Dict:
        """Collect all strategy performance data for one symbol"""

        print(f"\n{'='*80}")
        print(f"Processing {symbol}")
        print(f"{'='*80}")

        # Load data
        df = self.load_symbol_data(symbol)
        if df is None or len(df) < self.min_data_points:
            print(f"Skipping {symbol}: insufficient data")
            return None

        print(f"Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Calculate windows
        windows = self.calculate_rolling_windows(df, symbol)
        print(f"Generated {len(windows)} rolling windows")

        if len(windows) == 0:
            return None

        # Generate forecasts ONCE for entire symbol (optimization!)
        print(f"Generating forecasts for {symbol}...")
        try:
            data_root = self.data_dir
            if data_root.name.lower() == "train":
                data_root = data_root.parent
            with activate_simulation(symbols=[symbol], data_root=data_root):
                forecast_df = backtest_forecasts(symbol, num_simulations=len(df))

                if forecast_df is None or len(forecast_df) == 0:
                    print(f"  No forecasts generated for {symbol}, skipping")
                    return None

                print(f"  Forecasts generated: {len(forecast_df)} rows")
        except Exception as e:
            print(f"  Error generating forecasts for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Process each window
        total_results = 0

        for window_num, (start_idx, end_idx) in enumerate(tqdm(windows, desc=f"{symbol} windows")):
            results = self.process_window(symbol, df, forecast_df, start_idx, end_idx, window_num)

            for result in results:
                # Store strategy performance summary
                perf_record = {
                    'symbol': result['symbol'],
                    'strategy': result['strategy'],
                    'window_num': result['window_num'],
                    'start_time': result['start_time'],
                    'end_time': result['end_time'],
                    'is_crypto': result['is_crypto'],
                    **result['summary']
                }
                self.strategy_performance.append(perf_record)

                # Store individual trades
                for trade in result['trades']:
                    trade_record = {
                        'symbol': result['symbol'],
                        'strategy': result['strategy'],
                        'window_num': result['window_num'],
                        'is_crypto': result['is_crypto'],
                        **trade
                    }
                    self.strategy_trades.append(trade_record)

                total_results += 1

        print(f"Collected {total_results} strategy-window results ({total_results // len(STRATEGIES)} windows x {len(STRATEGIES)} strategies)")
        print(f"Total trades: {len([t for t in self.strategy_trades if t['symbol'] == symbol])}")

        return {
            'symbol': symbol,
            'num_windows': len(windows),
            'num_strategy_results': total_results,
            'num_trades': len([t for t in self.strategy_trades if t['symbol'] == symbol])
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

        print(f"\n{'='*80}")
        print(f"COLLECTING STRATEGY PNL DATASET")
        print(f"{'='*80}")
        print(f"Symbols: {len(symbols)}")
        print(f"Strategies: {len(STRATEGIES)}")
        print(f"Window: {self.window_days} days, Stride: {self.stride_days} days")
        print(f"Output: {self.output_dir}")
        print(f"Strategies: {', '.join(STRATEGIES)}")

        results = []
        for symbol in symbols:
            result = self.collect_symbol_data(symbol)
            if result:
                results.append(result)

        return results

    def save_dataset(self, dataset_name: str = "strategy_pnl_dataset"):
        """Save collected dataset"""

        print(f"\n{'='*80}")
        print(f"SAVING DATASET: {dataset_name}")
        print(f"{'='*80}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = self.output_dir / f"{dataset_name}_{timestamp}"

        # Save strategy performance (main dataset)
        perf_df = pd.DataFrame(self.strategy_performance)
        perf_path = f"{output_base}_strategy_performance.parquet"
        perf_df.to_parquet(perf_path, index=False)
        print(f"✓ Saved {len(perf_df)} strategy-window results: {perf_path}")

        # Save trades
        trades_df = pd.DataFrame(self.strategy_trades)
        trades_path = f"{output_base}_trades.parquet"
        trades_df.to_parquet(trades_path, index=False)
        print(f"✓ Saved {len(trades_df)} trades: {trades_path}")

        # Calculate and display summary statistics
        print(f"\n{'='*80}")
        print(f"DATASET SUMMARY")
        print(f"{'='*80}")

        unique_symbols = perf_df['symbol'].nunique()
        unique_strategies = perf_df['strategy'].nunique()
        unique_windows = perf_df.groupby('symbol')['window_num'].nunique().mean()

        print(f"\nOverall:")
        print(f"  Symbols: {unique_symbols}")
        print(f"  Strategies: {unique_strategies}")
        print(f"  Avg windows per symbol: {unique_windows:.1f}")
        print(f"  Total (symbol, strategy, window) records: {len(perf_df)}")

        print(f"\nPer Strategy Performance:")
        for strategy in STRATEGIES:
            strat_data = perf_df[perf_df['strategy'] == strategy]
            if len(strat_data) == 0:
                continue

            avg_return = strat_data['total_return'].mean()
            avg_sharpe = strat_data['sharpe_ratio'].mean()
            win_rate = strat_data['win_rate'].mean()
            total_pnl = strat_data['total_pnl'].sum()

            print(f"\n  {strategy}:")
            print(f"    Avg Return: {avg_return:.2%}")
            print(f"    Avg Sharpe: {avg_sharpe:.2f}")
            print(f"    Win Rate: {win_rate:.2%}")
            print(f"    Total PnL: ${total_pnl:,.2f}")

        # Save metadata
        metadata = {
            'dataset_name': dataset_name,
            'created_at': timestamp,
            'window_days': self.window_days,
            'stride_days': self.stride_days,
            'strategies': STRATEGIES,
            'num_symbols': int(unique_symbols),
            'num_strategies': int(unique_strategies),
            'num_strategy_windows': len(perf_df),
            'num_trades': len(trades_df),
            'symbols': perf_df['symbol'].unique().tolist()
        }

        metadata_path = f"{output_base}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\n✓ Saved metadata: {metadata_path}")

        return {
            'performance_path': perf_path,
            'trades_path': trades_path,
            'metadata_path': metadata_path,
            'base_path': str(output_base)
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect strategy PnL dataset for position sizing"
    )
    parser.add_argument('--data-dir', default='trainingdata/train')
    parser.add_argument('--output-dir', default='strategytraining/datasets')
    parser.add_argument('--window-days', type=int, default=7)
    parser.add_argument('--stride-days', type=int, default=7)
    parser.add_argument('--max-symbols', type=int, default=None)
    parser.add_argument('--symbols', nargs='+', default=None)
    parser.add_argument('--dataset-name', default='strategy_pnl_dataset')
    parser.add_argument('--min-data-points', type=int, default=2000,
                        help='Minimum number of bars required to process a symbol')

    args = parser.parse_args()

    collector = StrategyPnLCollector(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_days=args.window_days,
        stride_days=args.stride_days,
        min_data_points=args.min_data_points
    )

    results = collector.collect_all_symbols(
        symbols=args.symbols,
        max_symbols=args.max_symbols
    )

    if len(collector.strategy_performance) > 0:
        paths = collector.save_dataset(dataset_name=args.dataset_name)
        print(f"\n{'='*80}")
        print(f"SUCCESS! Dataset ready for position sizing training")
        print(f"{'='*80}")
        print(f"\nBase path: {paths['base_path']}")
    else:
        print("\nNo data collected")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
