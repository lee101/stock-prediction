#!/usr/bin/env python3
"""
Run HuggingFace Trading System
Main script for live trading and backtesting
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
# Make yfinance optional; provide a stub so tests can patch yf.download
try:
    import yfinance as yf  # Optional; may be unavailable in restricted envs
except Exception:
    class _YFStub:
        @staticmethod
        def download(*args, **kwargs):
            raise RuntimeError("yfinance unavailable and no local data provided")

        class Ticker:
            def __init__(self, *args, **kwargs):
                pass

            def history(self, *args, **kwargs):
                raise RuntimeError("yfinance unavailable and no local data provided")

    yf = _YFStub
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))

from hfinference.hf_trading_engine import HFTradingEngine, TradingSignal


class HFTrader:
    """High-level trader using HF models"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 config_path: str = None,
                 mode: str = 'backtest'):
        """
        Initialize trader
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file
            mode: 'live', 'paper', or 'backtest'
        """
        
        self.mode = mode
        self.engine = HFTradingEngine(checkpoint_path, config_path)
        self.setup_logging()
        
        # Trading parameters
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']  # Default symbols
        self.update_interval = 60  # seconds for live trading
        
        self.logger.info(f"HF Trader initialized in {mode} mode")
    
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger('HFTrader')
        
    def run_backtest(self, 
                     symbols: list = None,
                     start_date: str = None,
                     end_date: str = None,
                     initial_capital: float = 10000):
        """Run backtest"""
        
        if symbols:
            self.symbols = symbols
        
        # Default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        self.logger.info(f"Running backtest from {start_date} to {end_date}")
        self.logger.info(f"Symbols: {self.symbols}")
        self.logger.info(f"Initial capital: ${initial_capital:,.2f}")
        
        # Set initial capital
        self.engine.current_capital = initial_capital
        
        # Run backtest
        results = self.engine.run_backtest(self.symbols, start_date, end_date)
        
        # Print results
        self.print_backtest_results(results)
        
        # Save results
        self.save_results(results, 'backtest')
        
        return results
    
    def run_live_trading(self):
        """Run live trading (paper or real)"""
        
        self.logger.info(f"Starting {self.mode} trading")
        self.logger.info(f"Symbols: {self.symbols}")
        self.logger.info(f"Update interval: {self.update_interval} seconds")
        
        try:
            while True:
                self.trading_loop()
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Trading stopped by user")
            self.close_positions()
    
    def trading_loop(self):
        """Single iteration of trading loop"""
        
        for symbol in self.symbols:
            try:
                # Get recent data
                data = self.get_recent_data(symbol)
                
                if data is None or len(data) < getattr(self.engine.data_processor, 'sequence_length', 60):
                    continue
                
                # Generate signal
                signal = self.engine.generate_signal(symbol, data)
                
                if signal:
                    self.logger.info(f"Signal for {symbol}: {signal.action} "
                                   f"(confidence: {signal.confidence:.2f})")
                    
                    # Execute trade in paper/live mode
                    if self.mode == 'paper':
                        self.execute_paper_trade(signal)
                    elif self.mode == 'live':
                        self.execute_live_trade(signal)
                
                # Check existing positions
                self.check_positions(symbol, data['Close'].iloc[-1])
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
    
    def get_recent_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Get recent market data"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Try local CSVs first
        from pathlib import Path as _P
        def _load_local(symbol: str):
            for base in [_P('trainingdata'), _P('hftraining/trainingdata'), _P('externaldata/yahoo')]:
                for name in [f"{symbol}.csv", f"{symbol.upper()}.csv", f"{symbol.lower()}.csv"]:
                    p = base / name
                    if p.exists():
                        try:
                            df = pd.read_csv(p)
                            df.columns = [c.lower() for c in df.columns]
                            if 'date' in df.columns:
                                df['date'] = pd.to_datetime(df['date'])
                                df = df.sort_values('date').set_index('date')
                            # Clip to requested window if date index exists
                            if isinstance(df.index, pd.DatetimeIndex):
                                return df[df.index >= start_date][:days+5]
                            return df.tail(days)
                        except Exception:
                            continue
            return None

        data = _load_local(symbol)
        if data is not None and len(data) > 0:
            return data

        # Fallback to yfinance if available
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                interval='1d'
            )
            return data
        except Exception as e:
            self.logger.error(f"Failed to get data for {symbol}: {e}")
            return None
    
    def execute_paper_trade(self, signal: TradingSignal):
        """Execute paper trade"""
        
        result = self.engine.execute_trade(signal)
        
        if result['status'] == 'executed':
            self.logger.info(f"PAPER TRADE: {signal.action.upper()} {signal.symbol} "
                           f"@ ${signal.current_price:.2f}")
            
            # Log to file
            self.log_trade(signal, result, 'paper')
    
    def execute_live_trade(self, signal: TradingSignal):
        """Execute live trade (placeholder - needs broker integration)"""
        
        self.logger.warning("Live trading not implemented - use paper mode")
        # In production, integrate with broker API here
        # Example: self.broker.place_order(signal)
    
    def check_positions(self, symbol: str, current_price: float):
        """Check and update existing positions"""
        
        self.engine.update_positions(symbol, current_price)
    
    def close_positions(self):
        """Close all open positions"""
        
        self.logger.info("Closing all positions...")
        
        for symbol in list(self.engine.positions.keys()):
            # Get current price
            data = self.get_recent_data(symbol, days=5)
            if data is not None and len(data) > 0:
                current_price = data['Close'].iloc[-1]
                
                # Create sell signal
                signal = TradingSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action='sell',
                    confidence=1.0,
                    predicted_price=current_price,
                    current_price=current_price,
                    expected_return=0,
                    position_size=0
                )
                
                self.engine.execute_trade(signal)
    
    def print_backtest_results(self, results: dict):
        """Print backtest results"""
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        metrics = results.get('metrics', {})
        
        print(f"\nPerformance Metrics:")
        print(f"  Total Return: {metrics.get('total_return', 0)*100:.2f}%")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"  Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
        print(f"  Number of Trades: {metrics.get('num_trades', 0)}")
        
        # Print trade summary
        trades = results.get('trades', [])
        if trades:
            executed_trades = [t for t in trades if t['status'] == 'executed']
            
            print(f"\nTrade Summary:")
            print(f"  Total Trades: {len(executed_trades)}")
            
            buy_trades = [t for t in executed_trades if t.get('action') == 'buy']
            sell_trades = [t for t in executed_trades if t.get('action') == 'sell']
            
            print(f"  Buy Orders: {len(buy_trades)}")
            print(f"  Sell Orders: {len(sell_trades)}")
            
            # Calculate profit
            total_profit = sum(t.get('profit', 0) for t in executed_trades)
            print(f"  Total Profit: ${total_profit:.2f}")
        
        # Final portfolio value
        if results.get('equity_curve'):
            initial_value = results['equity_curve'][0]['value']
            final_value = results['equity_curve'][-1]['value']
            
            print(f"\nPortfolio Value:")
            print(f"  Initial: ${initial_value:,.2f}")
            print(f"  Final: ${final_value:,.2f}")
            print(f"  Change: ${final_value - initial_value:,.2f} "
                  f"({(final_value/initial_value - 1)*100:.2f}%)")
    
    def save_results(self, results: dict, prefix: str = 'results'):
        """Save results to file"""
        
        # Create results directory
        results_dir = Path('hfinference/results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = results_dir / f'{prefix}_{timestamp}.json'
        
        # Convert to serializable format
        serializable_results = {
            'metrics': results.get('metrics', {}),
            'num_trades': len(results.get('trades', [])),
            'final_equity': results['equity_curve'][-1]['value'] if results.get('equity_curve') else 0
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {filename}")
        
        # Save detailed trades
        if results.get('trades'):
            trades_df = pd.DataFrame(results['trades'])
            trades_file = results_dir / f'{prefix}_trades_{timestamp}.csv'
            trades_df.to_csv(trades_file, index=False)
            self.logger.info(f"Trades saved to {trades_file}")
    
    def log_trade(self, signal: TradingSignal, result: dict, mode: str):
        """Log trade to file"""
        
        log_dir = Path('hfinference/logs/trades')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f'{mode}_trades_{datetime.now().strftime("%Y%m%d")}.jsonl'
        
        trade_log = {
            **signal.to_dict(),
            'result': result,
            'mode': mode
        }
        
        with open(log_file, 'a') as f:
            json.dump(trade_log, f)
            f.write('\n')


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='HuggingFace Trading System')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    
    parser.add_argument('--mode', type=str, default='backtest',
                       choices=['backtest', 'paper', 'live'],
                       help='Trading mode')
    
    parser.add_argument('--symbols', type=str, nargs='+',
                       default=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
                       help='Stock symbols to trade')
    
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--capital', type=float, default=10000,
                       help='Initial capital')
    
    parser.add_argument('--update-interval', type=int, default=60,
                       help='Update interval in seconds for live trading')
    
    args = parser.parse_args()
    
    # Initialize trader
    trader = HFTrader(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        mode=args.mode
    )
    
    # Set parameters
    trader.symbols = args.symbols
    trader.update_interval = args.update_interval
    
    # Run appropriate mode
    if args.mode == 'backtest':
        trader.run_backtest(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.capital
        )
    else:
        trader.run_live_trading()


if __name__ == "__main__":
    main()
