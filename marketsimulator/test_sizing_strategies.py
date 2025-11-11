"""
Test different position sizing strategies on historical data.

This script:
1. Loads historical price data
2. Generates predictions using compiled models (toto/kronos)
3. Simulates trading with different sizing strategies
4. Accounts for leverage costs (6.75% annual on leverage >1x)
5. Outputs comparative performance metrics

Usage:
    # Test single symbol with all strategies
    MARKETSIM_FAST_SIMULATE=1 python marketsimulator/test_sizing_strategies.py --symbol BTCUSD

    # Test multiple symbols
    MARKETSIM_FAST_SIMULATE=1 python marketsimulator/test_sizing_strategies.py --symbols BTCUSD ETHUSD AAPL

    # Run with specific strategies
    python marketsimulator/test_sizing_strategies.py --symbol BTCUSD --strategies kelly_25 optimal_f fixed_50
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional
import time

import numpy as np
import pandas as pd

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from marketsimulator.sizing_strategies import (
    SIZING_STRATEGIES,
    MarketContext,
    SizingResult,
    SizingStrategy,
)
from src.leverage_settings import get_leverage_settings


@dataclass
class PositionState:
    """Track current position state."""
    symbol: str
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity == 0:
            return 0.0
        return (self.current_price - self.entry_price) * self.quantity


@dataclass
class PortfolioState:
    """Track portfolio state over time."""
    cash: float
    equity: float
    positions: Dict[str, PositionState]
    leverage_used: float
    timestamp: Optional[pd.Timestamp] = None

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        return sum(p.market_value for p in self.positions.values())


@dataclass
class SimulationResult:
    """Results from a single sizing strategy simulation."""
    strategy_name: str
    symbol: str
    initial_equity: float
    final_equity: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    avg_leverage: float
    total_leverage_cost: float
    win_rate: float
    equity_curve: List[float]
    timestamps: List[pd.Timestamp]
    trades: List[Dict]


def calculate_leverage_cost(
    gross_exposure: float,
    equity: float,
    daily_cost_rate: float
) -> float:
    """
    Calculate daily leverage cost.

    Cost applies to the leveraged portion only (exposure > equity).
    """
    if gross_exposure <= equity:
        return 0.0

    leveraged_amount = gross_exposure - equity
    return leveraged_amount * daily_cost_rate


def simulate_strategy(
    strategy: SizingStrategy,
    symbol: str,
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
    initial_equity: float = 100000.0,
    is_crypto: bool = False,
) -> SimulationResult:
    """
    Simulate trading with a given sizing strategy.

    Args:
        strategy: Sizing strategy to test
        symbol: Trading symbol
        prices: DataFrame with 'close', 'high', 'low', 'volume' columns
        predictions: DataFrame with 'predicted_return', 'predicted_volatility' columns
        initial_equity: Starting capital
        is_crypto: Whether this is a crypto asset (affects leverage/shorting)

    Returns:
        SimulationResult with performance metrics
    """
    leverage_settings = get_leverage_settings()
    daily_cost = leverage_settings.daily_cost

    # Align predictions with prices
    common_index = prices.index.intersection(predictions.index)
    if len(common_index) == 0:
        raise ValueError("No overlapping timestamps between prices and predictions")

    prices = prices.loc[common_index]
    predictions = predictions.loc[common_index]

    # Initialize state
    cash = initial_equity
    equity = initial_equity
    position = PositionState(symbol=symbol)

    equity_curve = [equity]
    timestamps = [prices.index[0]]
    trades = []
    leverage_values = []
    total_leverage_cost = 0.0

    for i, (timestamp, price_row) in enumerate(prices.iterrows()):
        current_price = price_row['close']

        if i >= len(predictions):
            break

        pred = predictions.iloc[i]
        predicted_return = pred['predicted_return']
        predicted_volatility = pred['predicted_volatility']

        # Update current position value
        position.current_price = current_price

        # Calculate current equity (cash + position value)
        equity = cash + position.market_value

        # Apply leverage cost from previous day
        gross_exposure = abs(position.market_value)
        if gross_exposure > equity:
            leverage_cost = calculate_leverage_cost(gross_exposure, equity, daily_cost)
            cash -= leverage_cost
            equity -= leverage_cost
            total_leverage_cost += leverage_cost

        # Calculate desired position using strategy
        ctx = MarketContext(
            symbol=symbol,
            predicted_return=predicted_return,
            predicted_volatility=predicted_volatility,
            current_price=current_price,
            equity=equity,
            is_crypto=is_crypto,
            existing_position_value=position.market_value,
        )

        sizing_result = strategy.calculate_size(ctx)
        target_qty = sizing_result.quantity

        # Round for crypto (3 decimals) or stocks (whole shares)
        if is_crypto:
            target_qty = np.floor(target_qty * 1000) / 1000
        else:
            target_qty = np.floor(target_qty)

        # Execute trade if position changes
        qty_delta = target_qty - position.quantity

        if abs(qty_delta) > (0.001 if is_crypto else 0.5):  # Minimum trade size
            trade_value = qty_delta * current_price

            # Check if we have enough cash (for long) or margin (for short)
            required_cash = max(0, trade_value)  # For long positions
            if required_cash > cash and qty_delta > 0:
                # Insufficient cash, scale down
                affordable_qty = cash / current_price
                qty_delta = affordable_qty
                target_qty = position.quantity + qty_delta
                trade_value = qty_delta * current_price

            # Execute trade
            cash -= trade_value

            trades.append({
                'timestamp': timestamp,
                'action': 'buy' if qty_delta > 0 else 'sell',
                'quantity': abs(qty_delta),
                'price': current_price,
                'value': abs(trade_value),
                'position_before': position.quantity,
                'position_after': target_qty,
                'equity': equity,
                'leverage': sizing_result.leverage_used,
            })

            # Update position
            if position.quantity == 0:
                position.entry_price = current_price
            elif target_qty == 0:
                position.entry_price = 0.0
            else:
                # Average entry price for adds
                position.entry_price = (
                    (position.entry_price * position.quantity + current_price * qty_delta) /
                    (position.quantity + qty_delta)
                )

            position.quantity = target_qty

        leverage_values.append(sizing_result.leverage_used)
        equity_curve.append(equity)
        timestamps.append(timestamp)

    # Calculate performance metrics
    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_equity) / initial_equity

    # Sharpe ratio (annualized)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(returns) > 1:
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (np.array(equity_curve) - peak) / peak
    max_dd = abs(drawdown.min())

    # Win rate
    winning_trades = sum(1 for t in trades if t.get('position_after', 0) == 0 and
                        (t['value'] * (1 if t['action'] == 'sell' else -1)) > 0)
    closing_trades = sum(1 for t in trades if t.get('position_after', 0) == 0)
    win_rate = winning_trades / closing_trades if closing_trades > 0 else 0.0

    return SimulationResult(
        strategy_name=strategy.name,
        symbol=symbol,
        initial_equity=initial_equity,
        final_equity=final_equity,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        num_trades=len(trades),
        avg_leverage=np.mean(leverage_values) if leverage_values else 1.0,
        total_leverage_cost=total_leverage_cost,
        win_rate=win_rate,
        equity_curve=equity_curve,
        timestamps=timestamps,
        trades=trades,
    )


def load_or_generate_predictions(
    symbol: str,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load predictions from models or generate synthetic predictions.

    Returns DataFrame with 'predicted_return' and 'predicted_volatility' columns.
    """
    # Try to load from backtest_test3_inline (uses real models)
    try:
        from backtest_test3_inline import backtest_forecasts

        print(f"Generating predictions for {symbol} using compiled models...")
        forecast_df = backtest_forecasts(symbol, num_simulations=len(prices))

        # Extract predicted returns and volatility from forecast
        if 'mean' in forecast_df.columns:
            predicted_return = forecast_df['mean'].values
        elif 'predicted_return' in forecast_df.columns:
            predicted_return = forecast_df['predicted_return'].values
        else:
            # Use simple momentum as fallback
            predicted_return = prices['close'].pct_change(5).values

        if 'std' in forecast_df.columns:
            predicted_volatility = forecast_df['std'].values
        elif 'predicted_volatility' in forecast_df.columns:
            predicted_volatility = forecast_df['predicted_volatility'].values
        else:
            # Use rolling volatility
            predicted_volatility = prices['close'].pct_change().rolling(20).std().values

        return pd.DataFrame({
            'predicted_return': predicted_return,
            'predicted_volatility': predicted_volatility,
        }, index=prices.index)

    except Exception as e:
        print(f"Could not load real predictions ({e}), using synthetic predictions")

        # Generate synthetic predictions based on historical patterns
        returns = prices['close'].pct_change()

        # Use momentum + mean reversion as prediction
        momentum = returns.rolling(5).mean()
        predicted_return = momentum.shift(1)  # Predict next period

        # Use realized volatility as prediction
        predicted_volatility = returns.rolling(20).std().shift(1)

        return pd.DataFrame({
            'predicted_return': predicted_return.fillna(0),
            'predicted_volatility': predicted_volatility.fillna(0.01),
        }, index=prices.index)


def load_price_data(symbol: str) -> pd.DataFrame:
    """Load historical price data for a symbol."""
    # Try loading from data feed
    try:
        from marketsimulator.data_feed import load_price_series

        series = load_price_series(symbol)
        if series is not None and hasattr(series, 'frame'):
            df = series.frame.copy()
            # Ensure we have required columns (case insensitive)
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        print(f"Could not load from data feed ({e}), trying alternative sources")

    # Fallback: try loading from strategytraining/raw_data
    raw_data_dir = Path(__file__).parent.parent / "strategytraining" / "raw_data"
    if raw_data_dir.exists():
        for file in raw_data_dir.glob(f"*{symbol}*.parquet"):
            try:
                df = pd.read_parquet(file)
                df.columns = [c.lower() for c in df.columns]
                return df
            except:
                pass

    raise ValueError(f"Could not load price data for {symbol}")


def run_experiments(
    symbols: List[str],
    strategies: List[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Run sizing strategy experiments on multiple symbols."""

    results = []

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Testing {symbol}")
        print('='*60)

        # Load price data
        try:
            prices = load_price_data(symbol)
            print(f"Loaded {len(prices)} bars for {symbol}")
        except Exception as e:
            print(f"ERROR loading {symbol}: {e}")
            continue

        is_crypto = symbol.endswith('USD') or 'BTC' in symbol or 'ETH' in symbol

        # Generate predictions
        try:
            predictions = load_or_generate_predictions(symbol, prices)
            print(f"Generated {len(predictions)} predictions")
        except Exception as e:
            print(f"ERROR generating predictions for {symbol}: {e}")
            continue

        # Test each strategy
        for strategy_name in strategies:
            if strategy_name not in SIZING_STRATEGIES:
                print(f"WARNING: Unknown strategy '{strategy_name}', skipping")
                continue

            strategy = SIZING_STRATEGIES[strategy_name]
            print(f"\n  Testing {strategy_name}...")

            try:
                result = simulate_strategy(
                    strategy=strategy,
                    symbol=symbol,
                    prices=prices,
                    predictions=predictions,
                    is_crypto=is_crypto,
                )

                results.append(result)

                print(f"    Return: {result.total_return:+.2%}")
                print(f"    Sharpe: {result.sharpe_ratio:.2f}")
                print(f"    MaxDD: {result.max_drawdown:.2%}")
                print(f"    Trades: {result.num_trades}")
                print(f"    Avg Leverage: {result.avg_leverage:.2f}x")
                print(f"    Leverage Cost: ${result.total_leverage_cost:,.0f}")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create summary DataFrame
    summary_data = []
    for r in results:
        summary_data.append({
            'strategy': r.strategy_name,
            'symbol': r.symbol,
            'total_return': r.total_return,
            'sharpe_ratio': r.sharpe_ratio,
            'max_drawdown': r.max_drawdown,
            'num_trades': r.num_trades,
            'avg_leverage': r.avg_leverage,
            'leverage_cost': r.total_leverage_cost,
            'win_rate': r.win_rate,
            'final_equity': r.final_equity,
        })

    summary_df = pd.DataFrame(summary_data)

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    summary_path = output_dir / f"sizing_strategy_results_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n\nResults saved to {summary_path}")

    # Save detailed results as JSON
    detailed_path = output_dir / f"sizing_strategy_detailed_{timestamp}.json"
    detailed_data = []
    for r in results:
        detailed_data.append({
            'strategy': r.strategy_name,
            'symbol': r.symbol,
            'metrics': {
                'total_return': r.total_return,
                'sharpe_ratio': r.sharpe_ratio,
                'max_drawdown': r.max_drawdown,
                'num_trades': r.num_trades,
                'avg_leverage': r.avg_leverage,
                'leverage_cost': r.total_leverage_cost,
                'win_rate': r.win_rate,
            },
            'equity_curve': [float(x) for x in r.equity_curve],
            'timestamps': [str(t) for t in r.timestamps],
            'trades': r.trades,
        })

    with open(detailed_path, 'w') as f:
        json.dump(detailed_data, f, indent=2, default=str)

    print(f"Detailed results saved to {detailed_path}")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Test position sizing strategies')
    parser.add_argument('--symbol', type=str, help='Single symbol to test')
    parser.add_argument('--symbols', nargs='+', help='Multiple symbols to test')
    parser.add_argument('--strategies', nargs='+', help='Strategies to test (default: all)')
    parser.add_argument('--output-dir', type=str, default='marketsimulator/results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Determine symbols to test
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = args.symbols
    else:
        # Default test set
        symbols = ['BTCUSD', 'ETHUSD', 'AAPL', 'MSFT']

    # Determine strategies to test
    if args.strategies:
        strategies = args.strategies
    else:
        # Default: test a representative subset
        strategies = [
            'fixed_50',
            'kelly_25',
            'voltarget_10',
            'riskparity_5',
            'optimal_f',
        ]

    output_dir = Path(args.output_dir)

    print(f"Testing {len(strategies)} strategies on {len(symbols)} symbols")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Output: {output_dir}")

    if os.getenv('MARKETSIM_FAST_SIMULATE') == '1':
        print("FAST_SIMULATE mode enabled")

    summary_df = run_experiments(symbols, strategies, output_dir)

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))

    # Best strategy per metric
    print("\n" + "="*80)
    print("BEST STRATEGIES BY METRIC")
    print("="*80)

    for metric in ['total_return', 'sharpe_ratio', 'max_drawdown']:
        best_idx = summary_df[metric].idxmax() if metric != 'max_drawdown' else summary_df[metric].idxmin()
        best = summary_df.loc[best_idx]
        print(f"\nBest {metric}: {best['strategy']} on {best['symbol']}")
        print(f"  Value: {best[metric]:.4f}")


if __name__ == '__main__':
    main()
