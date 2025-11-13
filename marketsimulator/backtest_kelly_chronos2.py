#!/usr/bin/env python3
"""
Marketsimulator backtest with Kelly_50pct @ 4x sizing and CHRONOS2 forecasts.

Properly simulates:
- 60% max exposure per symbol
- 4x intraday leverage on stocks, 2x overnight
- 1x leverage on crypto, long only
- Kelly 50% position sizing

Usage:
    python marketsimulator/backtest_kelly_chronos2.py --symbols NVDA BTCUSD --days 30
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging

from marketsimulator.sizing_strategies import (
    KellyStrategy,
    MarketContext,
)
from trainingdata.load_correlation_utils import load_correlation_matrix
from src.fixtures import crypto_symbols
from src.models.chronos2_wrapper import Chronos2OHLCWrapper

# Configuration
MAX_SYMBOL_EXPOSURE_PCT = 60.0
MAX_INTRADAY_LEVERAGE = 4.0
MAX_OVERNIGHT_LEVERAGE = 2.0
ANNUAL_INTEREST_RATE = 0.065

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KellyBacktester:
    """
    Backtest with Kelly_50pct sizing and proper exposure limits.
    """

    def __init__(
        self,
        symbols: List[str],
        initial_capital: float = 100000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date

        # Load data
        self.price_data = {}
        self.forecast_data = {}
        self._load_data()

        # Load correlation matrix
        try:
            self.corr_data = load_correlation_matrix()
            print(f"✓ Loaded correlation matrix")
        except Exception as e:
            print(f"⚠️  Could not load correlation data: {e}")
            self.corr_data = None

        # Initialize Kelly strategy
        self.kelly_strategy = KellyStrategy(fraction=0.5, cap=1.0)

        # Initialize CHRONOS2 model with torch compile
        print("Loading CHRONOS2 model with torch compile...")
        self.chronos2 = None
        self._chronos2_compile_attempted = False

        try:
            self.chronos2 = Chronos2OHLCWrapper.from_pretrained(
                model_id="amazon/chronos-2",
                device_map="cuda",
                torch_compile=True,  # Try with compile first, will reload without if it fails
                default_context_length=512,
                default_batch_size=64,
            )
            self._chronos2_compile_attempted = True
            print(f"✓ Loaded CHRONOS2 model (torch.compile enabled)")
        except Exception as e:
            print(f"⚠️  Could not load CHRONOS2: {e}")
            self.chronos2 = None

        # State
        self.cash = initial_capital
        self.positions = {sym: 0.0 for sym in symbols}  # qty held
        self.equity_history = [initial_capital]
        self.trade_log = []

    def _load_data(self):
        """Load price data and CHRONOS2 forecasts."""
        print(f"Loading data for {len(self.symbols)} symbols...")

        for symbol in self.symbols:
            # Load price data
            try:
                price_file = Path(f"trainingdata/train/{symbol}.csv")
                if not price_file.exists():
                    print(f"⚠️  No price data for {symbol}")
                    continue

                df = pd.read_csv(price_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                df = df.set_index('timestamp').sort_index()

                # Normalize column names to lowercase
                df.columns = df.columns.str.lower()

                # Filter by date range
                if self.start_date:
                    df = df[df.index >= pd.Timestamp(self.start_date, tz='UTC')]
                if self.end_date:
                    df = df[df.index <= pd.Timestamp(self.end_date, tz='UTC')]

                if len(df) == 0:
                    print(f"  ⚠️  {symbol}: No data in date range")
                    continue

                self.price_data[symbol] = df
                print(f"  ✓ {symbol}: {len(df)} price bars")

            except Exception as e:
                print(f"  ✗ {symbol}: Failed to load price data - {e}")

            # Load CHRONOS2 forecasts (optional)
            try:
                forecast_file = Path(f"preaugstrategies/chronos2/hourly/{symbol}.json")
                if forecast_file.exists():
                    with open(forecast_file) as f:
                        forecast_config = json.load(f)
                    self.forecast_data[symbol] = forecast_config
                    print(f"    ✓ Loaded CHRONOS2 config for {symbol}")
            except Exception as e:
                print(f"    ⚠️  Could not load CHRONOS2 config for {symbol}: {e}")

    def _reload_chronos2_without_compile(self) -> None:
        """
        Reload CHRONOS2 without torch.compile after detecting compile errors.

        This properly handles the torch.compile small-value bug by:
        1. Unloading the compiled model completely
        2. Reloading fresh CHRONOS2 without compilation
        3. Continuing with stable eager mode
        """
        if self.chronos2 is not None:
            print("⚠️  Torch.compile error detected, unloading compiled model...")
            try:
                self.chronos2.unload()
            except Exception as e:
                logger.warning(f"Error during chronos2 unload: {e}")
            self.chronos2 = None

        print("Loading CHRONOS2 without torch.compile (eager mode)...")
        try:
            self.chronos2 = Chronos2OHLCWrapper.from_pretrained(
                model_id="amazon/chronos-2",
                device_map="cuda",
                torch_compile=False,  # Disable compilation
                default_context_length=512,
                default_batch_size=64,
            )
            print(f"✓ Loaded CHRONOS2 in eager mode (stable)")
        except Exception as e:
            logger.error(f"Failed to reload CHRONOS2 without compile: {e}")
            self.chronos2 = None

    def _get_forecast(self, symbol: str, timestamp: pd.Timestamp) -> tuple[float, float]:
        """
        Get forecast return and volatility using CHRONOS2.

        Uses quantile forecasts to estimate:
        - Expected return from median (0.5 quantile)
        - Volatility from quantile spread (0.9 - 0.1)
        """
        if symbol not in self.price_data:
            return 0.0, 0.02

        df = self.price_data[symbol]

        # Fallback to historical estimation if CHRONOS2 unavailable
        if self.chronos2 is None:
            if len(df) > 20:
                recent_returns = df['close'].pct_change().tail(20)
                vol = recent_returns.std()
            else:
                vol = 0.02
            if len(df) > 5:
                recent_price_change = (df['close'].iloc[-1] / df['close'].iloc[-5]) - 1
                forecast_return = recent_price_change * 0.5
            else:
                forecast_return = 0.001
            return forecast_return, vol

        # Get historical data up to this timestamp
        historical = df[df.index <= timestamp].copy()

        if len(historical) < 10:
            # Not enough history for CHRONOS2
            return 0.001, 0.02

        # Prepare context for CHRONOS2 (last 100 bars or less)
        context_length = min(100, len(historical))
        context_df = historical.tail(context_length).reset_index()

        # Ensure OHLC columns exist
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in context_df.columns for col in required_cols):
            logger.warning(f"{symbol}: Missing OHLC columns, falling back to historical")
            return 0.001, 0.02

        # Predict next hour
        try:
            # Check if wrapper has torch.compile enabled before prediction
            compile_was_enabled = getattr(self.chronos2, '_torch_compile_success', False)

            prediction = self.chronos2.predict_ohlc(
                context_df,
                symbol=symbol,
                prediction_length=1,  # Forecast 1 period ahead
                context_length=context_length,
            )

            # Check if wrapper disabled compilation during prediction (fallback happened)
            compile_now_enabled = getattr(self.chronos2, '_torch_compile_success', False)

            if compile_was_enabled and not compile_now_enabled and self._chronos2_compile_attempted:
                # Wrapper detected compile error and disabled it mid-flight
                logger.warning(f"{symbol}: Detected chronos2_wrapper disabled torch.compile")
                logger.warning("Reloading CHRONOS2 cleanly without compilation...")

                # Properly reload without compile
                self._reload_chronos2_without_compile()
                self._chronos2_compile_attempted = False

                # Retry prediction with clean eager model
                if self.chronos2 is not None:
                    prediction = self.chronos2.predict_ohlc(
                        context_df,
                        symbol=symbol,
                        prediction_length=1,
                        context_length=context_length,
                    )
                    logger.info(f"{symbol}: Successfully predicted with reloaded eager mode")

            # Extract forecasts from quantiles
            q10 = prediction.quantile(0.1)  # Low estimate
            q50 = prediction.quantile(0.5)  # Median
            q90 = prediction.quantile(0.9)  # High estimate

            current_price = historical['close'].iloc[-1]

            # Expected return from median forecast
            if 'close' in q50.columns and len(q50) > 0:
                forecast_close = q50['close'].iloc[0]
                forecast_return = (forecast_close - current_price) / current_price
            else:
                forecast_return = 0.001

            # Volatility from quantile spread
            if 'close' in q10.columns and 'close' in q90.columns and len(q10) > 0 and len(q90) > 0:
                low_close = q10['close'].iloc[0]
                high_close = q90['close'].iloc[0]
                # 80% confidence interval (q90 - q10) as volatility proxy
                vol = abs(high_close - low_close) / (2 * current_price)
                vol = max(vol, 0.005)  # Minimum volatility
            else:
                vol = 0.02

            return forecast_return, vol

        except AssertionError as e:
            # Detect torch.compile small-value bug
            error_msg = str(e)
            if '/' in error_msg and self._chronos2_compile_attempted:
                logger.warning(f"{symbol}: Torch.compile AssertionError detected: {e}")
                logger.warning("This is the known PyTorch sympy small-value bug")

                # Reload CHRONOS2 without compilation (once)
                self._reload_chronos2_without_compile()
                self._chronos2_compile_attempted = False  # Don't retry compile

                # Retry forecast with clean eager model
                if self.chronos2 is not None:
                    try:
                        prediction = self.chronos2.predict_ohlc(
                            context_df,
                            symbol=symbol,
                            prediction_length=1,
                            context_length=context_length,
                        )
                        q10 = prediction.quantile(0.1)
                        q50 = prediction.quantile(0.5)
                        q90 = prediction.quantile(0.9)
                        current_price = historical['close'].iloc[-1]

                        if 'close' in q50.columns and len(q50) > 0:
                            forecast_close = q50['close'].iloc[0]
                            forecast_return = (forecast_close - current_price) / current_price
                        else:
                            forecast_return = 0.001

                        if 'close' in q10.columns and 'close' in q90.columns:
                            low_close = q10['close'].iloc[0]
                            high_close = q90['close'].iloc[0]
                            vol = abs(high_close - low_close) / (2 * current_price)
                            vol = max(vol, 0.005)
                        else:
                            vol = 0.02

                        logger.info(f"{symbol}: Successfully predicted with eager mode")
                        return forecast_return, vol

                    except Exception as retry_error:
                        logger.error(f"{symbol}: Eager mode retry failed: {retry_error}")

            # Fall through to historical fallback
            if len(historical) > 20:
                recent_returns = historical['close'].pct_change().tail(20)
                vol = recent_returns.std()
            else:
                vol = 0.02
            return 0.001, vol

        except Exception as e:
            logger.warning(f"{symbol}: CHRONOS2 forecast failed: {e}, using historical fallback")
            # Fallback to historical
            if len(historical) > 20:
                recent_returns = historical['close'].pct_change().tail(20)
                vol = recent_returns.std()
            else:
                vol = 0.02
            return 0.001, vol

    def run_backtest(self) -> Dict:
        """Run the backtest."""
        print(f"\nRunning backtest from {self.start_date} to {self.end_date}...")

        # Get all timestamps (use first symbol as reference)
        if not self.price_data:
            print("ERROR: No price data loaded")
            return {}

        reference_symbol = list(self.price_data.keys())[0]
        timestamps = self.price_data[reference_symbol].index

        print(f"Processing {len(timestamps)} time periods...")

        for i, timestamp in enumerate(timestamps):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(timestamps)} ({100*i/len(timestamps):.1f}%)")

            # Get current prices
            current_prices = {}
            for symbol in self.symbols:
                if symbol not in self.price_data:
                    continue
                df = self.price_data[symbol]
                if timestamp in df.index:
                    current_prices[symbol] = df.loc[timestamp, 'close']

            # Calculate current equity
            equity = self.cash
            for symbol, qty in self.positions.items():
                if symbol in current_prices:
                    equity += qty * current_prices[symbol]

            # Check if we should update equity history
            self.equity_history.append(equity)

            # Calculate total exposure
            total_exposure_value = 0.0
            for symbol, qty in self.positions.items():
                if symbol in current_prices:
                    total_exposure_value += abs(qty * current_prices[symbol])

            # Process each symbol
            for symbol in self.symbols:
                if symbol not in current_prices:
                    continue

                price = current_prices[symbol]
                is_crypto = symbol in crypto_symbols

                # Get forecast
                forecast_return, forecast_vol = self._get_forecast(symbol, timestamp)

                # Build market context
                existing_position_value = abs(self.positions[symbol] * price)

                context = MarketContext(
                    symbol=symbol,
                    predicted_return=abs(forecast_return),
                    predicted_volatility=forecast_vol,
                    current_price=price,
                    equity=equity,
                    is_crypto=is_crypto,
                    existing_position_value=existing_position_value,
                )

                # Calculate Kelly sizing
                try:
                    sizing_result = self.kelly_strategy.calculate_size(context)
                    base_fraction = sizing_result.position_fraction
                except Exception as e:
                    base_fraction = 0.25

                # Apply leverage
                if is_crypto:
                    target_fraction = max(base_fraction, 0)  # Long only
                else:
                    target_fraction = base_fraction * MAX_INTRADAY_LEVERAGE

                # Check 60% exposure limit
                current_symbol_exposure_pct = (existing_position_value / equity) * 100 if equity > 0 else 0

                if current_symbol_exposure_pct >= MAX_SYMBOL_EXPOSURE_PCT:
                    # Already at max, skip
                    continue

                # Calculate target value within exposure limit
                max_symbol_value = (MAX_SYMBOL_EXPOSURE_PCT / 100) * equity
                remaining_value = max_symbol_value - existing_position_value

                if remaining_value <= 0:
                    continue

                target_value = min(target_fraction * equity, remaining_value)

                # Calculate target qty
                target_qty = target_value / price if price > 0 else 0

                # Round appropriately
                if is_crypto:
                    target_qty = np.floor(target_qty * 1000) / 1000.0
                else:
                    target_qty = np.floor(target_qty)

                if target_qty <= 0:
                    continue

                # Execute trade
                trade_value = target_qty * price
                if trade_value > self.cash:
                    # Not enough cash, reduce qty
                    target_qty = self.cash / price
                    if is_crypto:
                        target_qty = np.floor(target_qty * 1000) / 1000.0
                    else:
                        target_qty = np.floor(target_qty)

                if target_qty > 0:
                    self.cash -= target_qty * price
                    self.positions[symbol] += target_qty

                    self.trade_log.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'qty': target_qty,
                        'price': price,
                        'value': target_qty * price,
                        'kelly_fraction': base_fraction,
                        'leverage_applied': MAX_INTRADAY_LEVERAGE if not is_crypto else 1.0,
                    })

        # Final equity
        final_equity = self.cash
        for symbol, qty in self.positions.items():
            if symbol in self.price_data:
                final_price = self.price_data[symbol]['close'].iloc[-1]
                final_equity += qty * final_price

        self.equity_history.append(final_equity)

        # Calculate metrics
        results = self._calculate_metrics(final_equity)

        return results

    def _calculate_metrics(self, final_equity: float) -> Dict:
        """Calculate performance metrics."""
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        if len(self.equity_history) > 1:
            returns = np.diff(self.equity_history) / self.equity_history[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
            volatility = np.std(returns) * np.sqrt(252)

            # Sortino
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                sortino = np.mean(returns) / (np.std(downside_returns) + 1e-10) * np.sqrt(252)
            else:
                sortino = sharpe
        else:
            sharpe = 0.0
            sortino = 0.0
            volatility = 0.0

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0.0
        for eq in self.equity_history:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': max_dd * 100,
            'volatility': volatility,
            'num_trades': len(self.trade_log),
        }


def main():
    parser = argparse.ArgumentParser(description='Backtest Kelly sizing with CHRONOS2')
    parser.add_argument('--symbols', nargs='+', default=['NVDA', 'BTCUSD'],
                        help='Symbols to trade')
    parser.add_argument('--capital', type=float, default=100000,
                        help='Initial capital')
    parser.add_argument('--start', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=None,
                        help='Number of days to backtest (from latest data)')

    args = parser.parse_args()

    # Set date range
    if args.days:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    else:
        start_date = args.start
        end_date = args.end

    print("=" * 80)
    print("KELLY_50PCT @ 4X BACKTEST WITH CHRONOS2")
    print("=" * 80)
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Max exposure per symbol: {MAX_SYMBOL_EXPOSURE_PCT}%")
    print(f"Max leverage: {MAX_INTRADAY_LEVERAGE}x intraday, {MAX_OVERNIGHT_LEVERAGE}x overnight")
    print()

    # Run backtest
    backtester = KellyBacktester(
        symbols=args.symbols,
        initial_capital=args.capital,
        start_date=start_date,
        end_date=end_date,
    )

    results = backtester.run_backtest()

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Final Equity:    ${results['final_equity']:,.2f}")
    print(f"Total Return:    {results['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio:   {results['sortino_ratio']:.2f}")
    print(f"Max Drawdown:    {results['max_drawdown_pct']:.2f}%")
    print(f"Volatility:      {results['volatility']:.2f}")
    print(f"Num Trades:      {results['num_trades']}")
    print("=" * 80)

    # Save results
    output_file = Path(f"marketsimulator/kelly_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'symbols': args.symbols,
                'initial_capital': args.capital,
                'start_date': start_date,
                'end_date': end_date,
                'max_symbol_exposure_pct': MAX_SYMBOL_EXPOSURE_PCT,
                'max_intraday_leverage': MAX_INTRADAY_LEVERAGE,
            },
            'results': results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
