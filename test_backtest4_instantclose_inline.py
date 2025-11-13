"""
Backtest comparison: Instant Close at EOD vs Keep Positions Open

This test compares two position management approaches for maxdiff strategies:
1. INSTANT_CLOSE: Close unfilled positions at end of day (incurs taker fees)
2. KEEP_OPEN: Leave positions open long-term (no close fees, but capital tied up)

The test evaluates both approaches with proper fee modeling to determine which performs better.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import hashlib
import pickle
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime

# Import from existing backtest
from backtest_test3_inline import (
    download_daily_stock_data,
    fetch_spread,
    run_single_simulation,
    StrategyEvaluation,
)
from src.fixtures import crypto_symbols
from src.logging_utils import setup_logging
from loss_utils import CRYPTO_TRADING_FEE, TRADING_FEE

# Setup logging
logger = setup_logging("test_backtest4_instantclose.log")

# Alpaca fee structure
# Note: When we backout_of_market, we use limit orders at current market price,
# which should use maker fees (CRYPTO_TRADING_FEE), not taker fees
ALPACA_MAKER_FEE = 0.0  # No maker fees on limit orders
ALPACA_TAKER_FEE_STOCK = 0.0  # No taker fees for stocks (but SEC fees apply)
SEC_FEE_RATE = 0.0000278  # SEC fee for stocks (sell side only)


@dataclass
class PositionCloseMetrics:
    """Metrics for position close strategies."""
    total_return: float
    num_filled_trades: int
    num_unfilled_positions: int
    close_fees_paid: float
    opportunity_cost: float  # Capital tied up in unfilled positions
    net_return_after_fees: float
    avg_hold_duration_days: float
    # Side-specific metrics
    buy_return: float = 0.0
    sell_return: float = 0.0
    buy_filled: int = 0
    sell_filled: int = 0
    buy_unfilled: int = 0
    sell_unfilled: int = 0


def calculate_profit_with_limit_orders(
    close_actual: torch.Tensor,
    high_actual: torch.Tensor,
    high_pred: torch.Tensor,
    low_actual: torch.Tensor,
    low_pred: torch.Tensor,
    indicator: torch.Tensor,
    *,
    close_at_eod: bool,
    is_crypto: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate profits with limit order modeling.

    Returns:
        filled_returns: Returns from filled trades
        unfilled_flags: 1 if position didn't fill, 0 if filled
        hold_durations: Days position was held before filling (0 if not filled)
    """
    n = len(close_actual)
    filled_returns = torch.zeros(n, dtype=torch.float32)
    unfilled_flags = torch.zeros(n, dtype=torch.float32)
    hold_durations = torch.zeros(n, dtype=torch.float32)

    for i in range(n):
        is_buy = indicator[i] > 0

        if is_buy:
            # Buy at low_pred, sell at high
            entry_limit = low_pred[i]
            exit_target = high_actual[i]

            # Check if entry filled (price reached low_pred)
            entry_filled = low_actual[i] <= entry_limit

            if entry_filled:
                # Entry filled - calculate return
                if close_at_eod:
                    # Close at end of day at close price
                    profit = (close_actual[i] - entry_limit) / entry_limit
                    filled_returns[i] = profit
                    hold_durations[i] = 1.0  # Held for 1 day
                else:
                    # Wait for exit target
                    exit_filled = high_actual[i] >= exit_target
                    if exit_filled:
                        profit = (exit_target - entry_limit) / entry_limit
                        filled_returns[i] = profit
                        hold_durations[i] = 1.0
                    else:
                        # Position still open
                        unfilled_flags[i] = 1.0
            else:
                # Entry never filled
                unfilled_flags[i] = 1.0
        else:
            # Sell at high_pred, buyback at low
            entry_limit = high_pred[i]
            exit_target = low_actual[i]

            # Check if entry filled
            entry_filled = high_actual[i] >= entry_limit

            if entry_filled:
                if close_at_eod:
                    profit = (entry_limit - close_actual[i]) / entry_limit
                    filled_returns[i] = profit
                    hold_durations[i] = 1.0
                else:
                    exit_filled = low_actual[i] <= exit_target
                    if exit_filled:
                        profit = (entry_limit - exit_target) / entry_limit
                        filled_returns[i] = profit
                        hold_durations[i] = 1.0
                    else:
                        unfilled_flags[i] = 1.0
            else:
                unfilled_flags[i] = 1.0

    return filled_returns, unfilled_flags, hold_durations


_CACHE_DIR = Path("backtest_cache/forecasts")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_DATA_CACHE_DIR = Path("backtest_cache/data")
_DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_data_cache = {}

def _get_data_cache_bucket(timestamp_str: str) -> str:
    """Round timestamp to 2-hour bucket for daily data caching."""
    from datetime import datetime
    if timestamp_str and '--' in timestamp_str:
        dt = datetime.strptime(timestamp_str, '%Y-%m-%d--%H-%M-%S')
    else:
        dt = datetime.now()
    bucket_hour = (dt.hour // 2) * 2
    return dt.strftime(f'%Y-%m-%d--{bucket_hour:02d}-00-00')

def _load_or_fetch_data(timestamp_str: str):
    """Load data from disk cache or fetch and cache."""
    bucket = _get_data_cache_bucket(timestamp_str)
    cache_path = _DATA_CACHE_DIR / f"data_{bucket}.pkl"

    if cache_path.exists():
        try:
            logger.info(f"Loading data from cache (bucket={bucket})")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load data cache: {e}")

    logger.info(f"Fetching fresh data (bucket={bucket})")
    data = download_daily_stock_data(timestamp_str, symbols=None)

    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved data cache to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save data cache: {e}")

    return data

def _get_all_predictions_cache_path(symbol: str, data_hash: str) -> Path:
    return _CACHE_DIR / f"{symbol}_all_preds_{data_hash}.pkl"

def _generate_all_predictions(
    stock_data: pd.DataFrame,
    symbol: str,
    trading_fee: float,
    spread: float,
    is_crypto: bool,
    num_simulations: int,
) -> Optional[List[Dict]]:
    """Pre-compute predictions for all simulations once."""
    data_hash = hashlib.md5(str(len(stock_data)).encode()).hexdigest()[:16]
    cache_path = _get_all_predictions_cache_path(symbol, data_hash)

    if cache_path.exists():
        try:
            logger.info(f"Loading cached predictions for {symbol}...")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load prediction cache: {e}")

    logger.info(f"Generating predictions for {num_simulations} simulations...")
    all_predictions = []

    for sim_idx in range(num_simulations):
        simulation_data = stock_data.iloc[:-(sim_idx + 1)].copy(deep=True)
        if simulation_data.empty or len(simulation_data) < 100:
            all_predictions.append(None)
            continue

        try:
            result = run_single_simulation(
                simulation_data,
                symbol,
                trading_fee,
                is_crypto,
                sim_idx,
                spread,
            )

            last_preds = result.get('last_preds')
            all_predictions.append(last_preds)

            if (sim_idx + 1) % 5 == 0:
                logger.info(f"  Generated predictions for {sim_idx + 1}/{num_simulations} simulations")

        except Exception as exc:
            logger.warning(f"Failed to generate predictions for sim {sim_idx}: {exc}")
            all_predictions.append(None)

    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(all_predictions, f)
        logger.info(f"Saved prediction cache to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save prediction cache: {e}")

    return all_predictions


def evaluate_strategy_with_close_policy(
    last_preds: Dict[str, torch.Tensor],
    simulation_data: pd.DataFrame,
    *,
    close_at_eod: bool,
    is_crypto: bool,
    strategy_name: str,
) -> PositionCloseMetrics:
    """Evaluate strategy with specific position close policy."""

    close_actual = torch.as_tensor(
        last_preds.get("close_actual_movement_values", torch.tensor([], dtype=torch.float32)),
        dtype=torch.float32,
    )
    validation_len = int(close_actual.numel())

    if validation_len == 0:
        return PositionCloseMetrics(
            total_return=0.0,
            num_filled_trades=0,
            num_unfilled_positions=0,
            close_fees_paid=0.0,
            opportunity_cost=0.0,
            net_return_after_fees=0.0,
            avg_hold_duration_days=0.0,
        )

    # Get price series
    high_series = simulation_data["High"].iloc[-(validation_len + 2):-2]
    low_series = simulation_data["Low"].iloc[-(validation_len + 2):-2]
    close_series = simulation_data["Close"].iloc[-(validation_len + 2):-2]

    close_vals = close_series.to_numpy(dtype=float)
    high_vals = high_series.to_numpy(dtype=float)
    low_vals = low_series.to_numpy(dtype=float)

    # Calculate adjustments
    with np.errstate(divide="ignore", invalid="ignore"):
        close_to_high_np = np.abs(1.0 - np.divide(high_vals, close_vals, out=np.zeros_like(high_vals), where=close_vals != 0.0))
        close_to_low_np = np.abs(1.0 - np.divide(low_vals, close_vals, out=np.zeros_like(low_vals), where=close_vals != 0.0))

    close_to_high = torch.tensor(close_to_high_np, dtype=torch.float32)
    close_to_low = torch.tensor(close_to_low_np, dtype=torch.float32)

    # Get predictions
    high_actual_values = last_preds.get("high_actual_movement_values")
    low_actual_values = last_preds.get("low_actual_movement_values")
    high_pred_values = last_preds.get("high_predictions")
    low_pred_values = last_preds.get("low_predictions")

    if None in [high_actual_values, low_actual_values, high_pred_values, low_pred_values]:
        return PositionCloseMetrics(
            total_return=0.0,
            num_filled_trades=0,
            num_unfilled_positions=0,
            close_fees_paid=0.0,
            opportunity_cost=0.0,
            net_return_after_fees=0.0,
            avg_hold_duration_days=0.0,
        )

    high_actual = torch.as_tensor(high_actual_values, dtype=torch.float32) + close_to_high
    low_actual = torch.as_tensor(low_actual_values, dtype=torch.float32) - close_to_low
    high_pred = torch.as_tensor(high_pred_values, dtype=torch.float32) + close_to_high
    low_pred = torch.as_tensor(low_pred_values, dtype=torch.float32) - close_to_low

    # Evaluate buy side
    buy_indicator = torch.ones_like(close_actual)
    buy_returns, buy_unfilled, buy_hold = calculate_profit_with_limit_orders(
        close_actual, high_actual, high_pred, low_actual, low_pred, buy_indicator,
        close_at_eod=close_at_eod, is_crypto=is_crypto
    )

    # Evaluate sell side (only for non-crypto)
    if is_crypto:
        sell_returns = torch.zeros_like(buy_returns)
        sell_unfilled = torch.zeros_like(buy_unfilled)
        sell_hold = torch.zeros_like(buy_hold)
    else:
        sell_indicator = -torch.ones_like(close_actual)
        sell_returns, sell_unfilled, sell_hold = calculate_profit_with_limit_orders(
            close_actual, high_actual, high_pred, low_actual, low_pred, sell_indicator,
            close_at_eod=close_at_eod, is_crypto=is_crypto
        )

    # Calculate metrics
    buy_returns_np = buy_returns.numpy()
    sell_returns_np = sell_returns.numpy()
    total_returns = buy_returns_np + sell_returns_np

    buy_unfilled_np = buy_unfilled.numpy()
    sell_unfilled_np = sell_unfilled.numpy()
    total_unfilled = buy_unfilled_np + sell_unfilled_np

    all_holds = np.concatenate([buy_hold.numpy(), sell_hold.numpy()])

    # Overall metrics
    num_filled = int(np.count_nonzero(total_returns))
    num_unfilled = int(np.sum(total_unfilled))
    gross_return = float(total_returns.sum())

    # Side-specific metrics
    buy_gross = float(buy_returns_np.sum())
    sell_gross = float(sell_returns_np.sum())
    buy_filled_count = int(np.count_nonzero(buy_returns_np))
    sell_filled_count = int(np.count_nonzero(sell_returns_np))
    buy_unfilled_count = int(np.sum(buy_unfilled_np))
    sell_unfilled_count = int(np.sum(sell_unfilled_np))

    # Calculate fees
    close_fees = 0.0
    if close_at_eod:
        # Use maker fees when backing out with limit orders at market price
        close_fee = CRYPTO_TRADING_FEE if is_crypto else TRADING_FEE
        # Assume each unfilled position that entered gets closed
        num_entered_unfilled = num_unfilled  # Simplified assumption
        close_fees = num_entered_unfilled * close_fee

    # Calculate opportunity cost (capital tied up)
    opportunity_cost = 0.0
    if not close_at_eod:
        # Assuming 5% annual return on freed capital
        annual_rate = 0.05
        avg_hold_days = float(all_holds[all_holds > 0].mean()) if len(all_holds[all_holds > 0]) > 0 else 0.0
        opportunity_cost = (num_unfilled * annual_rate * avg_hold_days) / 365.0

    avg_hold_duration = float(all_holds[all_holds > 0].mean()) if len(all_holds[all_holds > 0]) > 0 else 0.0
    net_return = gross_return - close_fees - opportunity_cost

    return PositionCloseMetrics(
        total_return=gross_return,
        num_filled_trades=num_filled,
        num_unfilled_positions=num_unfilled,
        close_fees_paid=close_fees,
        opportunity_cost=opportunity_cost,
        net_return_after_fees=net_return,
        avg_hold_duration_days=avg_hold_duration,
        buy_return=buy_gross,
        sell_return=sell_gross,
        buy_filled=buy_filled_count,
        sell_filled=sell_filled_count,
        buy_unfilled=buy_unfilled_count,
        sell_unfilled=sell_unfilled_count,
    )


def compare_close_policies(symbol: str, num_simulations: int = 50) -> Optional[Dict[str, float]]:
    """
    Compare instant-close vs keep-open policies for a symbol.

    Prints comparative results including fees and opportunity costs.
    """
    import time
    total_start = time.perf_counter()
    print(f"\n{'='*80}")
    print(f"Close Policy Comparison for {symbol}")
    print(f"{'='*80}\n")

    logger.info(f"Loading data for {symbol}...")

    data_start = time.perf_counter()
    current_time_formatted = '2024-09-07--03-36-27'  # Use same test dataset

    bucket = _get_data_cache_bucket(current_time_formatted)
    if bucket not in _data_cache:
        _data_cache[bucket] = _load_or_fetch_data(current_time_formatted)

    all_data = _data_cache[bucket]

    # Handle both indexed and non-indexed DataFrames
    if 'symbol' in all_data.index.names:
        stock_data = all_data.loc[symbol] if symbol in all_data.index.get_level_values('symbol') else all_data[all_data.index.get_level_values('symbol') == symbol]
    elif 'symbol' in all_data.columns:
        stock_data = all_data[all_data['symbol'] == symbol]
    else:
        # Assume single symbol data
        stock_data = all_data

    logger.info(f"Data loading took {time.perf_counter() - data_start:.3f}s")

    is_crypto = symbol in crypto_symbols
    trading_fee = CRYPTO_TRADING_FEE if is_crypto else TRADING_FEE
    trading_days_per_year = 365 if is_crypto else 252

    # Get spread
    spread = fetch_spread(symbol)
    logger.info(f"Using spread {spread} for {symbol}")

    # Adjust num_simulations based on available data
    if len(stock_data) < num_simulations + 10:
        logger.warning(f"Not enough data for {num_simulations} simulations. Using {len(stock_data) - 10} instead.")
        num_simulations = max(10, len(stock_data) - 10)

    logger.info(f"Running {num_simulations} simulations...")

    pred_start = time.perf_counter()
    # Pre-compute all predictions once
    all_predictions = _generate_all_predictions(
        stock_data, symbol, trading_fee, spread, is_crypto, num_simulations
    )
    logger.info(f"Prediction generation/loading took {time.perf_counter() - pred_start:.3f}s")

    if all_predictions is None:
        print("Failed to generate predictions\n")
        return None

    opt_start = time.perf_counter()

    # Collect metrics for each policy
    instant_close_metrics = []
    keep_open_metrics = []

    import time
    sim_times = []
    for sim_idx in range(num_simulations):
        sim_start = time.perf_counter()
        last_preds = all_predictions[sim_idx]
        if last_preds is None:
            continue

        simulation_data = stock_data.iloc[:-(sim_idx + 1)].copy(deep=True)
        if simulation_data.empty or len(simulation_data) < 100:
            continue

        try:

            # Evaluate both policies using the REAL MaxDiffAlwaysOn strategy with grid search
            from backtest_test3_inline import evaluate_maxdiff_always_on_strategy

            # instant_close: Run grid search WITH close_at_eod=True
            instant_eval, instant_returns, instant_meta = evaluate_maxdiff_always_on_strategy(
                last_preds, simulation_data,
                trading_fee=trading_fee,
                trading_days_per_year=trading_days_per_year,
                is_crypto=is_crypto,
                close_at_eod=True
            )

            # keep_open: Run grid search WITH close_at_eod=False
            keep_eval, keep_returns, keep_meta = evaluate_maxdiff_always_on_strategy(
                last_preds, simulation_data,
                trading_fee=trading_fee,
                trading_days_per_year=trading_days_per_year,
                is_crypto=is_crypto,
                close_at_eod=False
            )
            sim_times.append(time.perf_counter() - sim_start)

            # Convert to metrics format
            instant_metrics = PositionCloseMetrics(
                total_return=instant_eval.total_return * 100,
                num_filled_trades=instant_meta.get("maxdiffalwayson_trades_total", 0),
                num_unfilled_positions=0,
                close_fees_paid=0.0,  # Already included in returns
                opportunity_cost=0.0,
                net_return_after_fees=instant_eval.total_return * 100,
                avg_hold_duration_days=1.0 if True else 0.0,  # instant close = 1 day max
                buy_return=instant_meta.get("maxdiffalwayson_buy_contribution", 0.0) * 100,
                sell_return=instant_meta.get("maxdiffalwayson_sell_contribution", 0.0) * 100,
                buy_filled=instant_meta.get("maxdiffalwayson_filled_buy_trades", 0),
                sell_filled=instant_meta.get("maxdiffalwayson_filled_sell_trades", 0),
                buy_unfilled=0,
                sell_unfilled=0,
            )

            keep_metrics = PositionCloseMetrics(
                total_return=keep_eval.total_return * 100,
                num_filled_trades=keep_meta.get("maxdiffalwayson_trades_total", 0),
                num_unfilled_positions=0,
                close_fees_paid=0.0,  # Already included in returns
                opportunity_cost=0.0,
                net_return_after_fees=keep_eval.total_return * 100,
                avg_hold_duration_days=0.0,  # keep open may hold longer
                buy_return=keep_meta.get("maxdiffalwayson_buy_contribution", 0.0) * 100,
                sell_return=keep_meta.get("maxdiffalwayson_sell_contribution", 0.0) * 100,
                buy_filled=keep_meta.get("maxdiffalwayson_filled_buy_trades", 0),
                sell_filled=keep_meta.get("maxdiffalwayson_filled_sell_trades", 0),
                buy_unfilled=0,
                sell_unfilled=0,
            )

            instant_close_metrics.append(instant_metrics)
            keep_open_metrics.append(keep_metrics)

            if (sim_idx + 1) % 10 == 0:
                logger.info(f"  Completed {sim_idx + 1}/{num_simulations} simulations")

        except Exception as exc:
            logger.warning(f"Simulation {sim_idx} failed: {exc}")
            continue

    if not instant_close_metrics or not keep_open_metrics:
        print("❌ No valid simulations completed\n")
        return None

    logger.info(f"Optimization loop took {time.perf_counter() - opt_start:.3f}s")
    if sim_times:
        avg_time = sum(sim_times) / len(sim_times)
        logger.info(f"Avg time per sim: {avg_time:.3f}s (min={min(sim_times):.3f}s, max={max(sim_times):.3f}s)")

    # Aggregate results
    def avg_metrics(metrics_list: List[PositionCloseMetrics]) -> PositionCloseMetrics:
        return PositionCloseMetrics(
            total_return=sum(m.total_return for m in metrics_list) / len(metrics_list),
            num_filled_trades=int(sum(m.num_filled_trades for m in metrics_list) / len(metrics_list)),
            num_unfilled_positions=int(sum(m.num_unfilled_positions for m in metrics_list) / len(metrics_list)),
            close_fees_paid=sum(m.close_fees_paid for m in metrics_list) / len(metrics_list),
            opportunity_cost=sum(m.opportunity_cost for m in metrics_list) / len(metrics_list),
            net_return_after_fees=sum(m.net_return_after_fees for m in metrics_list) / len(metrics_list),
            avg_hold_duration_days=sum(m.avg_hold_duration_days for m in metrics_list) / len(metrics_list),
            buy_return=sum(m.buy_return for m in metrics_list) / len(metrics_list),
            sell_return=sum(m.sell_return for m in metrics_list) / len(metrics_list),
            buy_filled=int(sum(m.buy_filled for m in metrics_list) / len(metrics_list)),
            sell_filled=int(sum(m.sell_filled for m in metrics_list) / len(metrics_list)),
            buy_unfilled=int(sum(m.buy_unfilled for m in metrics_list) / len(metrics_list)),
            sell_unfilled=int(sum(m.sell_unfilled for m in metrics_list) / len(metrics_list)),
        )

    instant_avg = avg_metrics(instant_close_metrics)
    keep_avg = avg_metrics(keep_open_metrics)

    # Print results
    print(f"Completed {len(instant_close_metrics)} simulations\n")
    print("-" * 95)
    print(f"{'Policy':<15} {'Gross Ret':<12} {'Filled':<8} {'Unfilled':<9} {'Close Fee':<11} {'Opp Cost':<10} {'Net Return':<12}")
    print("-" * 95)

    print(f"{'instant_close':<15} {instant_avg.total_return:>10.4f}% {instant_avg.num_filled_trades:>7} "
          f"{instant_avg.num_unfilled_positions:>8} {instant_avg.close_fees_paid:>10.4f} "
          f"{instant_avg.opportunity_cost:>9.4f} {instant_avg.net_return_after_fees:>11.4f}%")

    print(f"{'keep_open':<15} {keep_avg.total_return:>10.4f}% {keep_avg.num_filled_trades:>7} "
          f"{keep_avg.num_unfilled_positions:>8} {keep_avg.close_fees_paid:>10.4f} "
          f"{keep_avg.opportunity_cost:>9.4f} {keep_avg.net_return_after_fees:>11.4f}%")

    print("-" * 95)

    # Side breakdown for stocks
    if not is_crypto:
        print(f"\n{'Side Breakdown':<15} {'Buy Return':<12} {'Sell Return':<13} {'Buy Filled':<11} {'Sell Filled':<12}")
        print("-" * 95)
        print(f"{'instant_close':<15} {instant_avg.buy_return:>10.4f}% {instant_avg.sell_return:>11.4f}% "
              f"{instant_avg.buy_filled:>10} {instant_avg.sell_filled:>11}")
        print(f"{'keep_open':<15} {keep_avg.buy_return:>10.4f}% {keep_avg.sell_return:>11.4f}% "
              f"{keep_avg.buy_filled:>10} {keep_avg.sell_filled:>11}")
        print("-" * 95)

        # Calculate which side benefits more from KEEP_OPEN
        buy_advantage = keep_avg.buy_return - instant_avg.buy_return
        sell_advantage = keep_avg.sell_return - instant_avg.sell_return

        print(f"\nSide-Specific Analysis:")
        print(f"   Buy side (long):  KEEP_OPEN {'+' if buy_advantage > 0 else ''}{buy_advantage:.4f}% vs INSTANT_CLOSE")
        print(f"   Sell side (short): KEEP_OPEN {'+' if sell_advantage > 0 else ''}{sell_advantage:.4f}% vs INSTANT_CLOSE")

    # Determine winner
    if instant_avg.net_return_after_fees > keep_avg.net_return_after_fees:
        winner = "INSTANT_CLOSE"
        advantage = instant_avg.net_return_after_fees - keep_avg.net_return_after_fees
    else:
        winner = "KEEP_OPEN"
        advantage = keep_avg.net_return_after_fees - instant_avg.net_return_after_fees

    print(f"\nOverall Recommendation for {symbol}:")
    print(f"   {winner} performs better (advantage: {advantage:.4f}%)")

    print(f"{'='*80}\n")

    # Return structured results
    result = {
        'symbol': symbol,
        'is_crypto': is_crypto,
        'num_simulations': len(instant_close_metrics),
        'instant_close_net_return': instant_avg.net_return_after_fees,
        'keep_open_net_return': keep_avg.net_return_after_fees,
        'best_policy': winner,
        'advantage': advantage,
        'instant_close_fees': instant_avg.close_fees_paid,
        'keep_open_opportunity_cost': keep_avg.opportunity_cost,
    }

    # Add side-specific results for stocks
    if not is_crypto:
        result['buy_advantage'] = buy_advantage
        result['sell_advantage'] = sell_advantage
        result['instant_close_buy_return'] = instant_avg.buy_return
        result['instant_close_sell_return'] = instant_avg.sell_return
        result['keep_open_buy_return'] = keep_avg.buy_return
        result['keep_open_sell_return'] = keep_avg.sell_return

    return result


if __name__ == "__main__":
    import sys

    # Test with crypto first (quick test with 10 simulations)
    print("\n🔬 CRYPTO ANALYSIS")
    print("="*80)

    logger.info("Starting backtest comparison analysis...")

    compare_close_policies("BTCUSD", num_simulations=10)
    compare_close_policies("ETHUSD", num_simulations=10)

    # Test with stocks
    print("\n🔬 STOCK ANALYSIS")
    print("="*80)
    compare_close_policies("GOOG", num_simulations=10)
    compare_close_policies("META", num_simulations=10)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")
