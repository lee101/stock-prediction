"""
Robust profitability test for Chronos2 - measures both MAE and trading profitability.
Runs walk-forward testing on training data to simulate real trading scenarios.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

os.environ["ONLY_CHRONOS2"] = "1"
os.environ["REAL_TESTING"] = "1"

print("="*80)
print("CHRONOS2 PROFITABILITY TEST - Training Data Walk-Forward")
print("="*80)

def simulate_trading(predictions, ground_truth_prices, initial_capital=10000):
    """
    Simulate trading based on predictions.

    Simple strategy:
    - If predicted price increase > 0.5%, buy (go long)
    - If predicted price decrease > 0.5%, sell/short
    - Otherwise, hold

    Returns: final capital, total return %, trade count, win rate
    """
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long, -1 = short
    entry_price = 0
    trades = []

    for i in range(len(predictions) - 1):
        current_price = ground_truth_prices[i]
        next_price = ground_truth_prices[i + 1]
        predicted_next = predictions[i]

        # Calculate predicted return
        pred_return = (predicted_next - current_price) / current_price

        # Trading decision
        if position == 0:  # No position
            if pred_return > 0.005:  # Predict >0.5% gain
                # Enter long
                position = 1
                entry_price = current_price
                shares = capital / current_price
            elif pred_return < -0.005:  # Predict >0.5% loss
                # Enter short
                position = -1
                entry_price = current_price
                shares = capital / current_price
        else:  # Have position
            # Exit position
            actual_return = (next_price - entry_price) / entry_price
            pnl = capital * actual_return * position
            capital += pnl

            trades.append({
                'entry': entry_price,
                'exit': next_price,
                'pnl': pnl,
                'return': actual_return * position,
                'position': position,
            })

            position = 0

    # Close any open position at the end
    if position != 0:
        final_price = ground_truth_prices[-1]
        actual_return = (final_price - entry_price) / entry_price
        pnl = capital * actual_return * position
        capital += pnl
        trades.append({
            'entry': entry_price,
            'exit': final_price,
            'pnl': pnl,
            'return': actual_return * position,
            'position': position,
        })

    if len(trades) > 0:
        total_return = ((capital - initial_capital) / initial_capital) * 100
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = (wins / len(trades)) * 100 if len(trades) > 0 else 0
        avg_pnl = np.mean([t['pnl'] for t in trades])
    else:
        total_return = 0
        win_rate = 0
        avg_pnl = 0

    return {
        'final_capital': capital,
        'total_return_pct': total_return,
        'trade_count': len(trades),
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'trades': trades,
    }

def walk_forward_test(symbol, mode_name, torch_compiled, n_windows=10, window_size=100, prediction_length=7):
    """
    Walk-forward testing on training data.

    For each window:
    1. Use window_size rows as context
    2. Predict next prediction_length days
    3. Compare predictions to actual values
    4. Simulate trading based on predictions
    """
    print(f"\n{'='*80}")
    print(f"Walk-Forward Test: {mode_name}")
    print(f"{'='*80}")

    # Set environment
    os.environ["TORCH_COMPILED"] = torch_compiled

    # Force reimport
    if "backtest_test3_inline" in sys.modules:
        del sys.modules["backtest_test3_inline"]

    from backtest_test3_inline import (
        load_chronos2_wrapper,
        resolve_chronos2_params,
    )

    # Load full training data
    data_path = Path(__file__).parent / "trainingdata" / f"{symbol}.csv"
    df_full = pd.read_csv(data_path)
    df_full = df_full.tail(window_size * n_windows + prediction_length * n_windows).copy()
    df_full = df_full.reset_index(drop=True)
    df_full.columns = [col.lower() for col in df_full.columns]

    print(f"Loaded {len(df_full)} rows of {symbol} data")
    print(f"Running {n_windows} walk-forward windows...")

    # Load model once
    params = resolve_chronos2_params(symbol)
    wrapper = load_chronos2_wrapper(params)
    print(f"✓ Model loaded")

    # Track results across all windows
    all_maes = []
    all_predictions = []
    all_ground_truth = []
    failed_windows = 0

    for window_idx in range(n_windows):
        start_idx = window_idx * (window_size + prediction_length)
        end_idx = start_idx + window_size
        pred_end_idx = end_idx + prediction_length

        if pred_end_idx > len(df_full):
            break

        # Extract context and ground truth
        context = df_full.iloc[start_idx:end_idx].copy()
        ground_truth = df_full.iloc[end_idx:pred_end_idx].copy()

        # Prepare context dataframe
        context["timestamp"] = pd.date_range("2024-01-01", periods=len(context), freq="D")
        context["symbol"] = symbol

        try:
            # Make prediction
            result = wrapper.predict_ohlc(
                context_df=context,
                symbol=symbol,
                prediction_length=prediction_length,
                context_length=min(params["context_length"], len(context)),
                batch_size=params["batch_size"],
            )

            # Extract predictions
            median_frame = result.quantile_frames[0.5]
            pred_close = median_frame["close"].values
            true_close = ground_truth["close"].values

            # Calculate MAE
            mae = np.mean(np.abs(pred_close - true_close))
            mae_pct = (mae / np.mean(true_close)) * 100

            all_maes.append(mae_pct)
            all_predictions.extend(pred_close.tolist())
            all_ground_truth.extend(true_close.tolist())

            print(f"  Window {window_idx+1}/{n_windows}: MAE={mae_pct:.2f}%")

        except Exception as e:
            print(f"  Window {window_idx+1}/{n_windows}: FAILED - {str(e)[:80]}")
            failed_windows += 1

    # Calculate overall statistics
    if len(all_maes) > 0:
        avg_mae = np.mean(all_maes)
        std_mae = np.std(all_maes)
        min_mae = np.min(all_maes)
        max_mae = np.max(all_maes)
        success_rate = ((n_windows - failed_windows) / n_windows) * 100

        # Simulate trading with all predictions
        trading_results = simulate_trading(
            np.array(all_predictions),
            np.array(all_ground_truth),
            initial_capital=10000
        )

        print(f"\n{'='*80}")
        print(f"RESULTS: {mode_name}")
        print(f"{'='*80}")
        print(f"Success rate:      {success_rate:.1f}% ({n_windows - failed_windows}/{n_windows} windows)")
        print(f"")
        print(f"ACCURACY METRICS:")
        print(f"  Avg MAE:         {avg_mae:.2f}% ± {std_mae:.2f}%")
        print(f"  Min MAE:         {min_mae:.2f}%")
        print(f"  Max MAE:         {max_mae:.2f}%")
        print(f"")
        print(f"PROFITABILITY METRICS:")
        print(f"  Total Return:    {trading_results['total_return_pct']:+.2f}%")
        print(f"  Final Capital:   ${trading_results['final_capital']:,.2f}")
        print(f"  Trade Count:     {trading_results['trade_count']}")
        print(f"  Win Rate:        {trading_results['win_rate']:.1f}%")
        print(f"  Avg PnL/Trade:   ${trading_results['avg_pnl']:+,.2f}")
        print(f"{'='*80}")

        return {
            'mode': mode_name,
            'torch_compiled': torch_compiled,
            'success_rate': success_rate,
            'avg_mae': avg_mae,
            'std_mae': std_mae,
            'min_mae': min_mae,
            'max_mae': max_mae,
            'trading': trading_results,
            'failed_windows': failed_windows,
        }
    else:
        print(f"\n{'='*80}")
        print(f"RESULTS: {mode_name}")
        print(f"{'='*80}")
        print(f"Success rate:      0% (ALL WINDOWS FAILED)")
        print(f"{'='*80}")

        return {
            'mode': mode_name,
            'torch_compiled': torch_compiled,
            'success_rate': 0,
            'avg_mae': None,
            'trading': None,
            'failed_windows': n_windows,
        }

def main():
    """Run walk-forward profitability test."""
    symbol = "BTCUSD"
    n_windows = 10
    window_size = 100
    prediction_length = 7

    print(f"\nTest Configuration:")
    print(f"  Symbol: {symbol}")
    print(f"  Windows: {n_windows}")
    print(f"  Window size: {window_size} days")
    print(f"  Prediction length: {prediction_length} days")
    print(f"  Total data needed: ~{n_windows * (window_size + prediction_length)} rows")

    # Test eager mode (TORCH_COMPILED=0) - default recommendation
    eager_results = walk_forward_test(
        symbol=symbol,
        mode_name="EAGER MODE (TORCH_COMPILED=0)",
        torch_compiled="0",
        n_windows=n_windows,
        window_size=window_size,
        prediction_length=prediction_length,
    )

    print("\n" + "="*80)
    print("Waiting 5 seconds before compiled mode test...")
    print("="*80)
    import time
    time.sleep(5)

    # Test compiled mode (TORCH_COMPILED=1)
    compiled_results = walk_forward_test(
        symbol=symbol,
        mode_name="COMPILED MODE (TORCH_COMPILED=1)",
        torch_compiled="1",
        n_windows=n_windows,
        window_size=window_size,
        prediction_length=prediction_length,
    )

    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)

    print(f"\n{'Metric':<25} {'Eager (TORCH_COMPILED=0)':<30} {'Compiled (TORCH_COMPILED=1)':<30}")
    print("-" * 85)

    # Success rate
    eager_success = f"{eager_results['success_rate']:.1f}%"
    compiled_success = f"{compiled_results['success_rate']:.1f}%"
    print(f"{'Success Rate':<25} {eager_success:<30} {compiled_success:<30}")

    # MAE
    if eager_results['avg_mae'] and compiled_results['avg_mae']:
        eager_mae = f"{eager_results['avg_mae']:.2f}% ± {eager_results['std_mae']:.2f}%"
        compiled_mae = f"{compiled_results['avg_mae']:.2f}% ± {compiled_results['std_mae']:.2f}%"
        print(f"{'Average MAE':<25} {eager_mae:<30} {compiled_mae:<30}")

    # Profitability
    if eager_results['trading'] and compiled_results['trading']:
        eager_return = f"{eager_results['trading']['total_return_pct']:+.2f}%"
        compiled_return = f"{compiled_results['trading']['total_return_pct']:+.2f}%"
        print(f"{'Total Return':<25} {eager_return:<30} {compiled_return:<30}")

        eager_winrate = f"{eager_results['trading']['win_rate']:.1f}%"
        compiled_winrate = f"{compiled_results['trading']['win_rate']:.1f}%"
        print(f"{'Win Rate':<25} {eager_winrate:<30} {compiled_winrate:<30}")

        eager_trades = f"{eager_results['trading']['trade_count']}"
        compiled_trades = f"{compiled_results['trading']['trade_count']}"
        print(f"{'Trade Count':<25} {eager_trades:<30} {compiled_trades:<30}")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    if eager_results['success_rate'] == 100 and compiled_results['success_rate'] < 100:
        print("❌ COMPILED MODE UNSTABLE")
        print("   Recommendation: Use TORCH_COMPILED=0 (eager mode)")
        print("   Reason: Compiled mode has failures, eager mode is reliable")
    elif eager_results['success_rate'] == 100 and compiled_results['success_rate'] == 100:
        # Both stable, compare profitability
        if eager_results['trading'] and compiled_results['trading']:
            eager_profit = eager_results['trading']['total_return_pct']
            compiled_profit = compiled_results['trading']['total_return_pct']

            if compiled_profit > eager_profit * 1.1:  # 10% better
                print("✅ COMPILED MODE MORE PROFITABLE")
                print(f"   Recommendation: Use TORCH_COMPILED=1")
                print(f"   {compiled_profit:.2f}% return vs {eager_profit:.2f}% return")
            elif eager_profit > compiled_profit * 1.1:
                print("✅ EAGER MODE MORE PROFITABLE")
                print(f"   Recommendation: Use TORCH_COMPILED=0")
                print(f"   {eager_profit:.2f}% return vs {compiled_profit:.2f}% return")
            else:
                print("➡️  SIMILAR PROFITABILITY")
                print("   Recommendation: Use TORCH_COMPILED=0 (eager mode)")
                print("   Reason: Similar returns, eager mode is simpler/more reliable")
    else:
        print("⚠️  BOTH MODES HAVE ISSUES")
        print("   Further investigation needed")

    print("="*80)

if __name__ == "__main__":
    main()
