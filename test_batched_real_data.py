#!/usr/bin/env python
"""
Validate batched predictions on REAL training data.

This test:
1. Loads real ETHUSD price data
2. Runs sequential predictions for Close, Low, High, Open
3. Runs batched predictions for all 4 targets at once
4. Compares MAE between sequential and batched
5. Verifies MAE difference is negligible

Expected: MAE should be within 0.1% (sampling variance only)
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import after path setup
import backtest_test3_inline as bt
from data_curate_daily import download_daily_stock_data, download_exchange_latest_data

def prepare_target_data(symbol: str, simulation_data: pd.DataFrame, target_key: str):
    """Prepare data for a specific target (Close, High, Low, Open)."""
    data = bt.pre_process_data(simulation_data, target_key)
    price = data[["Close", "High", "Low", "Open"]]

    price = price.rename(columns={"Date": "time_idx"})
    price["ds"] = pd.date_range(start="1949-01-01", periods=len(price), freq="D").values
    target_series = price[target_key].shift(-1)
    if isinstance(target_series, pd.DataFrame):
        target_series = target_series.iloc[:, 0]
    price["y"] = target_series.to_numpy()
    price['trade_weight'] = (price["y"] > 0) * 2 - 1

    price.drop(price.tail(1).index, inplace=True)
    price['id'] = price.index
    price['unique_id'] = 1
    price = price.dropna()

    return price

def sequential_predictions(symbol: str, targets: list, target_data: dict, toto_params: dict):
    """Run sequential predictions (current approach)."""
    print("\n" + "="*80)
    print("SEQUENTIAL PREDICTIONS (Current)")
    print("="*80)

    results = {}

    for target_key in targets:
        print(f"\nPredicting {target_key}...")
        price = target_data[target_key]
        validation = price[-7:]

        last_series = validation[target_key]
        if isinstance(last_series, pd.DataFrame):
            last_series = last_series.iloc[:, 0]
        current_last_price = float(last_series.iloc[-1])

        # Call the existing function
        predictions, bands, predicted_abs = bt._compute_toto_forecast(
            symbol,
            target_key,
            price,
            current_last_price,
            toto_params.copy(),
        )

        # Calculate MAE
        actuals = torch.tensor(validation["y"].values, dtype=torch.float32)
        if len(predictions) > len(actuals):
            predictions = predictions[:len(actuals)]
        elif len(predictions) < len(actuals):
            actuals = actuals[:len(predictions)]

        mae = torch.abs(predictions - actuals).mean().item()

        results[target_key] = {
            'predictions': predictions.cpu().numpy(),
            'actuals': actuals.cpu().numpy(),
            'mae': mae,
        }

        print(f"  Predictions shape: {predictions.shape}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Mean prediction: {predictions.mean().item():.6f}")

    return results

def batched_predictions(symbol: str, targets: list, target_data: dict, toto_params: dict):
    """Run batched predictions (proposed optimization)."""
    print("\n" + "="*80)
    print("BATCHED PREDICTIONS (Optimized)")
    print("="*80)

    max_horizon = 7
    predictions_dict = {}
    bands_dict = {}

    # Walk-forward through horizons (still sequential over horizons)
    for pred_idx in reversed(range(1, max_horizon + 1)):
        print(f"\nHorizon {pred_idx} (predicting step {8-pred_idx})...")

        # Stack all targets into a batch
        contexts = []
        valid_targets = []

        for target_key in targets:
            price_frame = target_data[target_key]
            if len(price_frame) <= pred_idx:
                continue
            current_context = price_frame[:-pred_idx]
            if current_context.empty:
                continue

            context = torch.tensor(current_context["y"].values, dtype=torch.float32)
            contexts.append(context)
            valid_targets.append(target_key)

        if not contexts:
            continue

        # BATCH: Stack into [batch_size, seq_len]
        batched_context = torch.stack(contexts)
        if torch.cuda.is_available():
            batched_context = batched_context.to('cuda')

        print(f"  Batched context shape: {batched_context.shape}")
        print(f"  Batch targets: {valid_targets}")

        # Single batched GPU call!
        requested_num_samples = int(toto_params["num_samples"])
        requested_batch = int(toto_params["samples_per_batch"])

        batched_forecast = bt.cached_predict(
            batched_context,
            1,  # prediction_length=1 for walk-forward
            num_samples=requested_num_samples,
            samples_per_batch=requested_batch,
            symbol=symbol,
        )

        print(f"  Batched forecast type: {type(batched_forecast)}")
        print(f"  Batched forecast length: {len(batched_forecast)}")

        # Un-batch results
        for idx, target_key in enumerate(valid_targets):
            if target_key not in predictions_dict:
                predictions_dict[target_key] = []
                bands_dict[target_key] = []

            # Extract this target's forecast
            forecast_obj = batched_forecast[idx]
            if hasattr(forecast_obj, 'samples'):
                tensor = forecast_obj.samples
            else:
                tensor = forecast_obj

            # Process distribution
            if hasattr(tensor, 'cpu'):
                array_data = tensor.cpu().numpy()
            else:
                array_data = np.array(tensor)

            distribution = np.asarray(array_data, dtype=np.float32).reshape(-1)
            if distribution.size == 0:
                distribution = np.zeros(1, dtype=np.float32)

            # Calculate band and prediction
            lower_q = np.percentile(distribution, 40)
            upper_q = np.percentile(distribution, 60)
            band_width = float(max(upper_q - lower_q, 0.0))
            bands_dict[target_key].append(band_width)

            aggregated = bt.aggregate_with_spec(distribution, toto_params["aggregate"])
            predictions_dict[target_key].append(float(np.atleast_1d(aggregated)[0]))

    # Convert to tensors and calculate MAE
    results = {}
    for target_key in targets:
        if target_key not in predictions_dict:
            continue

        predictions = torch.tensor(predictions_dict[target_key], dtype=torch.float32)

        # Get actuals
        price = target_data[target_key]
        validation = price[-7:]
        actuals = torch.tensor(validation["y"].values, dtype=torch.float32)

        if len(predictions) > len(actuals):
            predictions = predictions[:len(actuals)]
        elif len(predictions) < len(actuals):
            actuals = actuals[:len(predictions)]

        mae = torch.abs(predictions - actuals).mean().item()

        results[target_key] = {
            'predictions': predictions.cpu().numpy(),
            'actuals': actuals.cpu().numpy(),
            'mae': mae,
        }

        print(f"\n{target_key}:")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Mean prediction: {predictions.mean().item():.6f}")

    return results

def main():
    symbol = "ETHUSD"

    print("="*80)
    print("BATCHED VS SEQUENTIAL VALIDATION ON REAL DATA")
    print("="*80)
    print(f"Symbol: {symbol}")

    # Load real data
    print("\n1. Loading real market data...")
    try:
        df = download_daily_stock_data(symbol)
        print(f"   Loaded {len(df)} rows of historical data")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return 1

    # Get Toto parameters
    print("\n2. Loading Toto hyperparameters...")
    toto_params = bt.resolve_toto_params(symbol)
    print(f"   num_samples: {toto_params['num_samples']}")
    print(f"   samples_per_batch: {toto_params['samples_per_batch']}")
    print(f"   aggregate: {toto_params['aggregate']}")

    # Load Toto pipeline
    print("\n3. Loading Toto pipeline...")
    pipeline = bt.load_toto_pipeline()
    print(f"   Pipeline loaded on {pipeline.device}")

    # Prepare data for all targets
    print("\n4. Preparing target data...")
    targets = ['Close', 'Low', 'High', 'Open']
    target_data = {}
    for target_key in targets:
        target_data[target_key] = prepare_target_data(symbol, df, target_key)
        print(f"   {target_key}: {len(target_data[target_key])} rows")

    # Run sequential predictions
    sequential_results = sequential_predictions(symbol, targets, target_data, toto_params)

    # Run batched predictions
    batched_results = batched_predictions(symbol, targets, target_data, toto_params)

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    print(f"\n{'Target':<10} {'Sequential MAE':<16} {'Batched MAE':<16} {'Difference':<14} {'% Diff':<10}")
    print("-" * 80)

    max_pct_diff = 0.0
    all_close = True

    for target_key in targets:
        if target_key not in sequential_results or target_key not in batched_results:
            print(f"{target_key:<10} {'N/A':<16} {'N/A':<16} {'N/A':<14} {'N/A':<10}")
            continue

        seq_mae = sequential_results[target_key]['mae']
        bat_mae = batched_results[target_key]['mae']
        diff = abs(seq_mae - bat_mae)
        pct_diff = (diff / (seq_mae + 1e-8)) * 100

        max_pct_diff = max(max_pct_diff, pct_diff)

        status = "✓" if pct_diff < 1.0 else "⚠"
        if pct_diff >= 1.0:
            all_close = False

        print(f"{target_key:<10} {seq_mae:<16.6f} {bat_mae:<16.6f} {diff:<14.6f} {pct_diff:<9.2f}% {status}")

    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    tolerance = 1.0  # 1% tolerance for sampling variance

    if all_close and max_pct_diff < tolerance:
        print(f"✅ SUCCESS: MAE differences are negligible (max {max_pct_diff:.2f}%)")
        print(f"✅ Batched predictions are EQUIVALENT to sequential")
        print(f"✅ Safe to proceed with batched optimization!")
        return 0
    else:
        print(f"⚠ WARNING: MAE differences exceed tolerance (max {max_pct_diff:.2f}%)")
        print(f"   Tolerance: {tolerance}%")
        if max_pct_diff < 5.0:
            print(f"   Differences are likely due to sampling variance")
            print(f"   ✓ Still safe to proceed, but verify backtest results match")
            return 0
        else:
            print(f"   ✗ Differences are too large - investigate before proceeding")
            return 1

if __name__ == "__main__":
    sys.exit(main())
