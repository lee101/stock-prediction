#!/usr/bin/env python3
"""
Realistic hyperparameter optimization test using AAPL stock data.
Tests the Toto model's ability to predict the next Close price using historical data.
"""

import numpy as np
import pandas as pd
import torch
from src.models.toto_wrapper import TotoPipeline
from pathlib import Path

def test_real_stock_prediction():
    """Test Toto model with real AAPL stock data"""
    
    # Load AAPL data
    data_file = Path("/home/lee/code/stock/data/2023-07-08 01:30:11/AAPL-2023-07-08.csv")
    df = pd.read_csv(data_file)
    
    # Extract Close prices
    close_prices = df['Close'].values
    print(f"Loaded {len(close_prices)} AAPL Close prices")
    print(f"Price range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
    
    # Use all but last price as context, predict the last price
    context = close_prices[:-1]  # All except last
    actual_next = close_prices[-1]  # Last price to predict
    
    print(f"Context: Last 5 prices: {context[-5:]}")
    print(f"Actual next price: ${actual_next:.2f}")
    
    # Test different num_samples values
    pipeline = TotoPipeline.from_pretrained('Datadog/Toto-Open-Base-1.0', device_map='cuda')
    
    results = []
    
    for num_samples in [1024, 2048, 3072, 4096]:
        print(f"\nTesting num_samples={num_samples}:")
        
        # Run multiple predictions to test consistency
        predictions = []
        errors = []
        
        for run in range(3):
            forecasts = pipeline.predict(
                context=context.tolist(),
                prediction_length=1,
                num_samples=num_samples
            )
            
            tensor = forecasts[0]
            predicted_values = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
            mean_pred = np.mean(predicted_values)
            predictions.append(mean_pred)
            
            # Calculate percentage error
            error = abs(mean_pred - actual_next) / actual_next * 100
            errors.append(error)
            
            print(f"  Run {run+1}: Predicted=${mean_pred:.2f}, Error={error:.2f}%")
        
        # Calculate averages
        avg_prediction = np.mean(predictions)
        avg_error = np.mean(errors)
        std_error = np.std(errors)
        
        print(f"  Average: Predicted=${avg_prediction:.2f}, Error={avg_error:.2f}% (±{std_error:.2f}%)")
        
        results.append({
            'num_samples': num_samples,
            'avg_prediction': avg_prediction,
            'avg_error': avg_error,
            'std_error': std_error,
            'predictions': predictions
        })
    
    # Find best configuration
    best_result = min(results, key=lambda x: x['avg_error'])
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY:")
    print(f"{'='*60}")
    print(f"Actual next Close price: ${actual_next:.2f}")
    print()
    
    for result in results:
        status = "✅ BEST" if result == best_result else ""
        print(f"num_samples={result['num_samples']:4d}: "
              f"Pred=${result['avg_prediction']:6.2f}, "
              f"Error={result['avg_error']:5.2f}% (±{result['std_error']:4.2f}%) {status}")
    
    print(f"\nBest configuration: num_samples={best_result['num_samples']} "
          f"with {best_result['avg_error']:.2f}% average error")
    
    return best_result

if __name__ == "__main__":
    print("Testing Toto wrapper with real AAPL stock data...")
    test_real_stock_prediction()
