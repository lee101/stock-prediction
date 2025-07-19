#!/usr/bin/env python3
"""
Test script for toto_wrapper.py
Tests the model with sequence 2, 4, 6, 8, 10 -> should predict ~12
"""

import numpy as np
import torch
from src.models.toto_wrapper import TotoPipeline

def test_arithmetic_sequence():
    """Test Toto model with arithmetic sequence 2, 4, 6, 8, 10 -> 12"""
    
    # Input sequence: 2, 4, 6, 8, 10
    context = [2.0, 4.0, 6.0, 8.0, 10.0]
    
    print(f"Input sequence: {context}")
    print("Expected next value: ~12")
    
    try:
        # Load the Toto model
        print("\nLoading Toto model...")
        pipeline = TotoPipeline.from_pretrained()
        
        # Generate forecast for 1 step
        print("Generating forecast...")
        forecasts = pipeline.predict(
            context=context,
            prediction_length=1,
            num_samples=3072  # Optimal samples for best accuracy
        )
        
        # Get predictions
        samples = forecasts[0].numpy()  # Shape: (num_samples,) for prediction_length=1
        predicted_values = samples  # Already 1D array for single prediction step
        
        # Calculate statistics
        mean_pred = np.mean(predicted_values)
        median_pred = np.median(predicted_values)
        std_pred = np.std(predicted_values)
        
        print(f"\nResults:")
        print(f"Mean prediction: {mean_pred:.2f}")
        print(f"Median prediction: {median_pred:.2f}")
        print(f"Standard deviation: {std_pred:.2f}")
        print(f"Min prediction: {np.min(predicted_values):.2f}")
        print(f"Max prediction: {np.max(predicted_values):.2f}")
        
        # Check if prediction is close to expected value (12)
        expected = 12.0
        error = abs(mean_pred - expected)
        print(f"\nExpected: {expected}")
        print(f"Prediction error: {error:.2f}")
        
        if error < 2.0:  # Within 2 units
            print("✅ Test PASSED - Prediction is close to expected value")
        else:
            print("❌ Test FAILED - Prediction is far from expected value")
            
        return mean_pred, error < 2.0
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {e}")
        return None, False

if __name__ == "__main__":
    print("Testing Toto wrapper with arithmetic sequence...")
    test_arithmetic_sequence()