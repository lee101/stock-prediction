import torch
import numpy as np
from src.forecasting_bolt_wrapper import ForecastingBoltWrapper

def test_simple_sequence():
    """Test with simple increasing sequence: 2, 4, 6, 8, 10 -> should predict ~12"""
    wrapper = ForecastingBoltWrapper()
    
    # Simple test sequence
    test_data = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0], dtype=torch.float)
    
    # Single prediction
    prediction = wrapper.predict_single(test_data, prediction_length=1)
    print(f"Input sequence: {test_data.tolist()}")
    print(f"Single prediction: {prediction}")
    print(f"Expected ~12, got {prediction}")
    
    # Sequence predictions
    predictions = wrapper.predict_sequence(test_data, prediction_length=3)
    print(f"Sequence predictions (3 steps): {predictions}")
    
    return prediction, predictions

if __name__ == "__main__":
    test_simple_sequence()