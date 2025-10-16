#!/usr/bin/env python3
"""
Test script to verify that the ChronosBoltPipeline fix works
"""
import torch
import numpy as np
from chronos import BaseChronosPipeline


def test_chronos_bolt_fix():
    """Test that demonstrates the fix for ChronosBoltPipeline.predict"""
    
    # Load the Chronos Bolt pipeline (this creates a ChronosBoltPipeline)
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map="cuda",
    )
    
    # Create test context data
    context = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    prediction_length = 1
    
    print(f"Pipeline type: {type(pipeline)}")
    print(f"Pipeline class name: {pipeline.__class__.__name__}")
    
    # Test the fixed predict call (should work now)
    print("\nTest: Calling predict with only supported parameters...")
    try:
        forecast = pipeline.predict(
            context,
            prediction_length,
        )
        print(f"✓ Success! Forecast shape: {forecast[0].numpy().shape}")
        
        # Process the forecast the same way as the original code
        tensor = forecast[0]
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = np.asarray(tensor)
        low, median, high = np.quantile(tensor, [0.1, 0.5, 0.9], axis=0)
        print(f"✓ Successfully processed forecast: low={low}, median={median}, high={high}")
        
        # Check that we can get the median value as item (as done in original code)
        prediction_value = median.item()
        print(f"✓ Extracted prediction value: {prediction_value}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = test_chronos_bolt_fix()
    if success:
        print("\n✓ All tests passed! The fix should work.")
    else:
        print("\n✗ Tests failed!")
