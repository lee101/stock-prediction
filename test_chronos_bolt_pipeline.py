#!/usr/bin/env python3
"""
Test script to reproduce the ChronosBoltPipeline.predict unexpected num_samples error
"""
import torch
import numpy as np
from chronos import BaseChronosPipeline


def test_chronos_bolt_pipeline():
    """Test that demonstrates the num_samples parameter issue with ChronosBoltPipeline"""
    
    # Load the Chronos Bolt pipeline (this creates a ChronosBoltPipeline)
    pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map="cuda",
    )
    
    # Create test context data
    context = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    prediction_length = 3
    
    print(f"Pipeline type: {type(pipeline)}")
    print(f"Pipeline class name: {pipeline.__class__.__name__}")
    
    # Test 1: Call predict without num_samples (should work)
    print("\nTest 1: Calling predict without num_samples...")
    try:
        forecast1 = pipeline.predict(context, prediction_length)
        print(f"✓ Success! Forecast shape: {forecast1[0].numpy().shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 2: Call predict with num_samples (should fail)
    print("\nTest 2: Calling predict with num_samples=20...")
    try:
        forecast2 = pipeline.predict(
            context,
            prediction_length,
            num_samples=20,
            temperature=1.0,
            top_k=4000,
            top_p=1.0,
        )
        print(f"✓ Success! Forecast shape: {forecast2[0].numpy().shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        
    # Test 3: Check what parameters the predict method actually accepts
    print("\nTest 3: Checking predict method signature...")
    import inspect
    sig = inspect.signature(pipeline.predict)
    print(f"Predict method parameters: {list(sig.parameters.keys())}")


if __name__ == "__main__":
    test_chronos_bolt_pipeline()