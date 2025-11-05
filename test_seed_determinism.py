#!/usr/bin/env python3
"""
Test if torch.compile is deterministic when using same seed.

This is a simpler test that loads one pipeline at a time to avoid
resource exhaustion.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

# Test parameters
SYMBOL = "BTCUSD"
SEEDS = [42, 123]
NUM_SAMPLES = 1024

# Load test data
csv_path = Path("trainingdata") / f"{SYMBOL}.csv"
df = pd.read_csv(csv_path)
prices = df['close'].values[-512:].astype(np.float32)
context = torch.from_numpy(prices)

def test_pipeline(compiled: bool, seed: int):
    """Test a single pipeline with a seed."""
    if compiled:
        import toto_compile_config
        toto_compile_config.apply(verbose=False)
    
    from src.models.toto_wrapper import TotoPipeline
    
    # Load pipeline
    pipeline = TotoPipeline.from_pretrained(
        "Datadog/Toto-Open-Base-1.0",
        device_map="cuda",
        torch_compile=compiled,
    )
    
    # Set seed
    transformers.set_seed(seed)
    
    # Predict
    forecast = pipeline.predict(
        context=context,
        prediction_length=8,
        num_samples=NUM_SAMPLES,
        samples_per_batch=128,
    )
    
    samples = forecast[0].numpy()
    mae = np.mean(np.abs(samples))
    
    # Cleanup
    del pipeline
    torch.cuda.empty_cache()
    
    return mae, samples

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "uncompiled"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    
    compiled = (mode == "compiled")
    mae, samples = test_pipeline(compiled, seed)
    
    # Save results
    result_file = f"/tmp/seed_test_{mode}_seed{seed}.npz"
    np.savez(result_file, mae=mae, samples=samples)
    
    print(f"{mode.upper()} Seed {seed}: MAE = {mae:.6f}")
