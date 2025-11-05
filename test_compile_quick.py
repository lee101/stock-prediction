#!/usr/bin/env python3
"""
Quick test to verify Toto compilation warnings are fixed.
"""
import os
import sys
import torch

# Enable verbose logging
os.environ["TORCH_LOGS"] = "recompiles,cudagraphs"

# Make sure we use compiled version
os.environ["TOTO_COMPILE"] = "1"
os.environ["TOTO_COMPILE_MODE"] = "max-autotune"
os.environ["TOTO_COMPILE_BACKEND"] = "inductor"

print("=" * 80)
print("QUICK COMPILATION TEST")
print("=" * 80)
print()

# Import after setting env vars
from src.models.toto_wrapper import TotoPipeline

print("Loading Toto pipeline with torch.compile...")
pipeline = TotoPipeline.from_pretrained(
    model_id="Datadog/Toto-Open-Base-1.0",
    device_map="cuda",
    torch_dtype=torch.float32,
    torch_compile=True,
    compile_mode="max-autotune",
    compile_backend="inductor",
    warmup_sequence=0,
)

print("✓ Pipeline loaded")
print()

# Generate test data
print("Generating test forecast...")
context = torch.randn(512, dtype=torch.float32)

# Run inference
forecasts = pipeline.predict(
    context=context,
    prediction_length=8,
    num_samples=256,
    samples_per_batch=128,
)

print(f"✓ Forecast generated: {forecasts[0].samples.shape}")
print()

# Run again to trigger compilation
print("Running second inference (should use compiled version)...")
forecasts2 = pipeline.predict(
    context=context,
    prediction_length=8,
    num_samples=256,
    samples_per_batch=128,
)

print(f"✓ Second forecast generated: {forecasts2[0].samples.shape}")
print()

print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print()
print("Check above for:")
print("  ❌ 'skipping cudagraphs' warnings")
print("  ❌ 'recompile_limit' warnings")
print()
print("If no warnings appeared, the fix is working!")
