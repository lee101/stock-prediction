#!/usr/bin/env python3
"""Diagnose torch import issues."""

import sys
import importlib

print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nChecking torch import...")
try:
    # Try to find where torch is coming from
    import torch
    print(f"torch imported from: {torch.__file__ if hasattr(torch, '__file__') else 'Unknown'}")
    print(f"torch attributes: {dir(torch)[:10]}")
    print(f"Has nn? {hasattr(torch, 'nn')}")
    if hasattr(torch, 'nn'):
        print(f"nn attributes: {dir(torch.nn)[:10]}")
except Exception as e:
    print(f"Error importing torch: {e}")

print("\nChecking sys.modules for mock entries...")
for key in sys.modules:
    if 'torch' in key.lower() or 'mock' in key.lower():
        mod = sys.modules[key]
        if hasattr(mod, '__file__'):
            print(f"  {key}: {mod.__file__}")
        else:
            print(f"  {key}: {mod}")

print("\nTrying clean import...")
# Remove any torch-related modules
torch_keys = [k for k in sys.modules.keys() if 'torch' in k.lower()]
for k in torch_keys:
    del sys.modules[k]

# Try importing again
try:
    import torch
    print(f"Clean torch import successful")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"torch.nn.Module exists: {hasattr(torch.nn, 'Module')}")
except Exception as e:
    print(f"Clean import failed: {e}")