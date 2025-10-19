#!/usr/bin/env python3
"""Test GPU availability and fix common CUDA issues"""

import os
import sys
from typing import Iterable, Mapping

# Set library paths before importing torch
os.environ[
    "LD_LIBRARY_PATH"
] = "/home/lee/.pyenv/versions/3.12.7/lib/python3.12/site-packages/nvidia/nvjitlink/lib:" + os.environ.get(
    "LD_LIBRARY_PATH", ""
)

# Try different CUDA configurations
DEFAULT_CONFIGS = (
    {},  # Default
    {"CUDA_VISIBLE_DEVICES": "0"},  # Explicitly set device
    {"CUDA_LAUNCH_BLOCKING": "1"},  # Force synchronous
)


def check_gpu(configs_to_try: Iterable[Mapping[str, str]] = DEFAULT_CONFIGS) -> bool:
    """Run a quick GPU smoke test, returning True if CUDA succeeds."""
    success = False
    for i, env_vars in enumerate(configs_to_try):
        print(f"\n{'='*60}")
        print(f"Attempt {i + 1}: {env_vars if env_vars else 'Default config'}")
        print("=" * 60)

        # Set environment variables
        for key, val in env_vars.items():
            os.environ[key] = val

        try:
            import torch

            print(f"✓ PyTorch version: {torch.__version__}")
            print(f"✓ CUDA compiled: {torch.version.cuda}")
            print(f"✓ CUDA available: {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                print(f"✓ GPU count: {torch.cuda.device_count()}")
                print(f"✓ Current device: {torch.cuda.current_device()}")
                print(f"✓ Device name: {torch.cuda.get_device_name(0)}")
                total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"✓ Memory: {total_mem_gb:.1f} GB")

                # Test tensor operations
                print("\nTesting tensor operations...")
                x = torch.randn(100, 100).cuda()
                y = torch.randn(100, 100).cuda()
                z = torch.matmul(x, y)
                print(f"✓ Tensor operation successful: {z.shape}")

                print("\n✅ GPU is working properly!")
                success = True
                break
            else:
                print("✗ CUDA not available")

        except Exception as exc:  # noqa: BLE001
            print(f"✗ Error: {exc}")

    if not success:
        print("\n❌ Could not get GPU working. Falling back to CPU training.")
    return success


if __name__ == "__main__":
    sys.exit(0 if check_gpu() else 1)
