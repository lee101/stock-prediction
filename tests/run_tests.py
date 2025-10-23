#!/usr/bin/env python3
"""Simple test runner that requires a real PyTorch installation."""

import sys
from pathlib import Path

import pytest


def _ensure_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "PyTorch must be installed for this test suite."
        ) from e


if __name__ == "__main__":
    _ensure_torch()

    test_files = [
        "tests/experimental/hf/test_hfinference_comprehensive.py",
        "tests/experimental/hf/test_hftraining_comprehensive.py",
        "tests/experimental/hf/test_hfinference_engine_sim.py",
        "tests/experimental/hf/test_hftraining_data_utils.py",
        "tests/experimental/hf/test_hftraining_model.py",
        "tests/experimental/hf/test_hftraining_training.py",
    ]

    existing_tests = [f for f in test_files if Path(f).exists()]

    print(f"\nRunning {len(existing_tests)} test files...")
    for test in existing_tests:
        print(f"  - {test}")

    exit_code = pytest.main(["-v", "--tb=short"] + existing_tests)
    print(f"\nTests completed with exit code: {exit_code}")
    sys.exit(exit_code)
