#!/usr/bin/env python3
"""Pytest configuration for environments with real PyTorch installed."""

import os
import pytest

# Allow skipping the hard PyTorch requirement for lightweight coverage runs.
if os.getenv("SKIP_TORCH_CHECK", "0") not in ("1", "true", "TRUE", "yes", "YES"):
    # Ensure PyTorch is available; fail fast if not.
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "PyTorch must be installed for this test suite."
        ) from e
