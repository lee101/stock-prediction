#!/usr/bin/env python3
"""Pytest configuration file that handles torch import issues."""

import sys
import os
from pathlib import Path

# Try to import torch, if it fails use mock
def setup_torch_mock():
    try:
        import torch
        if not hasattr(torch, '__version__'):
            raise AttributeError("Torch not properly installed")
        print(f"Using real torch: {torch.__version__}")
        return False
    except (ImportError, AttributeError) as e:
        print(f"Torch import failed ({e}), setting up mock...")
        
        # Add tests directory to path
        tests_dir = Path(__file__).parent
        sys.path.insert(0, str(tests_dir))
        
        # Import and setup mock
        import mock_torch
        sys.modules['torch'] = mock_torch
        sys.modules['torch.nn'] = mock_torch.nn
        sys.modules['torch.nn.functional'] = mock_torch.functional  # Add F functions
        sys.modules['torch.optim'] = mock_torch.optim
        sys.modules['torch.cuda'] = mock_torch.cuda
        sys.modules['torch.utils'] = mock_torch.utils
        sys.modules['torch.utils.data'] = mock_torch.utils.data
        sys.modules['torch.utils.tensorboard'] = mock_torch.utils.tensorboard
        
        print("Mock torch modules installed")
        return True

# Setup mock before any tests import torch
USING_MOCK = setup_torch_mock()

# Additional pytest configuration
import pytest

def pytest_configure(config):
    """Configure pytest."""
    if USING_MOCK:
        config.addinivalue_line(
            "markers", "requires_real_torch: mark test as requiring real torch"
        )

def pytest_collection_modifyitems(config, items):
    """Skip tests that require real torch when using mock."""
    if USING_MOCK:
        skip_real_torch = pytest.mark.skip(reason="requires real torch installation")
        for item in items:
            if "requires_real_torch" in item.keywords:
                item.add_marker(skip_real_torch)