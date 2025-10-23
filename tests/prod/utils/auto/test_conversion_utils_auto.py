#!/usr/bin/env python3
import pytest
import sys
from pathlib import Path
import importlib
import types

pytestmark = pytest.mark.auto_generated

# Ensure project root on sys.path for 'src' imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DummyTensor:
    def __init__(self, dims, data):
        self._dims = dims
        self._data = data

    def dim(self):
        return self._dims

    def tolist(self):
        return self._data

    def __float__(self):
        # mimic scalar tensor conversion
        return float(self._data)


def test_conversion_utils_with_mock_torch():
    # Inject a minimal mock torch into sys.modules
    sys.modules['torch'] = types.SimpleNamespace(Tensor=DummyTensor)
    mod = importlib.import_module('src.conversion_utils')

    # Scalar tensor unwraps to float
    val = mod.unwrap_tensor(DummyTensor(0, 3.14))
    assert isinstance(val, float)

    # 1D tensor unwraps to list
    arr = mod.unwrap_tensor(DummyTensor(1, [1, 2, 3]))
    assert arr == [1, 2, 3]

    # Non-tensor returns as-is
    assert mod.unwrap_tensor({'a': 1}) == {'a': 1}

    # String to datetime conversion
    dt = mod.convert_string_to_datetime('2024-04-16T19:53:01.577838')
    assert dt.year == 2024
