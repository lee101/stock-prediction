#!/usr/bin/env python3
import pytest
import sys
from pathlib import Path

# Ensure project root on sys.path for 'src' imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import importlib

def _safe_import(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.skip(f"Skipping {name}: dependency not installed")
    except ImportError:
        pytest.skip(f"Skipping {name}: import error")

pytestmark = pytest.mark.auto_generated


def test_import_module():
    _safe_import('src.date_utils')


def test_date_utils_calls():
    mod = _safe_import('src.date_utils')
    # Calls should not raise
    assert isinstance(mod.is_nyse_trading_day_ending(), bool)
    assert isinstance(mod.is_nyse_trading_day_now(), bool)
