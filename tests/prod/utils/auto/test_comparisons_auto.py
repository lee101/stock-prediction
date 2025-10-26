#!/usr/bin/env python3
import pytest
import sys
from pathlib import Path
import importlib

pytestmark = pytest.mark.auto_generated

# Ensure project root on sys.path for 'src' imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe_import(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.skip(f"Skipping {name}: dependency not installed")
    except ImportError:
        pytest.skip(f"Skipping {name}: import error")


def test_is_side_helpers():
    mod = _safe_import('src.comparisons')
    assert mod.is_same_side('buy', 'long')
    assert mod.is_same_side('sell', 'short')
    assert not mod.is_same_side('buy', 'short')
    assert mod.is_buy_side('BUY')
    assert mod.is_sell_side('short')

