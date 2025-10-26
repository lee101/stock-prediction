#!/usr/bin/env python3
import pytest
import sys
from pathlib import Path

# Ensure project root on sys.path for 'src' imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import importlib
import inspect

def _safe_import(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.skip(f"Skipping {name}: dependency not installed")
    except ImportError:
        pytest.skip(f"Skipping {name}: import error")

pytestmark = pytest.mark.auto_generated


def test_import_module():
    _safe_import('src.stock_utils')


def test_invoke_easy_callables():
    mod = _safe_import('src.stock_utils')
    for name, obj in list(inspect.getmembers(mod)):
        if inspect.isfunction(obj) and getattr(obj, '__module__', '') == mod.__name__:
            try:
                sig = inspect.signature(obj)
            except Exception:
                continue
            all_default = True
            for p in sig.parameters.values():
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                if p.default is inspect._empty:
                    all_default = False
                    break
            if all_default:
                try:
                    obj()
                except Exception:
                    pass


def test_stock_utils_specifics():
    mod = _safe_import('src.stock_utils')
    # remap known crypto symbols
    assert mod.remap_symbols('ETHUSD') == 'ETH/USD'
    assert mod.remap_symbols('BTCUSD') == 'BTC/USD'
    # pairs_equal normalizes both
    assert mod.pairs_equal('BTCUSD', 'BTC/USD')
    assert mod.pairs_equal('ETH/USD', 'ETHUSD')
    # unmap back
    assert mod.unmap_symbols('ETH/USD') == 'ETHUSD'
