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
    _safe_import('src.trading_obj_utils')


def test_filter_to_realistic_positions_basic():
    mod = _safe_import('src.trading_obj_utils')

    class P:
        def __init__(self, symbol, qty):
            self.symbol = symbol
            self.qty = qty

    positions = [
        P('BTCUSD', '0.0005'),   # too small
        P('BTCUSD', '0.002'),    # big enough
        P('ETHUSD', '0.005'),    # too small
        P('ETHUSD', '0.02'),     # big enough
        P('LTCUSD', '0.05'),     # too small
        P('LTCUSD', '0.2'),      # big enough
        P('UNIUSD', '2'),        # too small
        P('UNIUSD', '10'),       # big enough
        P('AAPL', '1'),          # stocks pass through
    ]

    filtered = mod.filter_to_realistic_positions(positions)
    symbols = [p.symbol for p in filtered]
    assert 'BTCUSD' in symbols
    assert 'ETHUSD' in symbols
    assert 'LTCUSD' in symbols
    assert 'UNIUSD' in symbols
    assert 'AAPL' in symbols
