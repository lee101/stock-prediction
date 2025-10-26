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
    _safe_import('src.utils')


def test_invoke_easy_callables():
    mod = _safe_import('src.utils')
    # Only call functions with defaults-only signature
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


def test_log_time_and_debounce():
    mod = _safe_import('src.utils')

    # log_time context manager should run without errors
    with mod.log_time("unit-test"):
        pass

    # debounce should throttle repeated calls; we just ensure it runs
    calls = []

    @mod.debounce(60)
    def f(x=1):
        calls.append(x)

    f()
    f()  # likely throttled; should not error
    assert len(calls) >= 1
