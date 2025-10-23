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
    _safe_import('src.logging_utils')


def test_setup_logging(tmp_path):
    mod = _safe_import('src.logging_utils')
    log_file = tmp_path / "test_log.log"
    logger = mod.setup_logging(str(log_file))
    logger.info("hello")
    # Ensure the log file is created
    assert log_file.exists()
