"""Tests for profiling infrastructure.

Verifies that the profiling modules can be imported and their public API
is callable without actually running training.
"""
from __future__ import annotations

import sys
from pathlib import Path
import importlib

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

def test_pufferlib_profile_module_imports() -> None:
    """pufferlib_market.profile_training must import without error."""
    import pufferlib_market.profile_training as mod
    assert hasattr(mod, "run_pufferlib_profiling")
    assert hasattr(mod, "find_data_file")
    assert hasattr(mod, "main")


def test_neural_profile_module_imports() -> None:
    """binanceneural.profile_training must import without error."""
    import binanceneural.profile_training as mod
    assert hasattr(mod, "run_neural_profiling")
    assert hasattr(mod, "main")


# ---------------------------------------------------------------------------
# Helper function tests (no training)
# ---------------------------------------------------------------------------

def test_find_data_file_returns_existing_path() -> None:
    """find_data_file should return a path to an existing .bin file."""
    from pufferlib_market.profile_training import find_data_file
    path = find_data_file(PROJECT_ROOT)
    assert Path(path).exists(), f"Data file not found: {path}"
    assert path.endswith(".bin"), f"Expected .bin file, got: {path}"


def test_check_symbol_data_known_symbols() -> None:
    """_check_symbol_data should return True for at least one default symbol."""
    from binanceneural.profile_training import _check_symbol_data, DEFAULT_SYMBOLS
    available = [s for s in DEFAULT_SYMBOLS if _check_symbol_data(s)]
    assert len(available) > 0, (
        f"None of {DEFAULT_SYMBOLS} have data. "
        "Expected at least one to have both price CSV and h1 forecast cache."
    )


def test_select_symbols_filters_unavailable() -> None:
    """_select_symbols should return only symbols with data."""
    from binanceneural.profile_training import _select_symbols, DEFAULT_SYMBOLS
    result = _select_symbols(DEFAULT_SYMBOLS)
    assert len(result) > 0
    # All returned symbols must actually exist
    from binanceneural.profile_training import _check_symbol_data
    for sym in result:
        assert _check_symbol_data(sym), f"Symbol {sym} returned but data not available"


def test_select_symbols_raises_on_all_missing() -> None:
    """_select_symbols should raise FileNotFoundError when no symbol has data."""
    from binanceneural.profile_training import _select_symbols
    with pytest.raises(FileNotFoundError):
        _select_symbols(["FAKESYMBOL_DOES_NOT_EXIST_XYZ"])


def test_pyspy_finder_returns_none_or_str() -> None:
    """_find_pyspy returns either None or a non-empty string path."""
    from pufferlib_market.profile_training import _find_pyspy
    result = _find_pyspy()
    assert result is None or (isinstance(result, str) and len(result) > 0)


def test_profiles_dir_constant() -> None:
    """PROFILES_DIR should point to the profiles/ directory under project root."""
    from pufferlib_market.profile_training import PROFILES_DIR as pf_dir
    from binanceneural.profile_training import PROFILES_DIR as nn_dir
    assert pf_dir == PROJECT_ROOT / "profiles"
    assert nn_dir == PROJECT_ROOT / "profiles"


def test_pufferlib_script_is_executable() -> None:
    """profile_training.py scripts should be readable source files."""
    pf_script = PROJECT_ROOT / "pufferlib_market" / "profile_training.py"
    nn_script = PROJECT_ROOT / "binanceneural" / "profile_training.py"
    assert pf_script.exists(), f"Missing: {pf_script}"
    assert nn_script.exists(), f"Missing: {nn_script}"
    # Must be valid Python (parse check)
    import ast
    for script in [pf_script, nn_script]:
        source = script.read_text()
        ast.parse(source)  # raises SyntaxError if invalid


def test_profile_all_sh_exists() -> None:
    """scripts/profile_all.sh must exist."""
    sh = PROJECT_ROOT / "scripts" / "profile_all.sh"
    assert sh.exists(), f"Missing: {sh}"
    content = sh.read_text()
    assert "pufferlib_market/profile_training.py" in content
    assert "binanceneural/profile_training.py" in content


def test_profiles_in_gitignore() -> None:
    """profiles/ directory should be in .gitignore."""
    gitignore = PROJECT_ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore not found"
    content = gitignore.read_text()
    assert "profiles/" in content, "profiles/ not found in .gitignore"
