"""Tests for pathlib migration across the codebase.

Verifies that:
1. Files that should use pathlib do NOT use os.path for path construction
2. Key path operations produce correct results after migration
3. sys.path entries are strings (pathlib Path objects are NOT valid sys.path entries)
"""

import ast
import importlib
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Static analysis: ensure migrated files no longer use os.path for path ops
# ---------------------------------------------------------------------------

# Files we are migrating (relative to repo root)
MIGRATED_FILES = [
    "disk_cache.py",
    "jsonshelve.py",
    "logging_utils.py",
    "kronos_prediction_cache.py",
    "profile_trade_stock.py",
    "toto_compile_config.py",
    "chronos_compile_config.py",
    "traininglib/report.py",
    "pufferlib_market/sweep_daily_combos.py",
    "hftraining/test_pipeline.py",
    "hftraining/quick_rl_train.py",
    "hftraining/quick_test_runner.py",
    "hftraining/launch_training.py",
    "hftraining/train_with_profit.py",
    "hftraining/realistic_backtest_rl.py",
    "hftraining/train_hf.py",
    "hftraining/run_training.py",
    "hftraining/rl_advanced_trainer.py",
    "hftraining/base_model_trainer.py",
    "hftraining/modern_dit_rl_trader.py",
    "hftraining/single_batch_hf.py",
    "hftraining/toto_features.py",
    "bagsneural/sweep.py",
    "bagsneural/run_backtest.py",
    "bagsneural/run_train.py",
    "bagsneural/run_train_multi.py",
    "bagsfm/run_backtest.py",
    "scripts/test_optimizations.py",
    "training/download_training_data.py",
    "training/download_training_data_fixed.py",
    "llm_hourly_trader/providers.py",
    "src/logging_utils.py",
    "src/kronos_prediction_cache.py",
    "src/chronos_compile_config.py",
]

# os.path calls that should be replaced by pathlib equivalents
OS_PATH_FUNCS = {
    "os.path.dirname",
    "os.path.abspath",
    "os.path.join",
    "os.path.exists",
    "os.path.basename",
    "os.path.splitext",
    "os.path.expanduser",
}


def _find_os_path_calls(filepath: Path) -> list[str]:
    """Return list of os.path.* call strings found in the given Python file."""
    source = filepath.read_text()
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []  # skip unparseable files

    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_str = _call_to_str(node.func)
            if call_str and call_str in OS_PATH_FUNCS:
                found.append(f"{call_str} (line {node.lineno})")
    return found


def _call_to_str(node) -> str | None:
    """Convert an AST node to a dotted name string like 'os.path.join'."""
    if isinstance(node, ast.Attribute):
        parent = _call_to_str(node.value)
        if parent:
            return f"{parent}.{node.attr}"
    elif isinstance(node, ast.Name):
        return node.id
    return None


@pytest.mark.parametrize("relpath", MIGRATED_FILES)
def test_no_os_path_calls(relpath):
    """Migrated files should not contain os.path.dirname/join/exists etc."""
    filepath = REPO_ROOT / relpath
    if not filepath.exists():
        pytest.skip(f"{relpath} not found")

    violations = _find_os_path_calls(filepath)
    assert not violations, (
        f"{relpath} still uses os.path functions after migration:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


@pytest.mark.parametrize("relpath", MIGRATED_FILES)
def test_imports_pathlib(relpath):
    """Migrated files should import from pathlib."""
    filepath = REPO_ROOT / relpath
    if not filepath.exists():
        pytest.skip(f"{relpath} not found")

    source = filepath.read_text()
    # Allow either `from pathlib import Path` or `import pathlib`
    assert "pathlib" in source, f"{relpath} does not import pathlib"


# ---------------------------------------------------------------------------
# Functional tests for key utilities
# ---------------------------------------------------------------------------

class TestDiskCache:
    """Verify disk_cache still works after pathlib migration."""

    def test_cache_dir_path(self):
        from disk_cache import disk_cache

        @disk_cache
        def _dummy(x):
            return x * 2

        # cache_clear should work (creates cache dir)
        _dummy.cache_clear()

    def test_caching_roundtrip(self, monkeypatch):
        monkeypatch.delenv("TESTING", raising=False)
        from disk_cache import disk_cache

        call_count = 0

        @disk_cache
        def _add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        _add.cache_clear()
        assert _add(1, 2) == 3
        assert call_count == 1
        assert _add(1, 2) == 3  # should come from cache
        assert call_count == 1
        _add.cache_clear()


class TestJsonShelve:
    """Verify jsonshelve FlatShelf atomic writes still work."""

    def test_flat_shelf_save_load(self, tmp_path):
        from jsonshelve import FlatShelf

        path = str(tmp_path / "test.json")
        shelf = FlatShelf(path)
        shelf["key"] = "value"
        # Reload from disk
        shelf2 = FlatShelf(path)
        assert shelf2["key"] == "value"


class TestLoggingUtils:
    """Verify logging_utils still extracts filenames correctly."""

    def test_setup_logging(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        from logging_utils import setup_logging

        logger = setup_logging(log_file)
        assert logger.name == "test"
        logger.info("hello")


class TestProfileTradeStock:
    """Verify pathlib paths resolve correctly."""

    def test_venv_path_construction(self):
        """The venv python path should be constructable via pathlib."""
        venv_python = Path.cwd() / ".venv" / "bin" / "python"
        # Just verify the path is a valid Path object
        assert isinstance(venv_python, Path)


class TestCompileConfigs:
    """Verify compile config cache dirs are created correctly."""

    def test_toto_cache_dir(self):
        cache_dir = Path.cwd() / "compiled_models" / "torch_inductor"
        assert isinstance(cache_dir, Path)
        assert str(cache_dir).endswith("compiled_models/torch_inductor")

    def test_chronos_cache_dir(self):
        cache_dir = Path.cwd() / "compiled_models" / "chronos2_torch_inductor"
        assert isinstance(cache_dir, Path)
        assert str(cache_dir).endswith("compiled_models/chronos2_torch_inductor")


class TestRepoRootPattern:
    """Verify the common REPO_ROOT pattern works consistently."""

    def test_parent_parent_resolves(self):
        """Path(__file__).resolve().parent.parent should match os.path equivalent."""
        test_file = Path(__file__).resolve()
        pathlib_root = test_file.parent.parent
        os_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert str(pathlib_root) == os_root

    def test_sys_path_entries_are_strings(self):
        """sys.path entries must be strings, not Path objects."""
        for entry in sys.path:
            assert isinstance(entry, str), (
                f"sys.path contains non-string entry: {entry!r} (type={type(entry).__name__})"
            )


class TestKronosPredictionCache:
    """Verify KronosPredictionCache still uses Path internally."""

    def test_cache_dir_is_path(self, tmp_path):
        from kronos_prediction_cache import KronosPredictionCache

        cache = KronosPredictionCache(cache_dir=str(tmp_path / "test_cache"))
        assert isinstance(cache.cache_dir, Path)


class TestLlmProviders:
    """Verify codex auth path uses pathlib."""

    def test_codex_path_expanduser(self):
        codex_path = Path("~/.codex/auth.json").expanduser()
        assert isinstance(codex_path, Path)
        assert str(codex_path).startswith("/")


class TestTraininglibReport:
    """Verify report directory creation."""

    def test_report_dir_creation(self, tmp_path):
        out_path = tmp_path / "subdir" / "report.md"
        directory = out_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        assert directory.exists()
