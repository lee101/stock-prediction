from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path
from typing import Iterable

import pytest

from falmarket.app import MarketSimulatorApp

_COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.so", "*.pyd")


def _copy_module(module_name: str, destination_root: Path) -> None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        pytest.fail(f"Module {module_name!r} is not importable in the development environment.")

    if spec.submodule_search_locations:
        source_dir = Path(next(iter(spec.submodule_search_locations)))
        target_dir = destination_root / module_name.replace(".", "/")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir, ignore=_COPY_IGNORE)
        return

    origin = spec.origin
    if not origin:
        # Built-in or extension module; nothing to copy because the runtime will supply it.
        return

    source_path = Path(origin)
    if not source_path.exists():
        pytest.fail(f"Module origin for {module_name!r} not found at {source_path}.")

    target_path = destination_root / f"{module_name.replace('.', '/')}{source_path.suffix}"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def _prepare_runtime_modules(modules: Iterable[str], root: Path) -> None:
    for name in modules:
        _copy_module(name, root)


def test_fal_runtime_imports_survive_isolated_environment(tmp_path: Path) -> None:
    vendored_root = tmp_path / "vendored"
    vendored_root.mkdir()

    _prepare_runtime_modules(MarketSimulatorApp.local_python_modules, vendored_root)

    modules_to_verify = [
        "fal_marketsimulator.runner",
        "marketsimulator.environment",
        "trade_stock_e2e",
        "alpaca_wrapper",
        "data_curate_daily",
        "env_real",
        "jsonshelve",
    ]

    bootstrap = textwrap.dedent(
        f"""
        import importlib
        modules = {modules_to_verify!r}
        for name in modules:
            importlib.import_module(name)
        print("imports-ok")
        """
    ).strip()

    site_packages = sysconfig.get_paths()["purelib"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(vendored_root), site_packages])

    completed = subprocess.run(
        [sys.executable, "-c", bootstrap],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, (
        f"Runtime import check failed with exit code {completed.returncode}.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert "imports-ok" in completed.stdout
