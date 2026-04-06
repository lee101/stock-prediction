from __future__ import annotations

import importlib
import runpy
from pathlib import Path


def test_gpu_pool_rl_root_module_aliases_package_module():
    root_module = importlib.import_module("gpu_pool_rl")
    package_module = importlib.import_module("pufferlib_market.gpu_pool_rl")

    assert root_module is package_module
    assert root_module.DEFAULT_POOL_LIMITS is package_module.DEFAULT_POOL_LIMITS
    assert root_module.HOURLY_RATES is package_module.HOURLY_RATES
    assert root_module.GPU_ALIASES is package_module.GPU_ALIASES
    assert root_module.estimate_cost is package_module.estimate_cost


def test_gpu_pool_rl_script_mode_delegates_to_package_module(monkeypatch):
    called: dict[str, object] = {}

    def fake_run_module(name: str, *, run_name: str) -> dict[str, object]:
        called["name"] = name
        called["run_name"] = run_name
        return {}

    monkeypatch.setattr(runpy, "run_module", fake_run_module)

    runpy.run_path(
        str(Path(__file__).resolve().parent.parent / "gpu_pool_rl.py"),
        run_name="__main__",
    )

    assert called == {"name": "pufferlib_market.gpu_pool_rl", "run_name": "__main__"}
