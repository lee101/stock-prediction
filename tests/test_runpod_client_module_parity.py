from __future__ import annotations

import importlib
import runpy
from pathlib import Path


def test_runpod_client_root_module_aliases_src_module():
    root_module = importlib.import_module("runpod_client")
    src_module = importlib.import_module("src.runpod_client")

    assert root_module is src_module
    assert root_module.RunPodClient is src_module.RunPodClient
    assert root_module.PodConfig is src_module.PodConfig
    assert root_module.Pod is src_module.Pod


def test_runpod_client_root_and_src_share_live_constants():
    root_module = importlib.import_module("runpod_client")
    src_module = importlib.import_module("src.runpod_client")

    assert root_module.HOURLY_RATES is src_module.HOURLY_RATES
    assert root_module.GPU_ALIASES is src_module.GPU_ALIASES
    assert root_module.DEFAULT_GPU_FALLBACKS is src_module.DEFAULT_GPU_FALLBACKS


def test_runpod_client_script_mode_delegates_to_src_module(monkeypatch):
    called: dict[str, object] = {}

    def fake_run_module(name: str, *, run_name: str) -> dict[str, object]:
        called["name"] = name
        called["run_name"] = run_name
        return {}

    monkeypatch.setattr(runpy, "run_module", fake_run_module)

    runpy.run_path(
        str(Path(__file__).resolve().parent.parent / "runpod_client.py"),
        run_name="__main__",
    )

    assert called == {"name": "src.runpod_client", "run_name": "__main__"}
