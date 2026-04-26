from __future__ import annotations

import importlib
import inspect
import runpy
import subprocess
import sys
from pathlib import Path


def test_root_hourly_replay_aliases_package_module() -> None:
    package_module = importlib.import_module("pufferlib_market.hourly_replay")
    sys.modules.pop("hourly_replay", None)
    root_module = importlib.import_module("hourly_replay")

    assert root_module is package_module
    assert sys.modules["hourly_replay"] is package_module
    assert inspect.signature(root_module.simulate_daily_policy) == inspect.signature(
        package_module.simulate_daily_policy
    )


def test_hourly_replay_alias_is_stable_in_fresh_interpreters() -> None:
    code = """
from hourly_replay import simulate_daily_policy
import hourly_replay
import pufferlib_market.hourly_replay as package_module
assert hourly_replay is package_module
assert simulate_daily_policy is package_module.simulate_daily_policy
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_hourly_replay_shim_exposes_public_api_under_runpy() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    package_module = importlib.import_module("pufferlib_market.hourly_replay")

    namespace = runpy.run_path(str(repo_root / "hourly_replay.py"))

    assert namespace["simulate_daily_policy"] is package_module.simulate_daily_policy
    assert namespace["MktdData"] is package_module.MktdData

    code = """
import pufferlib_market.hourly_replay as package_module
import hourly_replay
assert hourly_replay is package_module
"""
    subprocess.run([sys.executable, "-c", code], check=True)
