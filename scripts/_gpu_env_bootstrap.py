from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_ENV_PYTHON = REPO_ROOT / "gpu_trading_env" / "python"
GPU_ENV_PACKAGE = GPU_ENV_PYTHON / "gpu_trading_env"


def ensure_gpu_trading_env(*, sys_module: ModuleType | Any = sys) -> Path:
    """Make the in-repo gpu_trading_env package importable.

    Pytest can pre-import the repo-root ``gpu_trading_env`` namespace before a
    script adds ``gpu_trading_env/python``. In that state, submodule imports may
    resolve but package-level constants from the real ``__init__`` never load.
    Drop only that incomplete namespace so the real package initializes.
    """
    if str(GPU_ENV_PYTHON) not in sys_module.path:
        sys_module.path.insert(0, str(GPU_ENV_PYTHON))

    loaded_gpu_env = sys_module.modules.get("gpu_trading_env")
    loaded_gpu_env_path = getattr(loaded_gpu_env, "__path__", None)
    if loaded_gpu_env is not None and not hasattr(loaded_gpu_env, "PRODUCTION_FEE_BPS"):
        sys_module.modules.pop("gpu_trading_env", None)
    elif loaded_gpu_env_path is not None and str(GPU_ENV_PACKAGE) not in list(loaded_gpu_env_path):
        loaded_gpu_env_path.append(str(GPU_ENV_PACKAGE))

    return GPU_ENV_PYTHON
