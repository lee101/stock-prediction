from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

from scripts._gpu_env_bootstrap import ensure_gpu_trading_env


REPO_ROOT = Path(__file__).resolve().parents[1]


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def test_bench_portfolio_ppo_help_imports_gpu_env_without_pythonpath() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/bench_portfolio_ppo.py", "--help"],
        cwd=REPO_ROOT,
        env=_clean_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--rollout-steps" in result.stdout


def test_check_portfolio_env_random_imports_gpu_env_without_pythonpath() -> None:
    code = (
        "import importlib.util, pathlib; "
        "path = pathlib.Path('scripts/check_portfolio_env_random.py'); "
        "spec = importlib.util.spec_from_file_location('check_portfolio_env_random', path); "
        "mod = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(mod); "
        "assert hasattr(mod.gpu_trading_env, 'PRODUCTION_FEE_BPS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=_clean_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_bracket_scripts_recover_preimported_gpu_env_namespace() -> None:
    code = (
        "import gpu_trading_env; "
        "import scripts.train_portfolio_bracket_real as train_mod; "
        "import importlib.util, pathlib; "
        "path = pathlib.Path('scripts/bench_portfolio_ppo.py'); "
        "spec = importlib.util.spec_from_file_location('bench_portfolio_ppo', path); "
        "bench_mod = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(bench_mod); "
        "assert hasattr(train_mod.gpu_trading_env, 'PRODUCTION_FEE_BPS'); "
        "assert hasattr(bench_mod.gpu_trading_env, 'PRODUCTION_FEE_BPS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=_clean_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_bracket_scripts_use_shared_gpu_env_bootstrap() -> None:
    for path in (
        REPO_ROOT / "scripts" / "bench_portfolio_ppo.py",
        REPO_ROOT / "scripts" / "check_portfolio_env_random.py",
        REPO_ROOT / "scripts" / "train_portfolio_bracket_real.py",
    ):
        text = path.read_text()
        assert "ensure_gpu_trading_env" in text
        assert "loaded_gpu_env" not in text
        assert "GPU_ENV_PACKAGE" not in text


def test_gpu_env_bootstrap_replaces_incomplete_namespace() -> None:
    fake_sys = types.SimpleNamespace(path=[], modules={})
    fake_sys.modules["gpu_trading_env"] = types.SimpleNamespace(__path__=[])

    gpu_env_python = ensure_gpu_trading_env(sys_module=fake_sys)

    assert str(gpu_env_python) in fake_sys.path
    assert "gpu_trading_env" not in fake_sys.modules


def test_gpu_env_bootstrap_extends_loaded_package_path() -> None:
    loaded = types.SimpleNamespace(PRODUCTION_FEE_BPS=10.0, __path__=[])
    fake_sys = types.SimpleNamespace(path=[], modules={"gpu_trading_env": loaded})

    gpu_env_python = ensure_gpu_trading_env(sys_module=fake_sys)

    assert str(gpu_env_python) in fake_sys.path
    assert fake_sys.modules["gpu_trading_env"] is loaded
    assert str(gpu_env_python / "gpu_trading_env") in loaded.__path__
