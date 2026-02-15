from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def test_ensure_compilation_artifacts_normalises_cache_paths(monkeypatch, tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMPILED_MODELS_DIR", "cache_root")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "cache_root/torch_inductor_rel")
    # Ensure we re-import the module under the test-specific env/cwd, but restore
    # sys.modules afterwards so other tests can safely reload the module.
    monkeypatch.delitem(sys.modules, "backtest_test3_inline", raising=False)

    module = importlib.import_module("backtest_test3_inline")
    module._ensure_compilation_artifacts()

    compiled_env = Path(os.environ["COMPILED_MODELS_DIR"])
    cache_env = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])

    assert module.COMPILED_MODELS_DIR.is_absolute()
    assert module.INDUCTOR_CACHE_DIR.is_absolute()
    assert compiled_env == module.COMPILED_MODELS_DIR
    assert compiled_env.exists()
    assert (compiled_env / "torch_inductor").exists()
    assert cache_env.is_absolute()
    assert str(cache_env).endswith("cache_root/torch_inductor_rel")
