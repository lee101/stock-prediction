from __future__ import annotations

import importlib
import sys


def test_cache_utils_root_module_aliases_src_module():
    sys.modules.pop("cache_utils", None)
    sys.modules.pop("src.cache_utils", None)

    root_module = importlib.import_module("cache_utils")
    src_module = importlib.import_module("src.cache_utils")

    assert root_module is src_module


def test_cache_utils_root_and_src_share_live_behavior(tmp_path, monkeypatch):
    sys.modules.pop("cache_utils", None)
    sys.modules.pop("src.cache_utils", None)

    root_module = importlib.import_module("cache_utils")
    src_module = importlib.import_module("src.cache_utils")

    for env_key in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(env_key, raising=False)

    env_cache = tmp_path / "env-cache"
    monkeypatch.setenv("HF_HOME", str(env_cache))
    newest = tmp_path / "extra-cache" / "hub" / "models--amazon--chronos-2" / "snapshots" / "newest"
    newest.mkdir(parents=True, exist_ok=True)
    (newest / "config.json").write_text("{}", encoding="utf-8")

    found = root_module.find_hf_snapshot_dir(
        "amazon/chronos-2",
        extra_candidates=[tmp_path / "extra-cache"],
    )

    assert found == newest
    assert root_module.find_hf_snapshot_dir is src_module.find_hf_snapshot_dir
