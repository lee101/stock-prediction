import os
import stat
from pathlib import Path

import pytest

from src.cache_utils import ensure_huggingface_cache_dir, find_hf_snapshot_dir


def _reset_permissions(path: Path) -> None:
    try:
        path.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    except PermissionError:
        # Best effort; some file systems may not support chmod adjustments.
        pass


def test_ensure_hf_cache_respects_existing_env(monkeypatch, tmp_path):
    desired = tmp_path / "hf_home"
    monkeypatch.setenv("HF_HOME", str(desired))
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    selected = ensure_huggingface_cache_dir()

    assert selected == desired.resolve()
    assert os.environ["HF_HOME"] == str(selected)
    assert desired.exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows does not reliably enforce chmod-based write restrictions.")
def test_ensure_hf_cache_falls_back_when_unwritable(monkeypatch, tmp_path):
    locked_parent = tmp_path / "locked_parent"
    locked_parent.mkdir()
    locked_parent.chmod(stat.S_IRUSR | stat.S_IXUSR)  # remove write permission

    problematic = locked_parent / "hf_home"
    monkeypatch.setenv("HF_HOME", str(problematic))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(problematic))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(problematic))

    home_dir = tmp_path / "alt_home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))

    fallback = ensure_huggingface_cache_dir()

    assert fallback != problematic.resolve()
    assert fallback.exists()
    assert os.access(fallback, os.W_OK)
    for env_key in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        assert os.environ[env_key] == str(fallback)

    _reset_permissions(locked_parent)


def test_find_hf_snapshot_dir_returns_latest_snapshot(tmp_path, monkeypatch):
    for env_key in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(env_key, raising=False)
    cache_root = tmp_path / "hf_cache"
    newest = cache_root / "hub" / "models--amazon--chronos-2" / "snapshots" / "newest"
    oldest = cache_root / "hub" / "models--amazon--chronos-2" / "snapshots" / "oldest"
    for path in (newest, oldest):
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
    os.utime(oldest, (oldest.stat().st_atime, oldest.stat().st_mtime - 100))

    found = find_hf_snapshot_dir("amazon/chronos-2", extra_candidates=[cache_root])

    assert found == newest


def test_find_hf_snapshot_dir_returns_none_when_missing(tmp_path, monkeypatch):
    for env_key in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(env_key, raising=False)
    cache_root = tmp_path / "hf_cache_missing"
    result = find_hf_snapshot_dir("amazon/chronos-2", extra_candidates=[cache_root])
    assert result is None
