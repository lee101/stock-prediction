from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_MODULE = REPO_ROOT / "daily_mixed_hybrid.py"
SRC_MODULE = REPO_ROOT / "src" / "daily_mixed_hybrid.py"


def test_daily_mixed_hybrid_root_and_src_modules_stay_in_sync() -> None:
    assert ROOT_MODULE.read_text(encoding="utf-8") == SRC_MODULE.read_text(encoding="utf-8"), (
        "daily_mixed_hybrid.py drifted between the repo root and src/"
    )
