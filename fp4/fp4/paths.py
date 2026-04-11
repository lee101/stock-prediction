"""Canonical paths for the repo: REPO_ROOT, LOCAL_TMP, ensure_tmp().

Every module that needs a writable scratch directory should import LOCAL_TMP
from here instead of hard-coding /tmp.  LOCAL_TMP lives under <repo>/tmp/ so
CUDA build artefacts, Triton caches, and other ephemeral data stay on the
local NVMe rather than filling a sandboxed or RAM-backed /tmp.
"""
from __future__ import annotations

from pathlib import Path

# fp4/fp4/paths.py  ->  fp4/fp4  ->  fp4  ->  repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

LOCAL_TMP: Path = REPO_ROOT / "tmp"


def ensure_tmp() -> Path:
    """Create LOCAL_TMP if it doesn't exist and return it."""
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    return LOCAL_TMP
