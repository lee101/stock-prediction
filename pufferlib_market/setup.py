"""
Build script for the C trading environment PufferLib binding.

Usage:
    cd pufferlib_market && python setup.py build_ext --inplace
    # or from repo root:
    python pufferlib_market/setup.py build_ext --inplace
"""

import os
import sys
from pathlib import Path
from setuptools import setup, Extension

# Paths
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
_local_ocean_candidates = [
    REPO / "PufferLib" / "ocean",
    REPO / "PufferLib" / "pufferlib" / "ocean",
]


def _has_env_binding_header(path: Path | None) -> bool:
    return path is not None and (path / "env_binding.h").exists()


def _candidate_ocean_dirs() -> list[Path]:
    candidates: list[Path] = []

    env_override = os.environ.get("PUFFERLIB_OCEAN_DIR")
    if env_override:
        candidates.append(Path(env_override).expanduser())

    candidates.extend(_local_ocean_candidates)

    try:
        import pufferlib
        candidates.append(Path(pufferlib.__file__).resolve().parent / "ocean")
    except ImportError:
        pass

    for venv_dir in sorted(REPO.glob(".venv*/lib/python*/site-packages/pufferlib/ocean")):
        candidates.append(venv_dir.resolve())

    try:
        import site
        for site_dir in site.getsitepackages():
            candidates.append(Path(site_dir) / "pufferlib" / "ocean")
    except Exception:
        pass

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


for candidate in _candidate_ocean_dirs():
    if _has_env_binding_header(candidate):
        PUFFERLIB_OCEAN = candidate
        break
else:
    raise FileNotFoundError(
        "Could not locate pufferlib/ocean/env_binding.h. "
        "Set PUFFERLIB_OCEAN_DIR or install pufferlib in an active/local venv."
    )

# Find numpy include
import numpy as np

ext = Extension(
    "pufferlib_market.binding",
    sources=[
        str(ROOT / "src" / "binding.c"),
        str(ROOT / "src" / "trading_env.c"),
    ],
    include_dirs=[
        str(ROOT / "include"),
        str(PUFFERLIB_OCEAN),
        np.get_include(),
    ],
    extra_compile_args=[
        "-O3", "-march=native", "-mavx2", "-ffast-math",
        "-funroll-loops",
        "-fomit-frame-pointer",
        "-DNDEBUG",
        "-std=c11",
        "-Wno-unused-function",
    ],
    language="c",
)

setup(
    name="pufferlib_market",
    version="0.1.0",
    ext_modules=[ext],
)
