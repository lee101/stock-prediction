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
# Try system PufferLib first, fall back to local checkout
_local_ocean = REPO / "PufferLib" / "pufferlib" / "ocean"
try:
    import pufferlib
    _system_ocean = Path(pufferlib.__file__).parent / "ocean"
except ImportError:
    _system_ocean = None


def _has_env_binding_header(path: Path | None) -> bool:
    return path is not None and (path / "env_binding.h").exists()


if _has_env_binding_header(_system_ocean):
    PUFFERLIB_OCEAN = _system_ocean
else:
    PUFFERLIB_OCEAN = _local_ocean

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
        "-O3", "-march=native", "-ffast-math",
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
