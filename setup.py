"""Root build script for editable installs that need the C trading binding.

This mirrors `pufferlib_market/setup.py`, but resolves paths from the repo
root so `uv pip install -e .` can build `pufferlib_market.binding`.
"""

from pathlib import Path
from setuptools import setup, Extension

# Paths.
REPO = Path(__file__).resolve().parent
ROOT = REPO / "pufferlib_market"
ROOT_REL = Path("pufferlib_market")

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
        str(ROOT_REL / "src" / "binding.c"),
        str(ROOT_REL / "src" / "trading_env.c"),
    ],
    include_dirs=[
        str(ROOT_REL / "include"),
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
