from __future__ import annotations

from pathlib import Path
from setuptools import Extension, setup
import numpy as np


def main() -> None:
    here = Path(__file__).parent
    ext = Extension(
        name="rlinc_cmarket",
        sources=["_cmarket.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fno-math-errno"],
    )

    setup(
        ext_modules=[ext],
    )


if __name__ == "__main__":
    main()
