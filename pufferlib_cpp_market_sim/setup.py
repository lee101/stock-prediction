"""Build script for the ``market_sim_py`` Python bindings.

This compiles ``python/bindings.cpp`` as a torch C++ extension and links it
against the prebuilt ``libmarket_sim.so`` (which contains the actual C++
simulator). The shared library is expected to live in ``build/`` after running
the existing CMake build:

    cd pufferlib_cpp_market_sim && mkdir -p build && cd build && cmake .. && make -j

We deliberately do *not* edit the project ``CMakeLists.txt`` because other
units may be modifying it concurrently.
"""

from __future__ import annotations

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIR = os.path.join(HERE, "include")
BUILD_DIR = os.path.join(HERE, "build")

ext_modules = [
    CppExtension(
        name="market_sim_py._market_sim_py_ext",
        sources=[os.path.join("python", "bindings.cpp")],
        include_dirs=[INCLUDE_DIR],
        library_dirs=[BUILD_DIR],
        libraries=["market_sim"],
        runtime_library_dirs=[BUILD_DIR],
        extra_compile_args=["-O3", "-std=c++17", "-fvisibility=hidden"],
    )
]

setup(
    name="market_sim_py",
    version="0.1.0",
    description="Python bindings for the pufferlib_cpp_market_sim C++ env",
    package_dir={"market_sim_py": "python/market_sim_py"},
    packages=["market_sim_py"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    zip_safe=False,
)
