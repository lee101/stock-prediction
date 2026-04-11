"""Build script for gpu_trading_env.

The Python package is installed as pure Python; the CUDA extension is compiled
lazily at first use via ``torch.utils.cpp_extension.load`` from
``gpu_trading_env/__init__.py``. This keeps ``pip install -e .`` trivial on
boxes where torch's CUDA toolchain version check would otherwise reject the
system nvcc (the JIT ``load`` path tolerates minor mismatches that the
``CUDAExtension`` setuptools path does not).
"""
from setuptools import setup, find_packages

setup(
    name="gpu_trading_env",
    version="0.1.0",
    description="SoA fused env_step CUDA kernel for GPU-resident RL trading",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    python_requires=">=3.10",
)
