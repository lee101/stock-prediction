from setuptools import setup, Extension
import numpy as np
import os

rl_dir = os.path.dirname(os.path.abspath(__file__))

ext = Extension(
    "binding",
    sources=[os.path.join(rl_dir, "binding.c")],
    include_dirs=[rl_dir, np.get_include()],
    extra_compile_args=["-O2", "-std=c99", "-Wno-unused-parameter"],
)

setup(
    name="rl_trading_binding",
    ext_modules=[ext],
)
