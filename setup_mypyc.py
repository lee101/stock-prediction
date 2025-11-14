"""Setup script for compiling Python modules with mypyc.

This script compiles type-annotated Python modules to C extensions
for improved performance.
"""

import sys
from pathlib import Path

from mypyc.build import mypycify
from setuptools import setup

# Modules to compile (must have full type annotations)
MODULES_TO_COMPILE = [
    "src/env_parsing.py",
    "src/price_calculations.py",
    "src/strategy_price_lookup.py",
    # "src/torch_device_utils.py",  # Skip torch module for now
]

# Check that modules exist
missing = [m for m in MODULES_TO_COMPILE if not Path(m).exists()]
if missing:
    print(f"Error: Missing modules: {missing}")
    sys.exit(1)

# Compile with mypyc
setup(
    name="stock-trading-suite-compiled",
    version="0.1.0",
    ext_modules=mypycify(
        MODULES_TO_COMPILE,
        opt_level="3",  # Maximum optimization
        debug_level="0",  # No debug info for production
        multi_file=True,  # Better optimization across modules
    ),
    zip_safe=False,
)
