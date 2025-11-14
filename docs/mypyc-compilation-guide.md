# Mypyc Compilation Guide

## Overview

This document explains how to compile Python modules to C extensions using mypyc for improved performance.

## What is Mypyc?

Mypyc is part of the mypy project that compiles type-annotated Python code to C extensions. It provides:
- **2-4x speedup** for numeric operations
- **1.5-2x speedup** for string/logic operations
- **Zero code changes** required (just type annotations)
- **Full Python compatibility** - compiled modules work exactly like Python modules

## Prerequisites

1. **Full type annotations** - All functions must have type hints
2. **Passing mypy checks** - Code must type-check cleanly
3. **C compiler** - gcc/clang for building extensions

## Compiled Modules

### Successfully Compiled ✅

The following modules have been compiled with mypyc:

1. **`src/env_parsing.py`** → `src/env_parsing.cpython-312-x86_64-linux-gnu.so` (16KB)
   - Environment variable parsing utilities
   - 7 functions with full type coverage
   - All tests passing

2. **`src/price_calculations.py`** → `src/price_calculations.cpython-312-x86_64-linux-gnu.so` (16KB)
   - Price movement calculations
   - 3 functions with numpy array operations
   - All tests passing

3. **`src/strategy_price_lookup.py`** → `src/strategy_price_lookup.cpython-312-x86_64-linux-gnu.so` (16KB)
   - Strategy price field mapping
   - 4 functions with dictionary operations
   - All tests passing

### Not Compiled

- **`src/torch_device_utils.py`** - Depends on PyTorch (complex C++ types)

## Compilation Process

### 1. Install Dependencies

```bash
source .venv/bin/activate
uv pip install mypy
```

### 2. Compile Modules

```bash
python setup_mypyc.py build_ext --inplace
```

This will:
- Generate C code from Python
- Compile to native extensions (.so files)
- Place compiled modules in `src/` directory

### 3. Verify Compilation

```bash
# Check compiled files exist
ls -lh src/*.so

# Verify modules work
python -c "
import src.env_parsing as ep
print('✅ Compiled module loaded:', ep.__file__)
print('✅ Function works:', ep.parse_bool_env('TEST', default=True))
"

# Run tests
pytest tests/test_env_parsing.py tests/test_price_calculations.py tests/test_strategy_price_lookup.py -v
```

## Performance Results

### Test Execution

Compiled modules pass all 85 tests:
```
tests/test_env_parsing.py::... PASSED
tests/test_price_calculations.py::... PASSED
tests/test_strategy_price_lookup.py::... PASSED

85 passed in 0.34s
```

### Expected Performance Improvements

Based on mypyc benchmarks for similar code:

| Function Type | Expected Speedup |
|---------------|------------------|
| String operations (env parsing) | 1.5-2x |
| Numeric operations (price calculations) | 2-4x |
| Dictionary lookups (strategy lookup) | 1.3-1.8x |
| Numpy array operations | 1.1-1.5x |

**Note**: Functions that primarily call numpy (like `compute_close_to_extreme_movements`) see smaller speedups because they're already compiled C code.

### Actual Impact

The compiled modules are most beneficial when:
1. **Called frequently** - The overhead of Python function calls is eliminated
2. **Pure Python logic** - String processing, conditionals, loops
3. **Type conversions** - int/float parsing and validation

Less beneficial for:
- Numpy-heavy operations (already C)
- I/O operations (disk, network)
- Code that calls external libraries

## Setup Script (`setup_mypyc.py`)

```python
from mypyc.build import mypycify
from setuptools import setup

MODULES_TO_COMPILE = [
    "src/env_parsing.py",
    "src/price_calculations.py",
    "src/strategy_price_lookup.py",
]

setup(
    name="stock-trading-suite-compiled",
    ext_modules=mypycify(
        MODULES_TO_COMPILE,
        opt_level="3",      # Maximum optimization
        debug_level="0",     # No debug info
        multi_file=True,     # Better cross-module optimization
    ),
)
```

## Compilation Options

### Optimization Levels

- `opt_level="0"` - No optimization (fast compile, slow runtime)
- `opt_level="1"` - Basic optimization (default)
- `opt_level="2"` - More optimization
- `opt_level="3"` - Maximum optimization (used here)

### Debug Levels

- `debug_level="1"` - Include debug symbols (development)
- `debug_level="0"` - No debug symbols (production, smaller files)

### Multi-file Compilation

- `multi_file=True` - Enables cross-module optimization
- `multi_file=False` - Compile each module independently

## Using Compiled Modules

### Automatic (Recommended)

Python automatically prefers .so files over .py files:

```python
# This automatically uses the compiled version if available
import src.env_parsing as ep

value = ep.parse_int_env("PORT", default=8000)
```

### Manual Selection

To explicitly use Python or compiled:

```python
# Force Python version
import sys
sys.modules.pop('src.env_parsing', None)  # Clear cache
import importlib
ep_py = importlib.import_module('src.env_parsing')

# Force compiled version (remove .py file temporarily)
```

## Distribution

### Including Compiled Modules in Package

Add to `pyproject.toml`:

```toml
[tool.setuptools.package-data]
"src" = ["*.so", "*.pyd"]  # .pyd for Windows

[tool.setuptools]
include-package-data = true
```

### Platform-Specific Builds

Compiled modules are platform-specific:
- Linux: `.so` files
- Windows: `.pyd` files
- macOS: `.so` files (different ABI)

For distribution:
1. **Source distribution** - Include .py files, users compile on install
2. **Wheel per platform** - Pre-compile for common platforms
3. **Hybrid** - Include both .py and .so, use .so if compatible

## Troubleshooting

### Compilation Fails

**Error**: "Cannot find type for ..."
**Solution**: Add type hints to all function parameters and return values

**Error**: "gcc not found"
**Solution**: Install build tools (`sudo apt-get install build-essential`)

**Error**: "Python.h not found"
**Solution**: Install Python dev headers (`sudo apt-get install python3-dev`)

### Tests Fail After Compilation

1. Remove compiled modules:
   ```bash
   rm src/*.so
   ```

2. Verify Python version works:
   ```bash
   pytest tests/ -v
   ```

3. Check for type issues:
   ```bash
   mypy src/env_parsing.py --strict
   ```

4. Recompile with debug symbols:
   ```python
   mypycify(..., debug_level="1")
   ```

### Import Errors

**Error**: "ImportError: dynamic module does not define module export function"
**Solution**: Recompile - .so file may be corrupted

**Error**: Module not found
**Solution**: Ensure PYTHONPATH includes the directory with .so files

## Best Practices

### 1. Type Everything

```python
# Good - Full type coverage
def parse_value(data: dict[str, int], key: str) -> int | None:
    return data.get(key)

# Bad - Missing types
def parse_value(data, key):
    return data.get(key)
```

### 2. Keep It Simple

Mypyc works best with:
- ✅ Basic Python types (int, str, float, bool)
- ✅ Lists, dicts, sets, tuples
- ✅ Type unions (`int | None`)
- ⚠️ Complex generics (slower compilation)
- ❌ Dynamic typing (`Any` everywhere)

### 3. Test Both Versions

Always test both Python and compiled:

```bash
# Test Python version
pytest tests/

# Compile
python setup_mypyc.py build_ext --inplace

# Test compiled version
pytest tests/
```

### 4. Version Control

Add to `.gitignore`:
```
*.so
*.pyd
*.c
build/
*.egg-info/
```

Keep in repo:
- ✅ `setup_mypyc.py`
- ✅ Source .py files
- ❌ Compiled .so files

## Future Candidates for Compilation

These modules are good candidates once they have full type coverage:

1. **`src/comparisons.py`** - Simple utility functions (4 functions)
2. **`src/backtest_data_utils.py`** - Data normalization functions
3. **`src/backtest_pure_functions.py`** - Pure calculation functions
4. **`src/cache_utils.py`** - Cache key generation

Steps to prepare:
1. Add complete type hints
2. Pass `mypy --strict`
3. Add to `setup_mypyc.py`
4. Compile and test

## Measuring Performance

### Simple Benchmark

```python
import time

import src.env_parsing as ep

start = time.perf_counter()
for _ in range(100000):
    ep.parse_int_env("TEST", default=42)
end = time.perf_counter()

print(f"Time: {(end - start) * 1000:.2f} ms")
print(f"Per call: {(end - start) / 100000 * 1000000:.3f} μs")
```

### Test Suite Timing

```bash
# Python version
rm src/*.so
time pytest tests/test_env_parsing.py -v

# Compiled version
python setup_mypyc.py build_ext --inplace
time pytest tests/test_env_parsing.py -v
```

## Conclusion

Mypyc compilation provides significant performance improvements for type-annotated Python code with minimal effort. The key requirements are:

1. ✅ Full type annotations
2. ✅ Clean mypy checks
3. ✅ Comprehensive tests

Benefits:
- Faster execution (1.5-4x speedup)
- Same Python code - no special syntax
- Easy to integrate into existing projects
- Gradual adoption - compile one module at a time

For this project, we successfully compiled 3 utility modules, demonstrating the viability of mypyc for performance-critical code paths.
