# Chronos2 Import Issue Investigation

## Problem
Getting error: `chronos>=2.0 is unavailable; install chronos-forecasting>=2.0 to enable Chronos2Pipeline.`

## Investigation

### Package Status
- `chronos-forecasting==2.0.1` is installed correctly
- Package files exist at `.venv/lib/python3.12/site-packages/chronos/`
- `chronos2` submodule directory exists with all necessary files

### Root Cause
The import fails with `std::bad_alloc` (C++ memory allocation error) when trying to import from `chronos.chronos2`.

This happens during module initialization, likely when:
1. Loading C++ extensions
2. Initializing CUDA/torch components
3. Pre-loading model weights or configs

### Error Trace
```
from chronos.chronos2 import Chronos2Pipeline
→ terminate called after throwing an instance of 'std::bad_alloc'
→ what():  std::bad_alloc
```

## Next Steps

### Option 1: Use ChronosBoltPipeline
The package includes `ChronosBoltPipeline` which imports successfully and is designed to be faster/more memory efficient:
```python
from chronos import ChronosBoltPipeline
```

### Option 2: Increase available memory
- Close other applications
- Reduce PyTorch memory usage
- Set environment variables:
  - `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
  - `OMP_NUM_THREADS=1`

### Option 3: Debug memory issue
Check system resources and torch memory settings during import attempt.

## Files Modified
- Created `tests/test_chronos2_e2e_compile.py` - E2E test for Chronos2 (currently fails due to import issue)

## Current State
Package is installed but cannot be imported due to memory allocation error during module initialization.
