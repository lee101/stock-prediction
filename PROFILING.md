# Profiling Guide

## Quick Start (Recommended Order)

### 1. PyTorch Profiler (Best for ML/PyTorch bottlenecks)
```bash
python profile_pytorch.py
# Creates pytorch_trace.json
# Open chrome://tracing and load the JSON
```
Shows: individual PyTorch ops (matmul, conv, etc), CPU/GPU time, memory per op

### 2. py-spy with Native (Best for C extensions/native code)
```bash
sudo py-spy record --native --rate 100 --format speedscope -o native.json \
  -- python -c 'import os; os.environ["PAPER"]="1"; exec(open("trade_stock_e2e.py").read())'
```
View at https://speedscope.app - drag and drop native.json

Shows: C/Cython/Rust stack frames, PyTorch C++ internals

### 3. Tracemalloc (Memory allocations)
```bash
python profile_tracemalloc.py
```
Shows: which lines allocate most memory, grouped by file

## Deep Dive Options

### Line Profiler (line-by-line timing)
Add `@profile` decorator to specific functions you want to profile:
```python
@profile
def main():
    ...
```

Run:
```bash
kernprof -l trade_stock_e2e.py
python -m line_profiler trade_stock_e2e.py.lprof
```

### Austin (Low overhead, native)
```bash
austin -o profile.austin python trade_stock_e2e.py
austin2speedscope profile.austin profile.json
```

### Scalene (CPU+GPU+Memory combined)
```bash
scalene trade_stock_e2e.py
```
Opens browser with detailed breakdown

## Tips
- For PyTorch: profile_pytorch.py gives best visibility
- For native code: py-spy --native
- For memory: tracemalloc or scalene
- Standard flamegraphs hide Python→C transitions
