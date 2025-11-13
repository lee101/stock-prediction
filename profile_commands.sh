#!/bin/bash
# Different profiling approaches for trade_stock_e2e.py

echo "=== Profiling Commands ==="
echo ""

echo "1. PyTorch Profiler (CPU/GPU ops, memory):"
echo "   python profile_pytorch.py"
echo "   # Creates pytorch_trace.json -> view in chrome://tracing"
echo ""

echo "2. py-spy with native stacks (C extensions visible):"
echo "   sudo py-spy record --native --rate 100 --format speedscope -o profile_native.json -- python -c 'import os; os.environ[\"PAPER\"]=\"1\"; exec(open(\"trade_stock_e2e.py\").read())'"
echo "   # View at speedscope.app"
echo ""

echo "3. py-spy with subprocesses:"
echo "   sudo py-spy record --subprocesses --rate 100 --format speedscope -o profile_all.json -- python -c 'import os; os.environ[\"PAPER\"]=\"1\"; exec(open(\"trade_stock_e2e.py\").read())'"
echo ""

echo "4. cProfile with snakeviz (interactive):"
echo "   python -m cProfile -o profile.pstats -c 'import os; os.environ[\"PAPER\"]=\"1\"; exec(open(\"trade_stock_e2e.py\").read())'"
echo "   snakeviz profile.pstats"
echo ""

echo "5. Memory profiler (line-by-line):"
echo "   python -m memory_profiler trade_stock_e2e.py"
echo "   # Needs @profile decorators in code"
echo ""

echo "6. tracemalloc (builtin memory):"
echo "   python profile_tracemalloc.py"
echo ""

echo "7. line_profiler (line-by-line time):"
echo "   kernprof -l -v trade_stock_e2e.py"
echo "   # Needs @profile decorators"
