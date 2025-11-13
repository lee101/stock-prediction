#!/usr/bin/env python3
"""PyTorch profiler wrapper for trade_stock_e2e.py - shows CPU/GPU ops, memory"""
import os
import sys
import torch
from torch.profiler import profile, ProfilerActivity, schedule

# Set env before importing trade code
os.environ['PAPER'] = '1'

def main():
    # Import after env is set
    import trade_stock_e2e

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        # Run one iteration of main loop
        # You may need to modify this to call the right entry point
        trade_stock_e2e.main()

    # Export to Chrome trace (view in chrome://tracing)
    prof.export_chrome_trace("pytorch_trace.json")

    # Print table
    print("\n=== Top CPU Time ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    print("\n=== Top Memory ===")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=30))

    if torch.cuda.is_available():
        print("\n=== Top CUDA Time ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

if __name__ == "__main__":
    main()
