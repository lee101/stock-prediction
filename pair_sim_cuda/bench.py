"""Benchmark fused pair step (CUDA) vs pure-PyTorch reference.

Run:
    python -m pair_sim_cuda.bench
"""

from __future__ import annotations

import argparse
import json
import time
from statistics import median
from typing import Callable

import torch

from pair_sim_cuda.wrapper import (
    PairStepConfig,
    build_extension,
    fused_pair_step,
    fused_pair_step_reference,
)


def _rand_inputs(B: int, P: int, device: str = "cuda"):
    g = torch.Generator(device=device).manual_seed(0)
    target_pos = torch.empty(B, P, device=device).uniform_(-1, 1, generator=g).requires_grad_()
    offset_bps = torch.empty(B, P, device=device).uniform_(0, 3, generator=g).requires_grad_()
    prev_pos = torch.empty(B, P, device=device).uniform_(-1, 1, generator=g).requires_grad_()
    pair_ret = torch.empty(B, P, device=device).uniform_(-0.02, 0.02, generator=g)
    reach_side = torch.empty(B, P, device=device).uniform_(0, 20, generator=g)
    half_spread = torch.empty(B, P, device=device).uniform_(0.5, 5, generator=g)
    session_mask = torch.ones(B, device=device)
    return (target_pos, offset_bps, prev_pos, pair_ret, reach_side,
            half_spread, session_mask)


def _bench(fn: Callable, inputs, iters: int = 50, warmup: int = 10):
    tp, ob, pp, *rest = inputs
    # Forward + backward
    torch.cuda.synchronize()
    for _ in range(warmup):
        np_, pn_, tv_ = fn(tp, ob, pp, *rest, PairStepConfig())
        loss = pn_.sum()
        loss.backward()
        tp.grad = None
        ob.grad = None
        pp.grad = None
    torch.cuda.synchronize()

    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        np_, pn_, tv_ = fn(tp, ob, pp, *rest, PairStepConfig())
        loss = pn_.sum()
        loss.backward()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
        tp.grad = None
        ob.grad = None
        pp.grad = None
    return ts


def _fmt(ts):
    return {
        "mean_ms": 1000.0 * sum(ts) / len(ts),
        "median_ms": 1000.0 * median(ts),
        "min_ms": 1000.0 * min(ts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=str, default="64,256,1024")
    ap.add_argument("--pairs", type=str, default="128,512,2000")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    build_extension(verbose=False)

    batches = [int(x) for x in args.batch_sizes.split(",")]
    pairs = [int(x) for x in args.pairs.split(",")]

    rows = []
    for B in batches:
        for P in pairs:
            inputs = _rand_inputs(B, P)
            ref_ts = _bench(fused_pair_step_reference, inputs, args.iters, args.warmup)
            fused_ts = _bench(fused_pair_step, inputs, args.iters, args.warmup)
            ref = _fmt(ref_ts)
            fused = _fmt(fused_ts)
            speedup = ref["median_ms"] / fused["median_ms"]
            rows.append({
                "B": B, "P": P,
                "ref_ms": ref["median_ms"],
                "fused_ms": fused["median_ms"],
                "speedup_x": speedup,
            })
            print(f"B={B:>5} P={P:>5}  ref={ref['median_ms']:8.3f} ms  "
                  f"fused={fused['median_ms']:8.3f} ms  speedup={speedup:5.2f}x")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "rows": rows,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "cap": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
            }, f, indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
