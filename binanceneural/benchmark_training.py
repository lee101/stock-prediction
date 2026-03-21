#!/usr/bin/env python3
"""Training benchmark harness for crypto RL pipeline throughput measurement.

Measures per-component timing breakdown for before/after optimization comparison.
Outputs JSON to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.optim import AdamW

from binanceneural.config import PolicyConfig, TrainingConfig
from binanceneural.model import build_policy
from differentiable_loss_utils import (
    HOURLY_PERIODS_PER_YEAR,
    compute_loss_by_type,
    simulate_hourly_trades,
)

N_FEATURES = 10


def create_synthetic_batch(batch_size: int, seq_len: int, device: torch.device):
    base_price = 100.0 + torch.randn(batch_size, 1, device=device) * 10
    noise = torch.randn(batch_size, seq_len, device=device) * 0.5
    closes = base_price + torch.cumsum(noise, dim=1)
    closes = torch.clamp(closes, min=1.0)
    highs = closes + torch.abs(torch.randn(batch_size, seq_len, device=device)) * 0.5
    lows = closes - torch.abs(torch.randn(batch_size, seq_len, device=device)) * 0.5
    lows = torch.clamp(lows, min=0.5)
    opens = closes + torch.randn(batch_size, seq_len, device=device) * 0.2
    opens = torch.clamp(opens, min=0.5)
    features = torch.randn(batch_size, seq_len, N_FEATURES, device=device) * 0.1
    reference_close = closes.clone()
    chronos_high = highs + torch.abs(torch.randn(batch_size, seq_len, device=device)) * 0.3
    chronos_low = lows - torch.abs(torch.randn(batch_size, seq_len, device=device)) * 0.3
    chronos_low = torch.clamp(chronos_low, min=0.1)
    return {
        "features": features,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "opens": opens,
        "reference_close": reference_close,
        "chronos_high": chronos_high,
        "chronos_low": chronos_low,
    }


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_cuda = device.type == "cuda"

    config = TrainingConfig(
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        learning_rate=3e-4,
        weight_decay=1e-4,
        transformer_dim=args.hidden_dim,
        transformer_layers=args.num_layers,
        transformer_heads=args.num_heads,
        maker_fee=0.001,
        max_leverage=1.0,
        fill_temperature=5e-4,
        return_weight=0.08,
        use_compile=args.use_compile,
    )

    policy_cfg = PolicyConfig(
        input_dim=N_FEATURES,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_len=args.seq_len,
        dropout=0.1,
    )

    model = build_policy(policy_cfg).to(device)
    if args.use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scale = float(config.trade_amount_scale)

    sim_fn = simulate_hourly_trades
    if args.use_fast_sim:
        try:
            from trainingefficiency.fast_differentiable_sim import simulate_hourly_trades_fast
            sim_fn = simulate_hourly_trades_fast
        except ImportError:
            pass

    torch.manual_seed(1337)
    batches = [create_synthetic_batch(args.batch_size, args.seq_len, device)
               for _ in range(args.warmup + args.steps)]

    timings = {"data": [], "forward": [], "sim": [], "loss": [], "backward": [], "optim": []}
    total_steps = args.warmup + args.steps

    if has_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)

    wall_start = time.perf_counter()

    for step in range(total_steps):
        optimizer.zero_grad(set_to_none=True)

        _sync(device)
        t0 = time.perf_counter()
        batch = batches[step]
        _sync(device)
        t1 = time.perf_counter()

        outputs = model(batch["features"])
        actions = model.decode_actions(
            outputs,
            reference_close=batch["reference_close"],
            chronos_high=batch["chronos_high"],
            chronos_low=batch["chronos_low"],
        )
        _sync(device)
        t2 = time.perf_counter()

        sim_result = sim_fn(
            highs=batch["highs"],
            lows=batch["lows"],
            closes=batch["closes"],
            opens=batch["opens"],
            buy_prices=actions["buy_price"],
            sell_prices=actions["sell_price"],
            trade_intensity=actions["trade_amount"] / scale,
            buy_trade_intensity=actions["buy_amount"] / scale,
            sell_trade_intensity=actions["sell_amount"] / scale,
            maker_fee=config.maker_fee,
            initial_cash=config.initial_cash,
            max_leverage=config.max_leverage,
            temperature=float(config.fill_temperature),
        )
        _sync(device)
        t3 = time.perf_counter()

        loss, score, sortino, ret = compute_loss_by_type(
            sim_result.returns,
            config.loss_type,
            target_sign=config.sortino_target_sign,
            periods_per_year=HOURLY_PERIODS_PER_YEAR,
            return_weight=config.return_weight,
        )
        _sync(device)
        t4 = time.perf_counter()

        loss.backward()
        _sync(device)
        t5 = time.perf_counter()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        _sync(device)
        t6 = time.perf_counter()

        if step >= args.warmup:
            timings["data"].append(t1 - t0)
            timings["forward"].append(t2 - t1)
            timings["sim"].append(t3 - t2)
            timings["loss"].append(t4 - t3)
            timings["backward"].append(t5 - t4)
            timings["optim"].append(t6 - t5)

    _sync(device)
    wall_end = time.perf_counter()

    total_time = wall_end - wall_start
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if has_cuda else 0.0

    avg = {k: sum(v) / len(v) if v else 0.0 for k, v in timings.items()}
    step_time = sum(avg.values())
    steps_per_sec = 1.0 / step_time if step_time > 0 else 0.0
    samples_per_sec = steps_per_sec * args.batch_size

    results = {
        "config": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "steps": args.steps,
            "warmup": args.warmup,
            "use_fast_sim": args.use_fast_sim,
            "use_compile": args.use_compile,
            "device": str(device),
        },
        "throughput": {
            "steps_per_sec": round(steps_per_sec, 2),
            "samples_per_sec": round(samples_per_sec, 2),
        },
        "gpu_peak_memory_mb": round(peak_mem_mb, 1),
        "total_time_sec": round(total_time, 3),
        "avg_step_time_sec": round(step_time, 6),
        "time_breakdown_sec": {k: round(v, 6) for k, v in avg.items()},
        "time_breakdown_pct": {
            k: round(100.0 * v / step_time, 1) if step_time > 0 else 0.0
            for k, v in avg.items()
        },
    }

    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark crypto RL training pipeline throughput")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--use-fast-sim", action="store_true")
    parser.add_argument("--use-compile", action="store_true")
    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
