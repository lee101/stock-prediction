#!/usr/bin/env python3
"""Profile the crypto RL training pipeline.

Runs N training steps with PyTorch profiler + cProfile, outputs per-component
timing breakdown (data loading, forward, sim, loss, backward), flame graph
data, GPU utilization, and peak memory.

Uses synthetic data by default so it runs without training CSVs.
Pass --real-data to use actual BinanceHourlyDataModule if CSVs are available.

Usage:
    python binanceneural/profile_training.py --steps 5
    python binanceneural/profile_training.py --steps 10 --arch nano
    python binanceneural/profile_training.py --steps 5 --real-data --symbol DOGEUSD
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule, build_default_feature_columns
from binanceneural.model import PolicyConfig, build_policy

try:
    from trainingefficiency.fast_differentiable_sim import simulate_hourly_trades_fast
except ImportError:
    simulate_hourly_trades_fast = None

from differentiable_loss_utils import (
    HOURLY_PERIODS_PER_YEAR,
    compute_loss_by_type,
    simulate_hourly_trades,
)


TIMING_KEYS = ["data_loading", "data_transfer", "forward", "simulation", "loss_compute", "backward", "optimizer_step", "total"]
DEFAULT_HORIZONS = (1, 4, 12, 24)


def _empty_timings():
    return {k: [] for k in TIMING_KEYS}


def _synthetic_loader(batch_size, seq_len, input_dim, num_batches=64):
    """Generate a DataLoader of synthetic batches matching real data shapes."""
    n = batch_size * num_batches
    base_price = 100.0
    features = torch.randn(n, seq_len, input_dim)
    closes = base_price + torch.randn(n, seq_len) * 2
    opens = closes + torch.randn(n, seq_len) * 0.5
    highs = closes + torch.abs(torch.randn(n, seq_len)) * 1.5
    lows = closes - torch.abs(torch.randn(n, seq_len)) * 1.5
    reference_close = closes[:, -1:].expand_as(closes).clone()
    chronos_high = highs + torch.abs(torch.randn(n, seq_len)) * 0.5
    chronos_low = lows - torch.abs(torch.randn(n, seq_len)) * 0.5

    ds = TensorDataset(features, opens, highs, lows, closes, reference_close, chronos_high, chronos_low)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def _batch_from_tuple(tensors):
    """Convert TensorDataset tuple to dict matching real data format."""
    features, opens, highs, lows, closes, ref_close, c_high, c_low = tensors
    return {
        "features": features,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "reference_close": ref_close,
        "chronos_high": c_high,
        "chronos_low": c_low,
    }


def build_components(args):
    input_dim = len(build_default_feature_columns(DEFAULT_HORIZONS))
    use_real = getattr(args, "real_data", False)

    if use_real:
        ds_cfg = DatasetConfig(
            symbol=args.symbol,
            data_root=Path("trainingdatahourly") / "crypto",
            forecast_cache_root=Path("binanceneural") / "forecast_cache",
            forecast_horizons=DEFAULT_HORIZONS,
            sequence_length=args.seq_len,
            validation_days=70,
            cache_only=True,
        )
        data_module = BinanceHourlyDataModule(ds_cfg)
        input_dim = len(data_module.feature_columns)
        loader = data_module.train_dataloader(args.batch_size, num_workers=0)
        loader_is_dict = True
    else:
        loader = _synthetic_loader(args.batch_size, args.seq_len, input_dim)
        loader_is_dict = False

    policy_cfg = PolicyConfig(
        input_dim=input_dim,
        hidden_dim=args.dim,
        num_heads=args.heads,
        num_layers=args.layers,
        max_len=max(args.seq_len, 32),
        dropout=0.1,
        model_arch=args.arch,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_policy(policy_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.03)
    return model, optimizer, loader, device, loader_is_dict


def run_one_step(model, optimizer, batch, device, sim_fn, step_timings):
    t0 = time.perf_counter()

    features = batch["features"].to(device)
    opens = batch["open"].to(device) if "open" in batch else None
    highs = batch["high"].to(device)
    lows = batch["low"].to(device)
    closes = batch["close"].to(device)
    reference_close = batch["reference_close"].to(device)
    chronos_high = batch["chronos_high"].to(device)
    chronos_low = batch["chronos_low"].to(device)

    t_data = time.perf_counter()
    step_timings["data_transfer"].append(t_data - t0)

    outputs = model(features)
    actions = model.decode_actions(
        outputs,
        reference_close=reference_close,
        chronos_high=chronos_high,
        chronos_low=chronos_low,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fwd = time.perf_counter()
    step_timings["forward"].append(t_fwd - t_data)

    scale = 100.0
    sim_result = sim_fn(
        highs=highs,
        lows=lows,
        closes=closes,
        opens=opens,
        buy_prices=actions["buy_price"],
        sell_prices=actions["sell_price"],
        trade_intensity=actions["trade_amount"] / scale,
        buy_trade_intensity=actions["buy_amount"] / scale,
        sell_trade_intensity=actions["sell_amount"] / scale,
        maker_fee=0.001,
        initial_cash=1.0,
        can_short=False,
        can_long=True,
        max_leverage=1.0,
        market_order_entry=False,
        fill_buffer_pct=0.0005,
        margin_annual_rate=0.0,
        temperature=5e-4,
        decision_lag_bars=0,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_sim = time.perf_counter()
    step_timings["simulation"].append(t_sim - t_fwd)

    returns = sim_result.returns.float()
    loss, _score, _sortino, _annual_return = compute_loss_by_type(
        returns, "sortino", periods_per_year=HOURLY_PERIODS_PER_YEAR
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_loss = time.perf_counter()
    step_timings["loss_compute"].append(t_loss - t_sim)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_bwd = time.perf_counter()
    step_timings["backward"].append(t_bwd - t_loss)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_opt = time.perf_counter()
    step_timings["optimizer_step"].append(t_opt - t_bwd)
    step_timings["total"].append(t_opt - t0)

    return loss.item()


def _next_batch(batch_iter, loader, loader_is_dict):
    try:
        raw = next(batch_iter)
    except StopIteration:
        batch_iter = iter(loader)
        raw = next(batch_iter)
    if loader_is_dict:
        return raw, batch_iter
    return _batch_from_tuple(raw), batch_iter


def print_timing_breakdown(step_timings, num_steps):
    total = sum(step_timings["total"])
    print(f"\n{'='*60}")
    print(f"TIMING BREAKDOWN ({num_steps} steps, {total:.3f}s total)")
    print(f"{'='*60}")
    print(f"{'Component':<20} {'Total (s)':>10} {'Per-step (ms)':>14} {'Pct':>8}")
    print(f"{'-'*52}")
    for key in TIMING_KEYS[:-1]:
        vals = step_timings[key]
        t = sum(vals)
        per_step = t / len(vals) * 1000
        pct = t / total * 100
        print(f"{key:<20} {t:>10.4f} {per_step:>14.2f} {pct:>7.1f}%")
    print(f"{'-'*52}")
    print(f"{'steps/sec':<20} {num_steps / total:>10.2f}")


def print_gpu_info(device):
    if device.type != "cuda":
        print("\nGPU: not available (CPU mode)")
        return
    print(f"\n{'='*60}")
    print("GPU INFO")
    print(f"{'='*60}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    current_mb = torch.cuda.memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    print(f"Peak memory allocated: {peak_mb:.1f} MB")
    print(f"Current memory allocated: {current_mb:.1f} MB")
    print(f"Memory reserved: {reserved_mb:.1f} MB")
    try:
        utilization = torch.cuda.utilization(0)
        print(f"GPU utilization: {utilization}%")
    except Exception:
        pass


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_pytorch_profiler(model, optimizer, loader, device, sim_fn, num_steps, loader_is_dict):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    batch_iter = iter(loader)
    prof_schedule = schedule(wait=0, warmup=1, active=min(num_steps, 3), repeat=1)

    with profile(
        activities=activities,
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        step_timings = _empty_timings()
        for _ in range(min(num_steps + 1, len(loader))):
            _sync(device)
            t_data_start = time.perf_counter()
            batch, batch_iter = _next_batch(batch_iter, loader, loader_is_dict)
            _sync(device)
            t_data_end = time.perf_counter()
            step_timings["data_loading"].append(t_data_end - t_data_start)
            run_one_step(model, optimizer, batch, device, sim_fn, step_timings)
            prof.step()

    trace_path = Path("binanceneural") / "profile_trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\nChrome trace saved to: {trace_path}")

    print(f"\n{'='*60}")
    print("PYTORCH PROFILER - TOP CPU OPS")
    print(f"{'='*60}")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))

    if device.type == "cuda":
        print(f"\n{'='*60}")
        print("PYTORCH PROFILER - TOP CUDA OPS")
        print(f"{'='*60}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    print(f"\n{'='*60}")
    print("PYTORCH PROFILER - TOP MEMORY")
    print(f"{'='*60}")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=15))

    return prof


def run_cprofile(model, optimizer, loader, device, sim_fn, num_steps, loader_is_dict):
    batch_iter = iter(loader)
    step_timings = _empty_timings()

    pr = cProfile.Profile()
    pr.enable()

    for _ in range(min(num_steps, len(loader))):
        _sync(device)
        t_data_start = time.perf_counter()
        batch, batch_iter = _next_batch(batch_iter, loader, loader_is_dict)
        _sync(device)
        t_data_end = time.perf_counter()
        step_timings["data_loading"].append(t_data_end - t_data_start)
        run_one_step(model, optimizer, batch, device, sim_fn, step_timings)

    pr.disable()

    prof_path = Path("binanceneural") / "profile_cprofile.prof"
    pr.dump_stats(str(prof_path))
    print(f"\ncProfile data saved to: {prof_path}")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative")
    ps.print_stats(30)
    print(f"\n{'='*60}")
    print("CPROFILE - TOP CUMULATIVE")
    print(f"{'='*60}")
    print(s.getvalue())

    return step_timings


def main():
    parser = argparse.ArgumentParser(description="Profile crypto RL training pipeline")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--symbol", default="DOGEUSD")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--arch", default="classic")
    parser.add_argument("--use-fast-sim", action="store_true")
    parser.add_argument("--skip-pytorch-profiler", action="store_true")
    parser.add_argument("--real-data", action="store_true",
                        help="Use real BinanceHourlyDataModule instead of synthetic data")
    args = parser.parse_args()

    print(f"Profiling {args.steps} training steps")
    print(f"Symbol: {args.symbol}, Arch: {args.arch}, Dim: {args.dim}, Layers: {args.layers}")
    print(f"Batch: {args.batch_size}, Seq: {args.seq_len}")
    print(f"Data: {'real' if args.real_data else 'synthetic'}")

    if args.use_fast_sim and simulate_hourly_trades_fast is not None:
        sim_fn = simulate_hourly_trades_fast
        print("Sim: fast_differentiable_sim")
    else:
        sim_fn = simulate_hourly_trades
        print("Sim: differentiable_loss_utils (baseline)")

    print("\nBuilding model...")
    model, optimizer, loader, device, loader_is_dict = build_components(args)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {param_count:,} ({trainable_count:,} trainable)")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("\nWarmup step...")
    batch_iter = iter(loader)
    warmup_batch, batch_iter = _next_batch(batch_iter, loader, loader_is_dict)
    run_one_step(model, optimizer, warmup_batch, device, sim_fn, _empty_timings())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"\nRunning {args.steps} steps with cProfile...")
    step_timings = run_cprofile(model, optimizer, loader, device, sim_fn, args.steps, loader_is_dict)
    print_timing_breakdown(step_timings, len(step_timings["total"]))

    if not args.skip_pytorch_profiler:
        print(f"\nRunning {args.steps} steps with PyTorch profiler...")
        run_pytorch_profiler(model, optimizer, loader, device, sim_fn, args.steps, loader_is_dict)

    print_gpu_info(device)

    total_time = sum(step_timings["total"])
    actual_steps = len(step_timings["total"])
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Steps completed: {actual_steps}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Steps/sec: {actual_steps / total_time:.2f}")
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
