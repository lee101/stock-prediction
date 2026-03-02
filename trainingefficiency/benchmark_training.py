#!/usr/bin/env python3
"""Benchmark training pipeline configurations.

Compares: FP32 baseline, BF16 split AMP, vectorized sim, compiled sim.
Reports wall time, GPU memory, and Sortino quality per config.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from loguru import logger

from binanceneural.data import MultiSymbolDataModule
from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.model import build_policy, PolicyConfig
from differentiable_loss_utils import (
    HOURLY_PERIODS_PER_YEAR,
    simulate_hourly_trades,
    compute_hourly_objective,
    combined_sortino_pnl_loss,
)
from trainingefficiency.fast_differentiable_sim import (
    simulate_hourly_trades_fast,
    simulate_hourly_trades_compiled,
)


def time_fn(fn, warmup=3, repeats=10):
    """GPU-accurate timing with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def run_forward_sim_backward(
    model, batch, device, config,
    use_amp=False, split_amp=False, sim_fn=simulate_hourly_trades,
):
    """Single training step: forward + sim + loss + backward."""
    features = batch["features"].to(device)
    highs = batch["high"].to(device)
    lows = batch["low"].to(device)
    closes = batch["close"].to(device)
    opens = batch["open"].to(device) if "open" in batch else None
    ref_close = batch["reference_close"].to(device)
    chr_high = batch["chronos_high"].to(device)
    chr_low = batch["chronos_low"].to(device)

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    if split_amp:
        with amp_ctx:
            outputs = model(features)
            actions = model.decode_actions(outputs, reference_close=ref_close,
                                           chronos_high=chr_high, chronos_low=chr_low)
        scale = float(config.trade_amount_scale)
        sim_kwargs = {
            "highs": highs.float(), "lows": lows.float(), "closes": closes.float(),
            "opens": opens.float() if opens is not None else None,
            "buy_prices": actions["buy_price"].float(),
            "sell_prices": actions["sell_price"].float(),
            "trade_intensity": actions["trade_amount"].float() / scale,
            "buy_trade_intensity": actions["buy_amount"].float() / scale,
            "sell_trade_intensity": actions["sell_amount"].float() / scale,
            "maker_fee": config.maker_fee, "initial_cash": config.initial_cash,
            "max_leverage": config.max_leverage,
        }
    else:
        with amp_ctx:
            outputs = model(features)
            actions = model.decode_actions(outputs, reference_close=ref_close,
                                           chronos_high=chr_high, chronos_low=chr_low)
            scale = float(config.trade_amount_scale)
            sim_kwargs = {
                "highs": highs, "lows": lows, "closes": closes, "opens": opens,
                "buy_prices": actions["buy_price"],
                "sell_prices": actions["sell_price"],
                "trade_intensity": actions["trade_amount"] / scale,
                "buy_trade_intensity": actions["buy_amount"] / scale,
                "sell_trade_intensity": actions["sell_amount"] / scale,
                "maker_fee": config.maker_fee, "initial_cash": config.initial_cash,
                "max_leverage": config.max_leverage,
            }

    sim = sim_fn(**sim_kwargs, temperature=float(config.fill_temperature))
    returns = sim.returns.float()
    loss = combined_sortino_pnl_loss(returns, return_weight=config.return_weight)
    loss.backward()

    return float(loss.detach()), float(compute_hourly_objective(
        returns.detach(), return_weight=config.return_weight)[1].mean())


def benchmark_config(name, model, batches, device, config, **kwargs):
    """Benchmark a single configuration."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    losses = []
    sortinos = []
    t0 = time.perf_counter()

    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        loss_val, sortino_val = run_forward_sim_backward(
            model, batch, device, config, **kwargs)
        optimizer.step()
        losses.append(loss_val)
        sortinos.append(sortino_val)

    torch.cuda.synchronize()
    wall_time = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    avg_loss = sum(losses) / len(losses)
    avg_sortino = sum(sortinos) / len(sortinos)

    logger.info("{:30s} | {:.2f}s | {:.0f}MB | loss={:.4f} sort={:.2f}",
                name, wall_time, peak_mem, avg_loss, avg_sortino)
    return {"name": name, "time": wall_time, "memory_mb": peak_mem,
            "loss": avg_loss, "sortino": avg_sortino}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,MTCH,NYT")
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--stock-data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--stock-cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--forecast-horizons", type=str, default="1,24")
    args = parser.parse_args()

    device = torch.device("cuda")
    stocks = [s.strip() for s in args.symbols.split(",")]
    horizons = [int(h) for h in args.forecast_horizons.split(",")]

    stock_config = DatasetConfig(
        symbol=stocks[0], data_root=str(args.stock_data_root),
        forecast_cache_root=str(args.stock_cache_root),
        forecast_horizons=horizons, sequence_length=args.sequence_length,
        val_fraction=0.15, min_history_hours=100, validation_days=30, cache_only=True,
    )
    data = MultiSymbolDataModule(stocks, stock_config)
    loader = data.train_dataloader(args.batch_size)

    config = TrainingConfig(
        batch_size=args.batch_size, sequence_length=args.sequence_length,
        learning_rate=1e-5, weight_decay=0.04, return_weight=0.15,
        transformer_dim=args.hidden_dim, transformer_heads=args.num_heads,
        transformer_layers=args.num_layers, maker_fee=0.001,
        max_leverage=2.0, fill_temperature=5e-4, logits_softcap=12.0,
    )

    batches = []
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        batches.append(batch)
    logger.info("Loaded {} batches", len(batches))

    policy_cfg = PolicyConfig(
        input_dim=len(data.feature_columns),
        hidden_dim=args.hidden_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, max_len=args.sequence_length,
    )

    results = []

    # 1. FP32 baseline
    model = build_policy(policy_cfg).to(device)
    torch.manual_seed(1337)
    results.append(benchmark_config(
        "FP32 baseline", model, batches, device, config,
        use_amp=False, split_amp=False, sim_fn=simulate_hourly_trades))

    # 2. BF16 full (current broken approach)
    model = build_policy(policy_cfg).to(device)
    torch.manual_seed(1337)
    results.append(benchmark_config(
        "BF16 full (sim in bf16)", model, batches, device, config,
        use_amp=True, split_amp=False, sim_fn=simulate_hourly_trades))

    # 3. BF16 split (model bf16, sim fp32)
    model = build_policy(policy_cfg).to(device)
    torch.manual_seed(1337)
    results.append(benchmark_config(
        "BF16 split (sim in fp32)", model, batches, device, config,
        use_amp=True, split_amp=True, sim_fn=simulate_hourly_trades))

    # 4. BF16 split + vectorized sim
    model = build_policy(policy_cfg).to(device)
    torch.manual_seed(1337)
    results.append(benchmark_config(
        "BF16 split + vectorized sim", model, batches, device, config,
        use_amp=True, split_amp=True, sim_fn=simulate_hourly_trades_fast))

    # 5. BF16 split + compiled sim
    model = build_policy(policy_cfg).to(device)
    torch.manual_seed(1337)
    results.append(benchmark_config(
        "BF16 split + compiled sim", model, batches, device, config,
        use_amp=True, split_amp=True, sim_fn=simulate_hourly_trades_compiled))

    # Summary
    logger.info("\n=== SUMMARY ===")
    baseline_time = results[0]["time"]
    for r in results:
        speedup = baseline_time / r["time"] if r["time"] > 0 else 0
        logger.info("{:30s} | {:.2f}s ({:.2f}x) | {:.0f}MB | sort={:.2f}",
                    r["name"], r["time"], speedup, r["memory_mb"], r["sortino"])


if __name__ == "__main__":
    main()
