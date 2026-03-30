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
from types import SimpleNamespace
from typing import Sequence

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule, build_default_feature_columns
from binanceneural.model import PolicyConfig, build_policy
from src.symbol_utils import is_crypto_symbol

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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = PROJECT_ROOT / "profiles"
DEFAULT_SYMBOLS = ("BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AAVEUSD", "LINKUSD", "UNIUSD")


def _normalize_symbol_list(symbols: Sequence[str] | str | None) -> list[str]:
    if symbols is None:
        return []
    if isinstance(symbols, str):
        raw_items = symbols.split(",")
    else:
        raw_items = list(symbols)
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in raw_items:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def _candidate_symbol_aliases(symbol: str) -> tuple[str, ...]:
    normalized = str(symbol).strip().upper()
    if not normalized:
        return tuple()

    aliases: list[str] = [normalized]
    if normalized.endswith("FDUSD"):
        aliases.append(normalized[:-5] + "USD")
        aliases.append(normalized[:-5] + "USDT")
    elif normalized.endswith("USDT"):
        aliases.append(normalized[:-4] + "USD")
        aliases.append(normalized[:-4] + "FDUSD")
    elif normalized.endswith("USD"):
        aliases.append(normalized[:-3] + "USDT")
        aliases.append(normalized[:-3] + "FDUSD")

    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        if alias and alias not in seen:
            seen.add(alias)
            deduped.append(alias)
    return tuple(deduped)


def _candidate_price_roots(symbol: str, project_root: Path = PROJECT_ROOT) -> tuple[Path, ...]:
    crypto_root = project_root / "trainingdatahourly" / "crypto"
    stock_root = project_root / "trainingdatahourly" / "stocks"
    if symbol.endswith(("USD", "USDT", "FDUSD")) or is_crypto_symbol(symbol):
        return (crypto_root, stock_root)
    return (stock_root, crypto_root)


def _candidate_forecast_roots(project_root: Path = PROJECT_ROOT) -> tuple[Path, ...]:
    return (
        project_root / "binanceneural" / "forecast_cache",
        project_root / "binanceneural" / "forecast_cache_shortable_stocks",
    )


def _resolve_symbol_dataset(
    symbol: str,
    *,
    project_root: Path = PROJECT_ROOT,
    preferred_horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> tuple[str, Path, Path, tuple[int, ...]]:
    requested = str(symbol).strip().upper()
    if not requested:
        raise FileNotFoundError("Symbol is empty")

    for alias in _candidate_symbol_aliases(requested):
        for data_root in _candidate_price_roots(alias, project_root):
            price_path = data_root / f"{alias}.csv"
            if not price_path.exists():
                continue
            for forecast_root in _candidate_forecast_roots(project_root):
                available_horizons = tuple(
                    int(h)
                    for h in preferred_horizons
                    if (forecast_root / f"h{int(h)}" / f"{alias}.parquet").exists()
                )
                if available_horizons:
                    return alias, data_root, forecast_root, available_horizons
    raise FileNotFoundError(
        f"No compatible price CSV + forecast cache found for {requested} "
        f"under {project_root / 'trainingdatahourly'} and {project_root / 'binanceneural'}."
    )


def _check_symbol_data(symbol: str, project_root: Path = PROJECT_ROOT) -> bool:
    try:
        _resolve_symbol_dataset(symbol, project_root=project_root, preferred_horizons=(1,))
    except FileNotFoundError:
        return False
    return True


def _select_symbols(symbols: Sequence[str] | str, project_root: Path = PROJECT_ROOT) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for symbol in _normalize_symbol_list(symbols):
        try:
            resolved, _data_root, _forecast_root, _horizons = _resolve_symbol_dataset(
                symbol,
                project_root=project_root,
                preferred_horizons=(1,),
            )
        except FileNotFoundError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        selected.append(resolved)
    if not selected:
        raise FileNotFoundError(
            f"No symbols with both hourly price history and h1 forecast cache found in {project_root}."
        )
    return selected


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
        resolved_symbol, data_root, forecast_cache_root, horizons = _resolve_symbol_dataset(args.symbol)
        ds_cfg = DatasetConfig(
            symbol=resolved_symbol,
            data_root=data_root,
            forecast_cache_root=forecast_cache_root,
            forecast_horizons=horizons,
            sequence_length=args.seq_len,
            validation_days=70,
            cache_only=True,
        )
        data_module = BinanceHourlyDataModule(ds_cfg)
        args.symbol = resolved_symbol
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


def run_pytorch_profiler(model, optimizer, loader, device, sim_fn, num_steps, loader_is_dict, output_dir: Path = PROFILES_DIR):
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "neural_cuda_trace.json"
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


def run_cprofile(model, optimizer, loader, device, sim_fn, num_steps, loader_is_dict, output_dir: Path = PROFILES_DIR):
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prof_path = output_dir / "neural_cprofile.prof"
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


def _write_neural_report(
    report_path: Path,
    *,
    symbol: str,
    step_timings: dict[str, list[float]],
    device: torch.device,
    trace_path: Path,
    profile_path: Path,
) -> Path:
    total_time = sum(step_timings.get("total", []))
    steps = len(step_timings.get("total", []))

    lines = [
        "# Neural Training Profiling Report",
        "",
        f"Symbol: {symbol}",
        f"Device: {device.type}",
        f"Steps: {steps}",
        f"Total time: {total_time:.3f}s",
        "",
        "| Component | Total (s) | Per-step (ms) | Pct |",
        "|-----------|-----------|---------------|-----|",
    ]
    for key in TIMING_KEYS[:-1]:
        vals = list(step_timings.get(key, []))
        total = sum(vals)
        per_step_ms = (total / len(vals) * 1000.0) if vals else 0.0
        pct = (total / total_time * 100.0) if total_time > 0.0 else 0.0
        lines.append(f"| {key} | {total:.4f} | {per_step_ms:.2f} | {pct:.1f}% |")
    lines.extend(
        [
            "",
            f"Chrome trace: `{trace_path.name}`{' (missing)' if not trace_path.exists() else ''}",
            f"cProfile: `{profile_path.name}`{' (missing)' if not profile_path.exists() else ''}",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))
    return report_path


def run_neural_profiling(
    symbols: Sequence[str] | str,
    steps: int,
    profiles_dir: Path,
    *,
    batch_size: int = 16,
    seq_len: int = 48,
    dim: int = 512,
    heads: int = 8,
    layers: int = 6,
    arch: str = "classic",
    use_fast_sim: bool = False,
    real_data: bool = True,
    quick: bool = False,
) -> dict[str, object]:
    profiles_dir = Path(profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    selected_symbols = _select_symbols(symbols) if real_data else (_normalize_symbol_list(symbols) or [DEFAULT_SYMBOLS[0]])
    symbol = selected_symbols[0]
    num_steps = max(1, int(steps))

    args = SimpleNamespace(
        steps=num_steps,
        symbol=symbol,
        batch_size=batch_size,
        seq_len=seq_len,
        dim=dim,
        heads=heads,
        layers=layers,
        arch=arch,
        use_fast_sim=use_fast_sim,
        real_data=real_data,
    )

    if use_fast_sim and simulate_hourly_trades_fast is not None:
        sim_fn = simulate_hourly_trades_fast
        print("Sim: fast_differentiable_sim")
    else:
        sim_fn = simulate_hourly_trades
        print("Sim: differentiable_loss_utils (baseline)")

    print(f"Profiling {num_steps} training steps")
    print(f"Symbol: {symbol}, Arch: {arch}, Dim: {dim}, Layers: {layers}")
    print(f"Batch: {batch_size}, Seq: {seq_len}")
    print(f"Data: {'real' if real_data else 'synthetic'}")

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

    print(f"\nRunning {num_steps} steps with cProfile...")
    step_timings = run_cprofile(
        model,
        optimizer,
        loader,
        device,
        sim_fn,
        num_steps,
        loader_is_dict,
        output_dir=profiles_dir,
    )
    print_timing_breakdown(step_timings, len(step_timings["total"]))

    trace_path = profiles_dir / "neural_cuda_trace.json"
    profile_path = profiles_dir / "neural_cprofile.prof"
    if not quick:
        print(f"\nRunning {num_steps} steps with PyTorch profiler...")
        run_pytorch_profiler(
            model,
            optimizer,
            loader,
            device,
            sim_fn,
            num_steps,
            loader_is_dict,
            output_dir=profiles_dir,
        )

    print_gpu_info(device)

    report_path = _write_neural_report(
        profiles_dir / "neural_report.md",
        symbol=symbol,
        step_timings=step_timings,
        device=device,
        trace_path=trace_path,
        profile_path=profile_path,
    )
    print(f"Report written to: {report_path}")

    return {
        "symbol": symbol,
        "steps": num_steps,
        "profiles_dir": profiles_dir,
        "report_path": report_path,
        "trace_path": trace_path,
        "profile_path": profile_path,
    }


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Profile crypto RL training pipeline")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Legacy alias for profiling iterations.")
    parser.add_argument("--symbol", default="DOGEUSD")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol candidates; first compatible symbol is used.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--arch", default="classic")
    parser.add_argument("--use-fast-sim", action="store_true")
    parser.add_argument("--skip-pytorch-profiler", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Skip PyTorch profiler trace generation.")
    parser.add_argument("--output-dir", type=Path, default=PROFILES_DIR)
    parser.add_argument("--real-data", action="store_true",
                        help="Use real BinanceHourlyDataModule instead of synthetic data")
    args = parser.parse_args(argv)

    symbol_list = _normalize_symbol_list(args.symbols) or [str(args.symbol).strip().upper()]
    steps = args.steps if args.steps is not None else (args.epochs if args.epochs is not None else 5)
    run_neural_profiling(
        symbol_list,
        steps,
        args.output_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        arch=args.arch,
        use_fast_sim=args.use_fast_sim,
        real_data=(args.real_data or bool(args.symbols)),
        quick=(args.quick or args.skip_pytorch_profiler),
    )


if __name__ == "__main__":
    main()
