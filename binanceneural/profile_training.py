#!/usr/bin/env python3
"""Profile binanceneural training for 2 epochs.

Outputs:
  profiles/neural_cuda_trace.json  — Chrome trace for chrome://tracing
  profiles/neural_report.md        — top-20 GPU kernels by time

Usage:
  cd /nvme0n1-disk/code/stock-prediction
  source .venv313/bin/activate
  python binanceneural/profile_training.py [--symbols AAPL,NVDA] [--epochs 2]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

PROFILES_DIR = PROJECT_ROOT / "profiles"
CUDA_PROFILE_SCRIPT = Path("/nvme0n1-disk/code/dotfiles/cuda_profile_to_md.py")

# Stocks that have both CSV price data and forecast cache parquets available
DEFAULT_SYMBOLS = ["AAPL", "NVDA", "DBX"]
STOCK_DATA_ROOT = PROJECT_ROOT / "trainingdatahourly" / "stocks"
STOCK_CACHE_ROOT = PROJECT_ROOT / "unified_hourly_experiment" / "forecast_cache"


def _check_symbol_data(symbol: str) -> bool:
    """Return True if both price CSV and h1 forecast cache exist for symbol."""
    csv = STOCK_DATA_ROOT / f"{symbol}.csv"
    parquet = STOCK_CACHE_ROOT / "h1" / f"{symbol}.parquet"
    return csv.exists() and parquet.exists()


def _select_symbols(requested: list[str]) -> list[str]:
    available = [s for s in requested if _check_symbol_data(s)]
    if not available:
        raise FileNotFoundError(
            f"None of {requested} have both price CSV and forecast cache. "
            f"Check {STOCK_DATA_ROOT} and {STOCK_CACHE_ROOT}/h1/"
        )
    return available


def run_neural_profiling(
    symbols: list[str],
    epochs: int,
    profiles_dir: Path,
) -> None:
    """Run binanceneural training under torch.profiler, write outputs."""
    import torch
    from torch.profiler import profile, ProfilerActivity, schedule

    profiles_dir.mkdir(parents=True, exist_ok=True)
    trace_path = profiles_dir / "neural_cuda_trace.json"
    report_path = profiles_dir / "neural_report.md"

    available_symbols = _select_symbols(symbols)
    print(f"[neural profiler] symbols={available_symbols}")
    print(f"[neural profiler] epochs={epochs}, output={profiles_dir}")

    from binanceneural.config import DatasetConfig, TrainingConfig
    from binanceneural.data import MultiSymbolDataModule
    from binanceneural.trainer import BinanceHourlyTrainer

    dataset_config = DatasetConfig(
        symbol=available_symbols[0],
        data_root=str(STOCK_DATA_ROOT),
        forecast_cache_root=str(STOCK_CACHE_ROOT),
        forecast_horizons=(1,),
        sequence_length=32,
        val_fraction=0.15,
        min_history_hours=100,
        validation_days=30,
        cache_only=True,
    )

    print(f"[neural profiler] Loading data for {available_symbols} ...")
    t0 = time.perf_counter()
    data_module = MultiSymbolDataModule(available_symbols, dataset_config)
    print(
        f"[neural profiler] Data loaded in {time.perf_counter() - t0:.1f}s "
        f"({len(data_module.train_dataset)} train samples, "
        f"{len(data_module.feature_columns)} features)"
    )

    train_config = TrainingConfig(
        epochs=epochs,
        batch_size=32,
        sequence_length=32,
        learning_rate=1e-4,
        weight_decay=1e-4,
        grad_clip=1.0,
        transformer_dim=256,
        transformer_layers=4,
        transformer_heads=8,
        transformer_dropout=0.1,
        return_weight=0.15,
        fill_temperature=5e-4,
        maker_fee=0.001,
        max_leverage=2.0,
        decision_lag_bars=1,
        fill_buffer_pct=0.0005,
        validation_use_binary_fills=True,
        use_amp=False,      # keep simple for profiling
        use_tf32=True,
        use_flash_attention=True,
        use_compile=False,  # skip compile to measure base kernels
        run_name="profile_run",
        checkpoint_root=Path("/tmp/neural_profile_checkpoints"),
        seed=42,
        num_workers=0,
        dry_train_steps=20,  # cap each epoch at 20 steps
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[neural profiler] device={device}")

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Profiler schedule: skip=2, wait=1, warmup=1, active=5, repeat=1
    prof_schedule = schedule(skip_first=2, wait=1, warmup=1, active=5, repeat=1)

    print("[neural profiler] Starting profiled training ...")
    t_start = time.perf_counter()

    trainer = BinanceHourlyTrainer(train_config, data_module)

    # Monkey-patch _run_epoch to call prof.step() after each batch
    # We inject profiler stepping by wrapping the original method
    _orig_run_epoch = trainer._run_epoch

    # We'll store the profiler in a mutable container so the closure can access it
    profiler_ref: list = [None]
    step_counter: list[int] = [0]

    def _patched_run_epoch(model, loader, optimizer, *, train, global_step, current_epoch=1):
        prof = profiler_ref[0]
        if prof is None:
            return _orig_run_epoch(
                model, loader, optimizer,
                train=train, global_step=global_step, current_epoch=current_epoch,
            )

        # We can't easily intercept batch-level steps in the original _run_epoch,
        # so we call it normally and step the profiler once per epoch.
        # This is sufficient to capture kernel activity across the epoch.
        result = _orig_run_epoch(
            model, loader, optimizer,
            train=train, global_step=global_step, current_epoch=current_epoch,
        )
        step_counter[0] += 1
        prof.step()
        return result

    trainer._run_epoch = _patched_run_epoch  # type: ignore[method-assign]

    with profile(
        activities=activities,
        schedule=prof_schedule,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        profiler_ref[0] = prof
        artifacts = trainer.train()

    elapsed = time.perf_counter() - t_start
    print(f"[neural profiler] Training complete in {elapsed:.1f}s")
    print(f"[neural profiler] Profiler steps: {step_counter[0]}")

    print(f"[neural profiler] Exporting trace to {trace_path} ...")
    prof.export_chrome_trace(str(trace_path))

    # Handle torch profiler .tmp rename (torch writes .json.tmp then renames)
    tmp_trace = profiles_dir / (trace_path.name + ".tmp")
    if not trace_path.exists() and tmp_trace.exists() and tmp_trace.stat().st_size > 0:
        print(f"[neural profiler] Renaming {tmp_trace.name} -> {trace_path.name}")
        tmp_trace.rename(trace_path)

    _generate_md_report(str(trace_path), str(report_path))
    _print_summary(trace_path, report_path)


def _generate_md_report(trace_path: str, report_path: str) -> None:
    """Generate markdown report from Chrome trace using cuda_profile_to_md.py."""
    if CUDA_PROFILE_SCRIPT.exists():
        result = subprocess.run(
            [
                sys.executable,
                str(CUDA_PROFILE_SCRIPT),
                "--trace", trace_path,
                "--output", report_path,
                "--top", "20",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[neural profiler] cuda_profile_to_md stderr: {result.stderr[:500]}")
    else:
        sys.path.insert(0, "/nvme0n1-disk/code/dotfiles")
        try:
            from cuda_profile_to_md import trace_to_markdown
            md = trace_to_markdown(trace_path, top_n=20)
            Path(report_path).write_text(md)
        except ImportError:
            Path(report_path).write_text(
                "# Neural CUDA Profiling Report\n\n_cuda_profile_to_md not available._\n"
            )


def _print_summary(trace_path: Path, report_path: Path) -> None:
    print("\n=== Neural Profiling Outputs ===")
    for path, label in [
        (trace_path, "Chrome trace (chrome://tracing)"),
        (report_path, "Markdown report"),
    ]:
        exists = Path(path).exists()
        size = Path(path).stat().st_size if exists else 0
        status = f"{size:,} bytes" if exists else "NOT GENERATED"
        print(f"  {label}: {path} [{status}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile binanceneural training")
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help=f"Comma-separated stock symbols (default: {','.join(DEFAULT_SYMBOLS)})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs to profile (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROFILES_DIR,
        help="Directory for profile outputs (default: profiles/)",
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    run_neural_profiling(symbols, args.epochs, args.output_dir)


if __name__ == "__main__":
    main()
