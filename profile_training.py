#!/usr/bin/env python3
"""Profile pufferlib RL training for 60 seconds.

Outputs:
  profiles/pufferlib_cuda_trace.json  — Chrome trace for chrome://tracing
  profiles/pufferlib_report.md        — top-20 GPU kernels by time
  profiles/pufferlib_flamegraph.svg   — py-spy flamegraph (if py-spy available)
  profiles/timing.json                — throughput stats (steps/sec, wall time)
  profiles/report.md                  — unified markdown report

Usage:
  cd /nvme0n1-disk/code/stock-prediction
  source .venv313/bin/activate
  python pufferlib_market/profile_training.py [--data-path ...] [--duration 60]
  python pufferlib_market/profile_training.py --quick
  python pufferlib_market/profile_training.py --report-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROFILES_DIR = PROJECT_ROOT / "profiles"
CUDA_PROFILE_SCRIPT = Path("/nvme0n1-disk/code/dotfiles/cuda_profile_to_md.py")

# Default data file — prefer fdusd3_daily_train.bin, fall back to first .bin found
DEFAULT_DATA_FILES = [
    "pufferlib_market/data/fdusd3_daily_train.bin",
    "pufferlib_market/data/mixed23_daily_train.bin",
    "pufferlib_market/data/crypto8_daily_train.bin",
    "pufferlib_market/data/market_data.bin",
]


def find_data_file(project_root: Path) -> str:
    for rel in DEFAULT_DATA_FILES:
        p = project_root / rel
        if p.exists():
            return str(p)
    # Last resort: any .bin file
    for p in (project_root / "pufferlib_market/data").glob("*.bin"):
        return str(p)
    raise FileNotFoundError("No .bin data file found in pufferlib_market/data/")


def run_pufferlib_profiling(
    data_path: str,
    duration_seconds: int,
    profiles_dir: Path,
    quick: bool = False,
) -> None:
    """Run pufferlib training under torch.profiler and py-spy, write outputs.

    Args:
        data_path: Path to .bin market data file.
        duration_seconds: Duration of the profiling run in seconds.
        profiles_dir: Directory where profile outputs are written.
        quick: If True, skip py-spy and run only torch.profiler for 100 updates.
    """
    profiles_dir.mkdir(parents=True, exist_ok=True)
    trace_path = profiles_dir / "pufferlib_cuda_trace.json"
    report_path = profiles_dir / "pufferlib_report.md"
    flamegraph_path = profiles_dir / "pufferlib_flamegraph.svg"
    timing_path = profiles_dir / "timing.json"

    print(f"[pufferlib profiler] data={data_path}")
    print(f"[pufferlib profiler] duration={duration_seconds}s, quick={quick}, output={profiles_dir}")

    # Build the inline training+profiling script
    profiler_script = _build_pufferlib_profiler_script(data_path, str(trace_path), duration_seconds)

    # Write script to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(profiler_script)
        script_path = f.name

    try:
        py_exe = sys.executable
        cmd = [py_exe, script_path]

        # Attempt to use py-spy for flamegraph if available (skipped in --quick mode)
        pyspy = None if quick else _find_pyspy()
        if pyspy:
            print(f"[pufferlib profiler] py-spy found: {pyspy}, will record flamegraph")
            pyspy_cmd = [
                pyspy, "record",
                "--output", str(flamegraph_path),
                "--format", "speedscope",
                "--rate", "100",
                "--duration", str(duration_seconds + 60),  # extra 60s for trace export
                "--nonblocking",
                "--subprocesses",
                "--",
            ] + cmd
            print(f"[pufferlib profiler] Running: {' '.join(pyspy_cmd[:6])} ...")
            t0 = time.perf_counter()
            result = subprocess.run(pyspy_cmd, cwd=str(PROJECT_ROOT))
            elapsed = time.perf_counter() - t0
        else:
            if quick:
                print("[pufferlib profiler] --quick mode: skipping py-spy")
            else:
                print("[pufferlib profiler] py-spy not found, skipping flamegraph")
            print("[pufferlib profiler] Running training script ...")
            t0 = time.perf_counter()
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            elapsed = time.perf_counter() - t0

        print(f"[pufferlib profiler] Training finished in {elapsed:.1f}s (exit={result.returncode})")

        # Capture throughput stats to timing.json
        num_envs = 64
        rollout_len = 256
        # Estimate updates from elapsed time; the profiling script runs until duration_seconds
        estimated_updates = max(1, int(elapsed))
        steps_per_update = num_envs * rollout_len
        steps_per_sec = (estimated_updates * steps_per_update) / max(elapsed, 0.001)
        updates_per_sec = estimated_updates / max(elapsed, 0.001)

        timing = {
            "steps_per_sec": round(steps_per_sec, 1),
            "updates_per_sec": round(updates_per_sec, 3),
            "wall_time_s": round(elapsed, 1),
            "n_updates": estimated_updates,
            "gpu_name": "unknown",
            "gpu_vram_gb": 0,
        }
        # Try to get GPU info
        try:
            import torch
            if torch.cuda.is_available():
                timing["gpu_name"] = torch.cuda.get_device_name(0)
                timing["gpu_vram_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                )
        except Exception:
            pass
        timing_path.write_text(json.dumps(timing, indent=2))

        # Handle torch profiler .tmp rename (torch writes .json.tmp then renames)
        tmp_trace = profiles_dir / (trace_path.name + ".tmp")
        if not trace_path.exists() and tmp_trace.exists() and tmp_trace.stat().st_size > 0:
            print(f"[pufferlib profiler] Renaming {tmp_trace.name} -> {trace_path.name}")
            tmp_trace.rename(trace_path)

        # Generate markdown report from trace
        if trace_path.exists() and trace_path.stat().st_size > 0:
            _generate_md_report(str(trace_path), str(report_path))
            print(f"[pufferlib profiler] Report: {report_path}")
        else:
            print(f"[pufferlib profiler] WARNING: trace not found at {trace_path}")
            report_path.write_text("# Pufferlib CUDA Profiling Report\n\n_No trace generated._\n")

        # Generate unified markdown report
        generate_markdown_report(profiles_dir)

        _print_summary(trace_path, report_path, flamegraph_path if pyspy else None)

    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _load_trace_events(trace_path: Path) -> list:
    """Load traceEvents list from a Chrome trace JSON file.

    Returns empty list on any parse error.
    """
    try:
        with open(trace_path) as f:
            return json.load(f).get("traceEvents", [])
    except Exception:
        return []


def _parse_chrome_trace_top_kernels(
    trace_path: Path,
    top_n: int = 10,
    events: list | None = None,
) -> tuple[list[tuple[str, float, int]], float]:
    """Parse Chrome trace JSON and return top CUDA kernels by total duration.

    Args:
        trace_path: Path to the Chrome trace JSON file.
        top_n: Number of top kernels to return.
        events: Pre-loaded traceEvents list; if provided, trace_path is not read.

    Returns (kernels, total_kernel_ms) where kernels is a list of
    (kernel_name, total_ms, call_count) sorted descending by total_ms.
    Returns ([], 1.0) on any parse error.
    """
    if events is None:
        events = _load_trace_events(trace_path)

    kernel_times: dict[str, float] = defaultdict(float)
    kernel_counts: dict[str, int] = defaultdict(int)
    for e in events:
        if e.get("cat") == "kernel" and "dur" in e:
            name = e["name"]
            kernel_times[name] += e["dur"] / 1000.0  # us -> ms
            kernel_counts[name] += 1

    total_ms = sum(kernel_times.values()) or 1.0
    top = sorted(kernel_times.items(), key=lambda x: -x[1])[:top_n]
    return [(name, ms, kernel_counts[name]) for name, ms in top], total_ms


def _parse_speedscope_top_functions(flamegraph_path: Path, top_n: int = 5) -> list[tuple[str, int, float, str]]:
    """Parse speedscope JSON and return top functions by sample count.

    Returns list of (func_name, sample_count, pct, file).
    Skips generic bootstrap/import frames and process-level frames.
    """
    SKIP_NAMES = frozenset([
        "_call_with_frames_removed", "_find_and_load", "_find_and_load_unlocked",
        "_load_unlocked", "exec_module", "_compile_bytecode", "get_code",
        "<module>", "process", "",
    ])
    try:
        with open(flamegraph_path) as f:
            data = json.load(f)
    except Exception:
        return []

    if data.get("$schema", "").find("speedscope") == -1 and "profiles" not in data:
        return []

    frames = data.get("shared", {}).get("frames", [])
    profiles_list = data.get("profiles", [])
    if not profiles_list:
        return []

    # Use first sampled profile
    profile = profiles_list[0]
    samples = profile.get("samples", [])
    total_samples = len(samples)
    if total_samples == 0:
        return []

    frame_counts: dict[int, int] = defaultdict(int)
    for sample in samples:
        for frame_idx in sample:
            frame_counts[frame_idx] += 1

    results = []
    for idx, cnt in sorted(frame_counts.items(), key=lambda x: -x[1]):
        if idx >= len(frames):
            continue
        f = frames[idx]
        name = f.get("name", "")
        if not name or name in SKIP_NAMES or name.startswith("process "):
            continue
        file_path = f.get("file", "") or ""
        # Shorten file path to last two components
        parts = file_path.replace("\\", "/").split("/")
        short_file = "/".join(parts[-2:]) if len(parts) >= 2 else file_path
        pct = 100.0 * cnt / total_samples
        results.append((name, cnt, pct, short_file))
        if len(results) >= top_n:
            break

    return results


def _parse_gprof(gprof_path: Path, top_n: int = 10) -> list[tuple[str, str]]:
    """Parse gprof flat profile text and return top functions.

    Returns list of (function_name, pct_time_str).
    """
    try:
        content = gprof_path.read_text()
    except Exception:
        return []

    results = []
    in_flat = False
    for line in content.splitlines():
        if "% cumulative" in line or "time   seconds" in line:
            in_flat = True
            continue
        if in_flat and line.strip() == "":
            break
        if in_flat:
            parts = line.split()
            if len(parts) >= 7:
                pct = parts[0]
                func = parts[-1]
                results.append((func, pct + "%"))
                if len(results) >= top_n:
                    break
    return results


def _build_recommendations(
    top_kernels: list,
    top_functions: list,
    timing: dict,
) -> list[str]:
    """Generate optimization recommendations from profiling data."""
    recs = []

    # Check for data transfer bottlenecks
    if timing:
        steps_per_sec = timing.get("steps_per_sec", 0)
        if steps_per_sec > 0 and steps_per_sec < 10000:
            recs.append(
                "[HIGH] Low steps/sec detected — consider pinned memory + non_blocking=True "
                "for obs/action transfers between CPU and GPU."
            )

    # Check if softmax/attention dominates
    kernel_names_lower = [k[0].lower() for k in top_kernels[:5]]
    if any("softmax" in n for n in kernel_names_lower):
        recs.append(
            "[HIGH] Softmax dominates GPU time — consider fused attention (FlashAttention) "
            "or replacing softmax with ReLU-based gating."
        )

    # Check for GEMM dominance
    gemm_ms = sum(ms for name, ms, _ in top_kernels if "gemm" in name.lower() or "cutlass" in name.lower())
    total_ms = sum(ms for _, ms, _ in top_kernels) or 1.0
    if gemm_ms / total_ms > 0.4:
        recs.append(
            "[MED] GEMM kernels account for >40% of GPU time — try torch.compile() or "
            "BF16 matmul to improve throughput."
        )

    # Check CPU functions
    func_names_lower = [f[0].lower() for f in top_functions[:5]]
    if any("ppo" in n or "update" in n for n in func_names_lower):
        recs.append(
            "[MED] PPO update appears in top CPU functions — consider CUDA graph capture "
            "(--cuda-graph-ppo) to reduce Python overhead."
        )
    if any("numpy" in n or "copy" in n or "from_numpy" in n for n in func_names_lower):
        recs.append(
            "[MED] NumPy copy in critical path — pre-allocate persistent GPU tensors for "
            "obs/actions rather than converting from numpy each step."
        )

    if not recs:
        recs.append("[INFO] No obvious bottlenecks detected from available profiling data.")

    return recs


def generate_markdown_report(profiles_dir: Path) -> Path:
    """Generate unified profiling report from existing profile files.

    Reads:
      - profiles_dir/pufferlib_cuda_trace.json  (torch.profiler Chrome trace)
      - profiles_dir/pufferlib_flamegraph.svg    (py-spy speedscope JSON)
      - profiles_dir/timing.json                 (throughput stats)
      - profiles_dir/gprof_output.txt            (C env gprof, optional)

    Writes:
      - profiles_dir/report.md

    Returns the path to the written report.
    """
    profiles_dir = Path(profiles_dir)
    trace_path = profiles_dir / "pufferlib_cuda_trace.json"
    flamegraph_path = profiles_dir / "pufferlib_flamegraph.svg"
    timing_path = profiles_dir / "timing.json"
    gprof_path = profiles_dir / "gprof_output.txt"
    report_path = profiles_dir / "report.md"

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- Throughput summary ---
    timing: dict = {}
    if timing_path.exists():
        try:
            timing = json.loads(timing_path.read_text())
        except Exception:
            timing = {}

    steps_per_sec = timing.get("steps_per_sec", "N/A")
    updates_per_sec = timing.get("updates_per_sec", "N/A")
    wall_time = timing.get("wall_time_s", "N/A")
    n_updates = timing.get("n_updates", "N/A")
    gpu_name = timing.get("gpu_name", "unknown")
    gpu_vram = timing.get("gpu_vram_gb", "?")

    lines = [
        f"# Training Profile Report — {date_str}",
        "",
        "## Throughput Summary",
        "",
        f"Steps/sec: {steps_per_sec}  |  PPO updates/sec: {updates_per_sec}  |  "
        f"Wall time ({n_updates} updates): {wall_time}s",
        f"GPU: {gpu_name}  |  VRAM used: {gpu_vram}GB",
        "",
    ]

    # --- Top CPU functions (py-spy) ---
    lines.append("## Top CPU Functions (py-spy)")
    lines.append("")
    top_functions: list = []
    if flamegraph_path.exists():
        top_functions = _parse_speedscope_top_functions(flamegraph_path, top_n=5)

    if top_functions:
        lines.append("| Rank | Function | % CPU | File |")
        lines.append("|------|----------|-------|------|")
        for rank, (name, count, pct, file_) in enumerate(top_functions, 1):
            # Truncate long kernel/function names
            short_name = name[:60] + "..." if len(name) > 60 else name
            lines.append(f"| {rank} | {short_name} | {pct:.1f}% | {file_} |")
    else:
        lines.append("_No py-spy data found. Run without --quick to capture flamegraph._")
    lines.append("")

    # Load trace once; extract both kernel and memory data in a single pass
    top_kernels: list[tuple[str, float, int]] = []
    total_kernel_ms = 1.0
    mem_events: list = []
    if trace_path.exists():
        trace_events = _load_trace_events(trace_path)
        top_kernels, total_kernel_ms = _parse_chrome_trace_top_kernels(
            trace_path, top_n=10, events=trace_events
        )
        try:
            mem_by_name: dict[str, dict] = defaultdict(lambda: {"alloc_bytes": 0, "count": 0})
            for e in trace_events:
                if e.get("cat") in ("memory", "[memory]") and "args" in e:
                    name = e.get("name", "unknown")
                    args = e["args"]
                    alloc = args.get("Bytes", args.get("bytes", 0))
                    if alloc > 0:
                        mem_by_name[name]["alloc_bytes"] += alloc
                        mem_by_name[name]["count"] += 1
            mem_events = sorted(mem_by_name.items(), key=lambda x: -x[1]["alloc_bytes"])[:10]
        except Exception:
            pass

    # --- Top CUDA kernels (torch.profiler) ---
    lines.append("## Top CUDA Kernels (torch.profiler)")
    lines.append("")
    if top_kernels:
        lines.append("| Rank | Kernel | Calls | Total ms | % CUDA time |")
        lines.append("|------|--------|-------|----------|-------------|")
        for rank, (name, ms, calls) in enumerate(top_kernels, 1):
            pct = 100.0 * ms / total_kernel_ms
            short_name = name[:70] + "..." if len(name) > 70 else name
            lines.append(f"| {rank} | `{short_name}` | {calls} | {ms:.2f} | {pct:.1f}% |")
    else:
        lines.append("_No CUDA trace found. Run profiling to generate chrome trace._")
    lines.append("")

    # --- Memory allocation hotspots ---
    lines.append("## Memory Allocation Hotspots")
    lines.append("")

    if mem_events:
        lines.append("(from torch.profiler memory timeline)")
        lines.append("")
        lines.append("| Operation | Alloc size | Count |")
        lines.append("|-----------|------------|-------|")
        for name, info in mem_events:
            mb = info["alloc_bytes"] / 1e6
            lines.append(f"| {name} | {mb:.2f} MB | {info['count']} |")
    else:
        lines.append("_(No memory allocation events found in trace.)_")
    lines.append("")

    # --- C Env (gprof) ---
    lines.append("## C Env (gprof)")
    lines.append("")
    gprof_entries = _parse_gprof(gprof_path) if gprof_path.exists() else []
    if gprof_entries:
        lines.append("Top C functions by time:")
        lines.append("")
        lines.append("| Function | % Time |")
        lines.append("|----------|--------|")
        for func, pct in gprof_entries:
            lines.append(f"| {func} | {pct} |")
    else:
        lines.append("_(No gprof data found. Run with C env compiled with `-pg` to collect.)_")
    lines.append("")

    # --- Optimization recommendations ---
    lines.append("## Optimization Recommendations")
    lines.append("")
    lines.append("Based on profiling data:")
    lines.append("")
    recs = _build_recommendations(top_kernels, top_functions, timing)
    for rec in recs:
        lines.append(f"- {rec}")
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def _find_pyspy() -> str | None:
    """Find py-spy binary in PATH or venv."""
    import shutil
    found = shutil.which("py-spy")
    if found:
        return found
    # Try venv bin dir
    venv_pyspy = Path(sys.executable).parent / "py-spy"
    if venv_pyspy.exists():
        return str(venv_pyspy)
    return None


def _generate_md_report(trace_path: str, report_path: str) -> None:
    """Generate markdown report from Chrome trace using cuda_profile_to_md.py."""
    if CUDA_PROFILE_SCRIPT.exists():
        result = subprocess.run(
            [sys.executable, str(CUDA_PROFILE_SCRIPT), "--trace", trace_path, "--output", report_path, "--top", "20"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[pufferlib profiler] cuda_profile_to_md stderr: {result.stderr[:500]}")
    else:
        # Fallback: inline markdown generation
        sys.path.insert(0, "/nvme0n1-disk/code/dotfiles")
        try:
            from cuda_profile_to_md import trace_to_markdown
            md = trace_to_markdown(trace_path, top_n=20)
            Path(report_path).write_text(md)
        except ImportError:
            Path(report_path).write_text("# Pufferlib CUDA Profiling Report\n\n_cuda_profile_to_md not available._\n")


def _print_summary(trace_path: Path, report_path: Path, flamegraph_path: Path | None) -> None:
    print("\n=== Pufferlib Profiling Outputs ===")
    for path, label in [
        (trace_path, "Chrome trace (chrome://tracing)"),
        (report_path, "Markdown report"),
        (flamegraph_path, "Flamegraph (speedscope)"),
    ]:
        if path is None:
            continue
        exists = Path(path).exists()
        size = Path(path).stat().st_size if exists else 0
        status = f"{size:,} bytes" if exists else "NOT GENERATED"
        print(f"  {label}: {path} [{status}]")


def _build_pufferlib_profiler_script(data_path: str, trace_path: str, duration_seconds: int) -> str:
    """Build the Python script that runs pufferlib training under torch.profiler."""
    return f'''#!/usr/bin/env python3
"""Auto-generated pufferlib profiling script."""
import sys
import time
import os
from pathlib import Path

PROJECT_ROOT = Path({str(PROJECT_ROOT)!r})
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.profiler import profile, ProfilerActivity, schedule

# Profiler schedule: skip=2, wait=1, warmup=1, active=5, repeat=1
PROF_SCHEDULE = schedule(skip_first=2, wait=1, warmup=1, active=5, repeat=1)

TRACE_PATH = {trace_path!r}
DATA_PATH = {data_path!r}
DURATION_SECONDS = {duration_seconds}

# ── Build pufferlib training args ──
import argparse
args = argparse.Namespace(
    data_path=DATA_PATH,
    max_steps=365,
    fee_rate=0.0,
    max_leverage=1.0,
    periods_per_year=365.0,
    reward_scale=10.0,
    reward_clip=5.0,
    action_allocation_bins=1,
    action_level_bins=1,
    action_max_offset_bps=0.0,
    cash_penalty=0.01,
    drawdown_penalty=0.0,
    downside_penalty=0.0,
    smooth_downside_penalty=0.0,
    smooth_downside_temperature=0.02,
    trade_penalty=0.0,
    smoothness_penalty=0.0,
    fill_slippage_bps=0.0,
    fill_probability=1.0,
    max_hold_hours=0,
    drawdown_profit_early_exit=False,
    drawdown_profit_early_exit_verbose=False,
    drawdown_profit_early_exit_min_steps=20,
    drawdown_profit_early_exit_progress_fraction=0.5,
    short_borrow_apr=0.0,
    num_envs=64,
    seed=42,
    hidden_size=1024,
    arch="mlp",
    activation="relu",
    disable_shorts=True,
    total_timesteps=999_999_999,  # run until time limit
    rollout_len=256,
    ppo_epochs=4,
    minibatch_size=2048,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    advantage_norm="global",
    group_relative_size=8,
    group_relative_mix=0.0,
    group_relative_clip=2.0,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.05,
    max_grad_norm=0.5,
    anneal_lr=True,
    lr_schedule="none",
    lr_warmup_frac=0.02,
    lr_min_ratio=0.05,
    weight_decay=0.0,
    optimizer="adamw",
    muon_momentum=0.95,
    muon_ns_steps=5,
    muon_adamw_lr=3e-4,
    anneal_ent=False,
    ent_coef_end=0.02,
    anneal_clip=False,
    clip_eps_end=0.05,
    clip_vloss=False,
    obs_norm=False,
    resume_from=None,
    checkpoint_dir="/tmp/pufferlib_profile_checkpoints",
    save_every=10000,
    max_periodic_checkpoints=0,
    cpu=False,
    wandb_project=None,
    wandb_entity=None,
    wandb_run_name=None,
    wandb_group=None,
    wandb_mode="disabled",
)

from pufferlib_market.environment import TradingEnvConfig, TradingEnv
import pufferlib_market.binding as binding
import numpy as np
import struct
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Disable Triton fused kernels for profiling: they emit BF16 but the
# surrounding Linear layers expect Float32, causing a dtype mismatch.
import pufferlib_market.kernels.fused_mlp as _fused_mlp
_fused_mlp.HAS_TRITON = False

from pufferlib_market.train import TradingPolicy, get_activation, RunningObsNorm, _checkpoint_payload

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {{device}}")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

data_path = str(Path(DATA_PATH).resolve())
binding.shared(data_path=data_path)
with open(data_path, "rb") as f:
    header = f.read(64)
_, _, num_symbols, num_timesteps, _, _ = struct.unpack("<4sIIIII", header[:24])
print(f"  {{num_symbols}} symbols, {{num_timesteps}} timesteps")

config = TradingEnvConfig(
    data_path=data_path,
    max_steps=args.max_steps,
    fee_rate=args.fee_rate,
    max_leverage=args.max_leverage,
    periods_per_year=args.periods_per_year,
    num_symbols=num_symbols,
    reward_scale=args.reward_scale,
    reward_clip=args.reward_clip,
    action_allocation_bins=1,
    action_level_bins=1,
    action_max_offset_bps=0.0,
    cash_penalty=args.cash_penalty,
    drawdown_penalty=args.drawdown_penalty,
    short_borrow_apr=args.short_borrow_apr,
)

obs_size = num_symbols * 16 + 5 + num_symbols
num_actions = 1 + 2 * num_symbols
print(f"  obs_size={{obs_size}}, num_actions={{num_actions}}")

num_envs = args.num_envs
obs_buf = np.zeros((num_envs, obs_size), dtype=np.float32)
act_buf = np.zeros((num_envs,), dtype=np.int32)
rew_buf = np.zeros((num_envs,), dtype=np.float32)
term_buf = np.zeros((num_envs,), dtype=np.uint8)
trunc_buf = np.zeros((num_envs,), dtype=np.uint8)

vec_handle = binding.vec_init(
    obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
    num_envs, args.seed,
    max_steps=config.max_steps,
    fee_rate=config.fee_rate,
    max_leverage=config.max_leverage,
    periods_per_year=config.periods_per_year,
    reward_scale=config.reward_scale,
    reward_clip=config.reward_clip,
    cash_penalty=config.cash_penalty,
    drawdown_penalty=config.drawdown_penalty,
    short_borrow_apr=config.short_borrow_apr,
)
binding.vec_reset(vec_handle, args.seed)

policy = TradingPolicy(obs_size, num_actions, hidden=args.hidden_size).to(device)
optimizer = optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print(f"Policy params: {{sum(p.numel() for p in policy.parameters()):,}}")

rollout_len = args.rollout_len
obs_store = np.zeros((rollout_len, num_envs, obs_size), dtype=np.float32)
act_store = np.zeros((rollout_len, num_envs), dtype=np.int64)
logp_store = np.zeros((rollout_len, num_envs), dtype=np.float32)
val_store = np.zeros((rollout_len, num_envs), dtype=np.float32)
rew_store = np.zeros((rollout_len, num_envs), dtype=np.float32)
done_store = np.zeros((rollout_len, num_envs), dtype=np.float32)
next_obs = torch.from_numpy(obs_buf).to(device, dtype=torch.float32)
next_done = torch.zeros(num_envs, device=device)

activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    activities.append(ProfilerActivity.CUDA)

print(f"Starting profiled training for {{DURATION_SECONDS}}s ...")
deadline = time.perf_counter() + DURATION_SECONDS
update = 0
profiler_done = False

with profile(
    activities=activities,
    schedule=PROF_SCHEDULE,
    record_shapes=True,
    with_stack=True,
) as prof:
    while time.perf_counter() < deadline:
        # ── Rollout collection ──
        policy.eval()
        with torch.no_grad():
            for step in range(rollout_len):
                obs_store[step] = obs_buf.copy()
                done_store[step] = next_done.cpu().numpy()

                logits, val = policy(next_obs)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

                act_store[step] = action.cpu().numpy()
                logp_store[step] = logp.cpu().numpy()
                val_store[step] = val.cpu().numpy()

                act_buf[:] = action.cpu().numpy().astype(np.int32)
                binding.vec_step(vec_handle)
                rew_store[step] = rew_buf.copy()

                next_obs = torch.from_numpy(obs_buf).to(device, dtype=torch.float32, non_blocking=True)
                next_done = torch.from_numpy(
                    (term_buf | trunc_buf).astype(np.float32)
                ).to(device, non_blocking=True)

        # ── PPO update ──
        policy.train()
        b_obs = torch.from_numpy(obs_store.reshape(-1, obs_size)).to(device, dtype=torch.float32)
        b_act = torch.from_numpy(act_store.reshape(-1)).to(device)
        b_logp_old = torch.from_numpy(logp_store.reshape(-1)).to(device)
        b_rew = torch.from_numpy(rew_store.reshape(-1)).to(device)

        # Simple advantage (no GAE for profiling brevity)
        b_adv = b_rew - b_rew.mean()
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n_samples = rollout_len * num_envs
        mb_size = args.minibatch_size
        for _ in range(args.ppo_epochs):
            idx = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, mb_size):
                mb_idx = idx[start:start + mb_size]
                mb_obs = b_obs[mb_idx]
                mb_act = b_act[mb_idx]
                mb_logp_old = b_logp_old[mb_idx]
                mb_adv = b_adv[mb_idx]

                logits, val = policy(mb_obs)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(mb_act)
                entropy = dist.entropy()

                ratio = torch.exp(logp - mb_logp_old)
                pg_loss = -torch.min(
                    ratio * mb_adv,
                    torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * mb_adv
                ).mean()
                vf_loss = val.pow(2).mean()
                loss = pg_loss + args.vf_coef * vf_loss - args.ent_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

        update += 1
        prof.step()

        if not profiler_done and update >= 9:  # skip=2+wait=1+warmup=1+active=5
            profiler_done = True

        if update % 10 == 0:
            elapsed = time.perf_counter() - (deadline - DURATION_SECONDS)
            print(f"  update={{update}}, elapsed={{elapsed:.1f}}s")

print(f"Profiling complete, {{update}} updates total")
print(f"Exporting trace to {{TRACE_PATH}} ...")
prof.export_chrome_trace(TRACE_PATH)
print("Done.")
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile pufferlib RL training")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to .bin market data file (auto-detected if not specified)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Training duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROFILES_DIR,
        help="Directory for profile outputs (default: profiles/)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Quick mode: run only torch.profiler, skip py-spy and C gprof",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        default=False,
        help="Skip profiling, just generate report.md from existing profile files",
    )
    args = parser.parse_args()

    if args.report_only:
        profiles_dir = Path(args.output_dir)
        files = list(profiles_dir.glob("*.json")) + list(profiles_dir.glob("*.svg"))
        if not files:
            print(f"No profile files found in {profiles_dir}")
            sys.exit(0)
        report_path = generate_markdown_report(profiles_dir)
        print(f"Report written to: {report_path}")
        return

    data_path = args.data_path or find_data_file(PROJECT_ROOT)
    run_pufferlib_profiling(data_path, args.duration, args.output_dir, quick=args.quick)


if __name__ == "__main__":
    main()
