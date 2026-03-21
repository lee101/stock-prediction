#!/usr/bin/env python3
"""Profile pufferlib RL training for 60 seconds.

Outputs:
  profiles/pufferlib_cuda_trace.json  — Chrome trace for chrome://tracing
  profiles/pufferlib_report.md        — top-20 GPU kernels by time
  profiles/pufferlib_flamegraph.svg   — py-spy flamegraph (if py-spy available)

Usage:
  cd /nvme0n1-disk/code/stock-prediction
  source .venv313/bin/activate
  python pufferlib_market/profile_training.py [--data-path ...] [--duration 60]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
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


def run_pufferlib_profiling(data_path: str, duration_seconds: int, profiles_dir: Path) -> None:
    """Run pufferlib training under torch.profiler and py-spy, write outputs."""
    profiles_dir.mkdir(parents=True, exist_ok=True)
    trace_path = profiles_dir / "pufferlib_cuda_trace.json"
    report_path = profiles_dir / "pufferlib_report.md"
    flamegraph_path = profiles_dir / "pufferlib_flamegraph.svg"

    print(f"[pufferlib profiler] data={data_path}")
    print(f"[pufferlib profiler] duration={duration_seconds}s, output={profiles_dir}")

    # Build the inline training+profiling script
    profiler_script = _build_pufferlib_profiler_script(data_path, str(trace_path), duration_seconds)

    # Write script to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(profiler_script)
        script_path = f.name

    try:
        py_exe = sys.executable
        cmd = [py_exe, script_path]

        # Attempt to use py-spy for flamegraph if available
        pyspy = _find_pyspy()
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
            print("[pufferlib profiler] py-spy not found, skipping flamegraph")
            print(f"[pufferlib profiler] Running training script ...")
            t0 = time.perf_counter()
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            elapsed = time.perf_counter() - t0

        print(f"[pufferlib profiler] Training finished in {elapsed:.1f}s (exit={result.returncode})")

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

        _print_summary(trace_path, report_path, flamegraph_path if pyspy else None)

    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


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
    args = parser.parse_args()

    data_path = args.data_path or find_data_file(PROJECT_ROOT)
    run_pufferlib_profiling(data_path, args.duration, args.output_dir)


if __name__ == "__main__":
    main()
