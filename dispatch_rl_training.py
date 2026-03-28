#!/usr/bin/env python3
"""Local-first → remote dispatch for RL autoresearch experiments.

Routes to the local GPU when sufficient VRAM is available, otherwise
provisions a RunPod pod and runs the experiment remotely.

Usage:
    python scripts/dispatch_rl_training.py \\
        --data-train pufferlib_market/data/crypto10_daily_train.bin \\
        --data-val   pufferlib_market/data/crypto10_daily_val.bin \\
        --dry-run --gpu-type a100
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gpu_utils import detect_total_vram_bytes, get_gpu_name  # noqa: E402
from src.runpod_client import (  # noqa: E402
    GPU_ALIASES,
    HOURLY_RATES,
    TRAINING_GPU_TYPES,
    PodConfig,
    RunPodClient,
    resolve_gpu_type,
)
from src.remote_training_pipeline import DEFAULT_REMOTE_ENV  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_REMOTE_WORKSPACE = "/workspace/stock-prediction"
DEFAULT_VRAM_THRESHOLD_GB = 16.0
DEFAULT_TIME_BUDGET = 300
DEFAULT_MAX_TRIALS = 50
DEFAULT_WANDB_PROJECT = "stock"

_STOCKS_DEFAULT_TRAIN = "pufferlib_market/data/stocks12_daily_train.bin"
_STOCKS_DEFAULT_VAL = "pufferlib_market/data/stocks12_daily_val.bin"
_STOCKS_FEE_RATE = 0.001
_STOCKS_MAX_STEPS = 252
_STOCKS_PERIODS_PER_YEAR = 252.0
_STOCKS_HOLDOUT_EVAL_STEPS = 90

# SSH options used consistently across all remote calls.
_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]

_SETUP_OVERHEAD_SECS = 1800  # code sync + bootstrap + teardown


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_cost(gpu_type: str, num_seeds: int = 1, time_budget_secs: int = 300) -> float:
    """Estimate total cost for a training run including setup overhead.

    Args:
        gpu_type: Full GPU display name or short alias (e.g. "5090", "a100").
        num_seeds: Number of sequential seeds / trials.
        time_budget_secs: Seconds per seed.

    Returns:
        Estimated cost in USD.
    """
    resolved = GPU_ALIASES.get(gpu_type.lower(), gpu_type)
    rate = HOURLY_RATES.get(resolved, HOURLY_RATES.get(gpu_type, 0.0))
    total_secs = _SETUP_OVERHEAD_SECS + num_seeds * time_budget_secs
    return rate * (total_secs / 3600)


def _print_cost_estimate(gpu_type: str, num_seeds: int, time_budget_secs: int) -> float:
    """Print a formatted cost estimate and return the USD amount."""
    resolved = GPU_ALIASES.get(gpu_type.lower(), gpu_type)
    rate = HOURLY_RATES.get(resolved, HOURLY_RATES.get(gpu_type, 0.0))
    setup_min = _SETUP_OVERHEAD_SECS // 60
    training_min = num_seeds * time_budget_secs // 60
    total_min = (_SETUP_OVERHEAD_SECS + num_seeds * time_budget_secs) // 60
    est_cost = estimate_cost(gpu_type, num_seeds, time_budget_secs)

    # Friendly GPU name for display
    display_name = resolved if resolved != gpu_type else GPU_ALIASES.get(gpu_type, gpu_type)
    print("Cost estimate:")
    print(f"  GPU: {display_name} @ ${rate:.2f}/hr")
    print(f"  Setup overhead: {setup_min}min")
    print(f"  Training: {num_seeds} seed x {time_budget_secs // 60}min = {training_min}min")
    print(f"  Total: {total_min}min -> ~${est_cost:.2f}")
    return est_cost


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------


def should_run_remote(args: argparse.Namespace) -> bool:
    """Return True when the experiment should be dispatched to RunPod."""
    if args.force_remote:
        return True
    if args.gpu_type and args.gpu_type != "local":
        return True
    try:
        vram_bytes = detect_total_vram_bytes()
        if vram_bytes is None:
            return False
        vram_gb = vram_bytes / 1e9
        return vram_gb < args.vram_threshold_gb
    except Exception:
        return False  # assume local GPU available on error


def _detect_vram_gb() -> float | None:
    """Return local VRAM in GB, or None if unavailable."""
    try:
        vram_bytes = detect_total_vram_bytes()
        if vram_bytes is None:
            return None
        return vram_bytes / 1e9
    except Exception:
        return None


# ---------------------------------------------------------------------------
# rsync helper (mirrors pattern in launch_mixed23_retrain.py)
# ---------------------------------------------------------------------------


def _rsync_to_pod(
    ssh_host: str,
    ssh_port: int,
    local_dir: Path,
    remote_dir: str,
) -> None:
    """Sync the repo to the remote pod, excluding large / generated directories."""
    cmd = [
        "rsync",
        "-az",
        "--delete",
        "--exclude", "__pycache__/",
        "--exclude", ".git/",
        "--exclude", "pufferlib_market/data/",
        "--exclude", "pufferlib_market/checkpoints/",
        "--exclude", ".venv*/",
        "--exclude", "*.pyc",
        "-e", f"ssh {' '.join(_SSH_OPTS)} -p {ssh_port}",
        f"{local_dir}/",
        f"root@{ssh_host}:{remote_dir}/",
    ]
    print(f"[rsync] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)  # no timeout — may take time for large repos


def _rsync_data_file(
    ssh_host: str,
    ssh_port: int,
    local_path: Path,
    remote_dir: str,
    relative_to: Path,
) -> None:
    """Upload a single data binary to the pod, preserving relative directory structure."""
    try:
        rel = local_path.relative_to(relative_to)
    except ValueError:
        rel = Path(local_path.name)
    remote_path = f"root@{ssh_host}:{remote_dir}/{rel}"
    remote_parent = str(Path(remote_dir) / rel.parent)
    # Ensure remote directory exists first.
    _ssh_run(ssh_host, ssh_port, f"mkdir -p {remote_parent}")
    cmd = [
        "rsync", "-az",
        "-e", f"ssh {' '.join(_SSH_OPTS)} -p {ssh_port}",
        str(local_path),
        remote_path,
    ]
    print(f"[rsync data] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _ssh_run(ssh_host: str, ssh_port: int, remote_cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command on the remote pod via SSH."""
    cmd = [
        "ssh",
        *_SSH_OPTS,
        "-p", str(ssh_port),
        f"root@{ssh_host}",
        remote_cmd,
    ]
    print(f"[ssh] {remote_cmd}")
    return subprocess.run(cmd, check=check)  # no timeout — training may run for hours


def _scp_from_pod(
    ssh_host: str,
    ssh_port: int,
    remote_path: str,
    local_path: Path,
) -> bool:
    """Download a file from the pod; return True on success."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp",
        *_SSH_OPTS,
        "-P", str(ssh_port),
        f"root@{ssh_host}:{remote_path}",
        str(local_path),
    ]
    print(f"[scp] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Local run
# ---------------------------------------------------------------------------


def run_local(args: argparse.Namespace, *, seed: int | None = None) -> int:
    """Run autoresearch_rl directly on this machine."""
    leaderboard = args.leaderboard or f"analysis/{args.run_id}_leaderboard.csv"
    checkpoint_dir = args.checkpoint_dir or f"pufferlib_market/checkpoints/{args.run_id}"

    cmd = [
        sys.executable, "-u", "-m", "pufferlib_market.autoresearch_rl",
        "--train-data", args.data_train,
        "--val-data", args.data_val,
        "--time-budget", str(args.time_budget),
        "--max-trials", str(args.max_trials),
        "--leaderboard", leaderboard,
        "--checkpoint-root", checkpoint_dir,
        "--wandb-project", args.wandb_project,
    ]
    if args.stocks:
        cmd += ["--stocks"]
    if args.descriptions:
        cmd += ["--descriptions", args.descriptions]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    print(f"[local] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO))  # no timeout
    return result.returncode


# ---------------------------------------------------------------------------
# Remote run
# ---------------------------------------------------------------------------


def _select_gpu_type(args: argparse.Namespace) -> str:
    """Return the full GPU display name to provision."""
    alias = args.gpu_type or "a100"
    return resolve_gpu_type(alias)


def _estimate_hours(args: argparse.Namespace) -> float:
    """Rough upper-bound estimate: trials × budget seconds, plus overhead."""
    training_s = args.max_trials * args.time_budget
    return (training_s + _SETUP_OVERHEAD_SECS) / 3600


def _build_remote_autoresearch_cmd(
    args: argparse.Namespace,
    remote_dir: str,
    remote_leaderboard: str,
    remote_checkpoints: str,
    *,
    seed: int | None = None,
) -> str:
    """Build the shell command string to run autoresearch_rl on the remote pod."""
    parts = [
        "python", "-u", "-m", "pufferlib_market.autoresearch_rl",
        "--train-data", args.data_train,
        "--val-data", args.data_val,
        "--time-budget", str(args.time_budget),
        "--max-trials", str(args.max_trials),
        "--leaderboard", remote_leaderboard,
        "--checkpoint-root", remote_checkpoints,
        "--wandb-project", args.wandb_project,
    ]
    if args.stocks:
        parts += ["--stocks"]
    if args.descriptions:
        parts += ["--descriptions", args.descriptions]
    if seed is not None:
        parts += ["--seed", str(seed)]
    return " ".join(shlex.quote(str(p)) for p in parts)


def run_remote(args: argparse.Namespace, *, seed: int | None = None) -> int:
    """Provision a RunPod pod and execute autoresearch_rl on it."""
    gpu_type = _select_gpu_type(args)
    remote_dir = DEFAULT_REMOTE_WORKSPACE
    remote_env = DEFAULT_REMOTE_ENV
    remote_leaderboard = f"pufferlib_market/{args.run_id}_leaderboard.csv"
    remote_checkpoints = f"pufferlib_market/checkpoints/{args.run_id}"

    # Budget check before provisioning.
    budget_limit: float = getattr(args, "budget_limit", 5.0)
    est_cost = _print_cost_estimate(gpu_type, num_seeds=1, time_budget_secs=args.time_budget)
    if budget_limit > 0 and est_cost > budget_limit:
        print(f"Estimated cost: ${est_cost:.2f} exceeds budget limit ${budget_limit:.2f}")
        print(
            f"Use --budget-limit {est_cost + 0.5:.1f} to allow this, "
            f"or --budget-limit 0 to disable limit."
        )
        raise SystemExit(1)

    # 1. Provision pod.
    try:
        client = RunPodClient()
    except ValueError as exc:
        print(f"[error] Cannot create RunPod client: {exc}")
        print("  Set RUNPOD_API_KEY environment variable and retry.")
        return 1

    pod_name = f"rl-dispatch-{args.run_id}"
    print(f"[remote] Provisioning pod name={pod_name!r} gpu={gpu_type!r}")
    config = PodConfig(name=pod_name, gpu_type=gpu_type)
    pod = client.create_pod(config)
    print(f"[remote] Created pod id={pod.id}")

    # 2. Wait for SSH.
    print("[remote] Waiting for pod to become ready ...")
    pod = client.wait_for_pod(pod.id)
    ssh_host = pod.ssh_host
    ssh_port = pod.ssh_port
    print(f"[remote] Pod ready: {ssh_host}:{ssh_port}")

    try:
        # 3. rsync code.
        print("[remote] Syncing repo code ...")
        _rsync_to_pod(ssh_host, ssh_port, REPO, remote_dir)

        # 4. Upload data files.
        for data_path_str in (args.data_train, args.data_val):
            data_path = Path(data_path_str)
            if not data_path.is_absolute():
                data_path = REPO / data_path
            if data_path.exists():
                print(f"[remote] Uploading data: {data_path}")
                _rsync_data_file(ssh_host, ssh_port, data_path, remote_dir, REPO)
            else:
                print(f"[warning] Data file not found locally, skipping upload: {data_path}")

        # 5. Bootstrap: create venv + install dependencies + build C ext.
        bootstrap_cmd = (
            f"set -euo pipefail && "
            f"cd {remote_dir} && "
            f"pip install uv -q && "
            f"uv venv {remote_env} --python python3.13 2>/dev/null || uv venv {remote_env} && "
            f"source {remote_env}/bin/activate && "
            f"uv pip install -e . -q && "
            f"{{ [ -d PufferLib ] && uv pip install -e PufferLib/ -q || true; }} && "
            f"cd pufferlib_market && python setup.py build_ext --inplace -q && cd .."
        )
        _ssh_run(ssh_host, ssh_port, bootstrap_cmd)

        # 6. Set WANDB_API_KEY on remote if available locally.
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        wandb_export = f"export WANDB_API_KEY={wandb_key} && " if wandb_key else ""

        # 7. Run autoresearch_rl.
        autoresearch_cmd = _build_remote_autoresearch_cmd(
            args, remote_dir, remote_leaderboard, remote_checkpoints, seed=seed
        )
        run_cmd = (
            f"cd {remote_dir} && "
            f"source {remote_env}/bin/activate && "
            f"export PYTHONPATH={remote_dir}:${{PYTHONPATH:-}} && "
            f"{wandb_export}"
            f"{autoresearch_cmd}"
        )
        result = _ssh_run(ssh_host, ssh_port, run_cmd, check=False)
        exit_code = result.returncode

        # 8. Download leaderboard CSV.
        local_leaderboard = Path(
            args.leaderboard or f"analysis/{args.run_id}_leaderboard.csv"
        )
        ok = _scp_from_pod(
            ssh_host, ssh_port,
            f"{remote_dir}/{remote_leaderboard}",
            local_leaderboard,
        )
        if ok:
            print(f"[remote] Leaderboard saved to: {local_leaderboard}")
        else:
            print(f"[warning] Could not download leaderboard from pod.")

        # 9. Download top-5 checkpoints.
        local_checkpoint_dir = Path(
            args.checkpoint_dir or f"pufferlib_market/checkpoints/{args.run_id}"
        )
        print(f"[remote] Downloading checkpoints to {local_checkpoint_dir} ...")
        local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        rsync_checkpoints_cmd = [
            "rsync", "-az",
            "-e", f"ssh {' '.join(_SSH_OPTS)} -p {ssh_port}",
            f"root@{ssh_host}:{remote_dir}/{remote_checkpoints}/",
            f"{local_checkpoint_dir}/",
        ]
        subprocess.run(rsync_checkpoints_cmd, check=False)

        # 10. Upload to R2 if configured.
        r2_endpoint = os.environ.get("R2_ENDPOINT", "")
        if r2_endpoint:
            print(f"[remote] R2_ENDPOINT set — uploading leaderboard to R2 ...")
            _upload_to_r2(local_leaderboard, args.run_id, r2_endpoint)

        return exit_code

    finally:
        print(f"[remote] Terminating pod {pod.id} ...")
        try:
            client.terminate_pod(pod.id)
        except Exception as exc:
            print(f"[warning] Failed to terminate pod {pod.id}: {exc}")


def _upload_to_r2(local_path: Path, run_id: str, r2_endpoint: str) -> None:
    """Upload a file to R2 using the aws CLI (best-effort)."""
    if not local_path.exists():
        print(f"[r2] File not found, skipping: {local_path}")
        return
    bucket = os.environ.get("R2_BUCKET", "trading-experiments")
    key = f"rl-dispatch/{run_id}/{local_path.name}"
    cmd = [
        "aws", "s3", "cp",
        str(local_path),
        f"s3://{bucket}/{key}",
        "--endpoint-url", r2_endpoint,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print(f"[r2] Uploaded: s3://{bucket}/{key}")
    else:
        print(f"[r2] Upload failed (exit {result.returncode}), continuing.")


# ---------------------------------------------------------------------------
# Multi-seed helpers
# ---------------------------------------------------------------------------

# Default seed pool — first N entries are used when --num-seeds is given.
DEFAULT_SEEDS = [42, 123, 7, 99, 17]


def _resolve_seeds(args: argparse.Namespace) -> list[int]:
    """Return the list of seeds to train with."""
    if getattr(args, "seeds", None):
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    num = max(1, getattr(args, "num_seeds", 1))
    return DEFAULT_SEEDS[:num]


def _read_leaderboard_metrics(path: Path) -> dict[str, float | None]:
    """Return the best val_return and val_sortino from a leaderboard CSV."""
    if not path.exists():
        return {"val_return": None, "val_sortino": None}
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return {"val_return": None, "val_sortino": None}

        def _sort_key(r: dict) -> float:
            for col in ("rank_score", "val_return"):
                v = r.get(col)
                if v not in (None, "", "None"):
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
            return -float("inf")

        rows.sort(key=_sort_key, reverse=True)
        best = rows[0]

        def _maybe_float(v: object) -> float | None:
            if v in (None, "", "None"):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        return {
            "val_return": _maybe_float(best.get("val_return")),
            "val_sortino": _maybe_float(best.get("val_sortino")),
        }
    except Exception:
        return {"val_return": None, "val_sortino": None}


def _print_variance_report(
    seed_results: list[dict],
    multiseed_leaderboard: Path,
) -> None:
    """Print a variance summary across seed runs and save an aggregated CSV."""
    print("\nSeed Results Summary")
    print("====================")
    for entry in seed_results:
        seed = entry["seed"]
        vr = entry["val_return"]
        vs = entry["val_sortino"]
        vr_str = f"{vr:+.1%}" if vr is not None else "n/a"
        vs_str = f"{vs:.2f}" if vs is not None else "n/a"
        print(f"Seed {seed:<6d}: val_return={vr_str}  sortino={vs_str}")

    returns = [e["val_return"] for e in seed_results if e["val_return"] is not None]
    sortinos = [e["val_sortino"] for e in seed_results if e["val_sortino"] is not None]

    print()
    if returns:
        mean_r = statistics.mean(returns)
        std_r = statistics.stdev(returns) if len(returns) > 1 else 0.0
        print(f"Mean val_return: {mean_r:+.1%} +/- {std_r:.1%}")
    else:
        print("Mean val_return: n/a")
    if sortinos:
        mean_s = statistics.mean(sortinos)
        std_s = statistics.stdev(sortinos) if len(sortinos) > 1 else 0.0
        print(f"Mean sortino:    {mean_s:.2f} +/- {std_s:.2f}")
    else:
        print("Mean sortino:    n/a")
    print()
    print("Note: RL training is non-deterministic even with the same seed due to")
    print("GPU parallelism. The seed controls init and data shuffling only.")

    # Save aggregated leaderboard CSV.
    try:
        multiseed_leaderboard.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["seed", "val_return", "val_sortino", "leaderboard_path", "exit_code"]
        with open(multiseed_leaderboard, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in seed_results:
                writer.writerow(entry)
        print(f"Aggregated leaderboard saved to: {multiseed_leaderboard}")
    except Exception as exc:
        print(f"[warning] Could not save multiseed leaderboard: {exc}")


# ---------------------------------------------------------------------------
# Dry-run output
# ---------------------------------------------------------------------------


def print_dry_run_plan(
    args: argparse.Namespace, *, remote: bool, seeds: list[int] | None = None
) -> None:
    """Print a human-readable dispatch plan without executing anything."""
    gpu_type = _select_gpu_type(args) if remote else get_gpu_name() or "local GPU"
    vram_gb = _detect_vram_gb()
    est_hours = _estimate_hours(args)
    seeds = seeds or [DEFAULT_SEEDS[0]]

    if remote:
        hourly_rate = HOURLY_RATES.get(gpu_type, 0.0)
        routing_reason = ""
        if args.force_remote:
            routing_reason = " (--force-remote)"
        elif args.gpu_type and args.gpu_type != "local":
            routing_reason = f" (--gpu-type {args.gpu_type})"
        elif vram_gb is not None:
            routing_reason = f" (local VRAM: {vram_gb:.1f} GB < threshold {args.vram_threshold_gb:.1f} GB)"
        else:
            routing_reason = " (no local GPU detected)"
        routing_str = f"REMOTE{routing_reason}"
    else:
        routing_str = f"LOCAL (VRAM: {vram_gb:.1f} GB)" if vram_gb is not None else "LOCAL"

    leaderboard = args.leaderboard or f"analysis/{args.run_id}_leaderboard.csv"
    checkpoint_dir = args.checkpoint_dir or f"pufferlib_market/checkpoints/{args.run_id}"

    print("[dry-run] Dispatch RL Training Plan")
    print(f"  Run ID:       {args.run_id}")
    print(f"  Mode:         {'stocks' if getattr(args, 'stocks', False) else 'crypto'}")
    print(f"  Data (train): {args.data_train}")
    print(f"  Data (val):   {args.data_val}")
    if args.stocks:
        print(f"  fee_rate:     {_STOCKS_FEE_RATE}")
        print(f"  max_steps:    {_STOCKS_MAX_STEPS}")
        print(f"  periods/year: {_STOCKS_PERIODS_PER_YEAR}")
        print(f"  holdout_eval_steps: {_STOCKS_HOLDOUT_EVAL_STEPS}")
    print(f"  Routing:      {routing_str}")
    print(f"  GPU:          {gpu_type}")
    print(f"  W&B project:  {args.wandb_project}")
    print(f"  Leaderboard:  {leaderboard}")
    print(f"  Checkpoints:  {checkpoint_dir}")
    print(f"  Seeds:        {seeds}")
    if args.descriptions:
        print(f"  Descriptions: {args.descriptions}")
    print()

    budget_limit: float = getattr(args, "budget_limit", 5.0)
    if remote:
        # Always print cost estimate (even on dry-run).
        est_cost = _print_cost_estimate(gpu_type, num_seeds=1, time_budget_secs=args.time_budget)

        if budget_limit > 0 and est_cost > budget_limit:
            print(f"Estimated cost: ${est_cost:.2f} exceeds budget limit ${budget_limit:.2f}")
            print(
                f"Use --budget-limit {est_cost + 0.5:.1f} to allow this, "
                f"or --budget-limit 0 to disable limit."
            )
        print()
        print("  Steps:")
        print(f"    1. Provision/reuse RunPod pod ({gpu_type})")
        print(f"    2. rsync code to {DEFAULT_REMOTE_WORKSPACE}")
        print(f"    3. bootstrap: pip install uv, uv pip install -e ., uv pip install -e PufferLib/, build_ext --inplace")
        print(f"    4. Upload data files ({args.data_train}, {args.data_val})")
        print(
            f"    5. autoresearch_rl: {args.max_trials} trials x {args.time_budget}s each "
            f"(~{est_hours:.1f}h estimated)"
        )
        print(f"    6. Download leaderboard -> {leaderboard}")
        print(f"    7. Download top checkpoints -> {checkpoint_dir}")
        r2_endpoint = os.environ.get("R2_ENDPOINT", "")
        if r2_endpoint:
            print(f"    8. Upload to R2 (R2_ENDPOINT={r2_endpoint})")
        else:
            print("    8. Upload to R2 (skipped -- R2_ENDPOINT not set)")
        print("    9. Terminate pod")
    else:
        print("  Steps:")
        print(
            f"    1. python -m pufferlib_market.autoresearch_rl "
            f"(--max-trials {args.max_trials} --time-budget {args.time_budget})"
        )
        print(f"    2. Leaderboard written to: {leaderboard}")
        print(f"    3. Checkpoints written to: {checkpoint_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_run_id() -> str:
    return time.strftime("rl_dispatch_%Y%m%d_%H%M%S")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dispatch an RL autoresearch experiment locally or to RunPod, "
            "routing based on available local VRAM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --stocks must be parsed before required check so we can make data paths optional.
    _stocks_mode = "--stocks" in (argv if argv is not None else sys.argv[1:])

    req = parser.add_argument_group("required")
    req.add_argument("--data-train", required=not _stocks_mode, default=None, metavar="PATH",
                     help="Training MKTD binary file (.bin)")
    req.add_argument("--data-val", required=not _stocks_mode, default=None, metavar="PATH",
                     help="Validation MKTD binary file (.bin)")

    opt = parser.add_argument_group("optional")
    opt.add_argument(
        "--stocks", action="store_true",
        help=(
            "Use stock-specific configs: stocks12_daily_{train,val}.bin as data defaults, "
            f"fee_rate={_STOCKS_FEE_RATE}, max_steps={_STOCKS_MAX_STEPS}, "
            f"periods_per_year={_STOCKS_PERIODS_PER_YEAR}, "
            f"holdout_eval_steps={_STOCKS_HOLDOUT_EVAL_STEPS}. "
            "Passes --stocks to autoresearch_rl on the remote pod."
        ),
    )
    opt.add_argument("--run-id", default=None, metavar="STR",
                     help="Experiment run ID (auto-generated if not set)")
    opt.add_argument("--time-budget", type=int, default=DEFAULT_TIME_BUDGET,
                     help="Seconds per trial")
    opt.add_argument("--max-trials", type=int, default=DEFAULT_MAX_TRIALS,
                     help="Max number of trials")
    opt.add_argument("--descriptions", default="", metavar="STR",
                     help="Comma-separated experiment names to run (default: all)")
    opt.add_argument(
        "--gpu-type",
        default="",
        choices=["", "local"] + sorted(TRAINING_GPU_TYPES.keys()),
        metavar="TYPE",
        help=f"Force GPU type: {', '.join(sorted(TRAINING_GPU_TYPES.keys()))}, local (default: auto)",
    )
    opt.add_argument("--force-remote", action="store_true",
                     help="Skip local GPU check, always use RunPod")
    opt.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT,
                     help="W&B project name")
    opt.add_argument("--checkpoint-dir", default="", metavar="PATH",
                     help="Local checkpoint output directory")
    opt.add_argument("--leaderboard", default="", metavar="PATH",
                     help="Local leaderboard CSV output path")
    opt.add_argument("--dry-run", action="store_true",
                     help="Print plan without executing")
    opt.add_argument(
        "--vram-threshold-gb", type=float, default=DEFAULT_VRAM_THRESHOLD_GB,
        help="Min local VRAM (GB) to run locally; below this -> RunPod",
    )
    opt.add_argument(
        "--budget-limit", type=float, default=5.0,
        help="Max USD to spend on this dispatch (setup + training). 0 = no limit.",
    )
    opt.add_argument(
        "--num-seeds", type=int, default=1,
        help="Number of seeds to train with. Seeds: [42, 123, 7, 99, 17][0:num_seeds]",
    )
    opt.add_argument(
        "--seeds", type=str, default=None, metavar="SEEDS",
        help="Comma-separated list of seeds, overrides --num-seeds. Example: 42,123,7",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # --stocks: apply data-path defaults if user did not supply them.
    if args.stocks:
        if not args.data_train:
            args.data_train = _STOCKS_DEFAULT_TRAIN
        if not args.data_val:
            args.data_val = _STOCKS_DEFAULT_VAL

    # Auto-generate run ID if not supplied.
    if not args.run_id:
        args.run_id = _default_run_id()

    remote = should_run_remote(args)

    if args.dry_run:
        seeds = _resolve_seeds(args)
        print_dry_run_plan(args, remote=remote, seeds=seeds)
        return 0

    seeds = _resolve_seeds(args)
    multi_seed = len(seeds) > 1

    if not multi_seed:
        # Single-seed path: preserve original behaviour.
        seed = seeds[0] if seeds else None
        if remote:
            return run_remote(args, seed=seed)
        else:
            return run_local(args, seed=seed)

    # Multi-seed path: run sequentially, collect results, print variance report.
    base_run_id = args.run_id
    base_leaderboard = args.leaderboard
    base_checkpoint_dir = args.checkpoint_dir

    seed_results: list[dict] = []
    last_exit_code = 0

    for seed in seeds:
        seed_run_id = f"{base_run_id}_seed{seed}"
        args.run_id = seed_run_id
        # Per-seed leaderboard path (so runs don't clobber each other).
        if base_leaderboard:
            lb_path = Path(base_leaderboard)
            args.leaderboard = str(lb_path.with_stem(f"{lb_path.stem}_seed{seed}"))
        else:
            args.leaderboard = f"analysis/{seed_run_id}_leaderboard.csv"
        if base_checkpoint_dir:
            args.checkpoint_dir = f"{base_checkpoint_dir}_seed{seed}"
        else:
            args.checkpoint_dir = ""

        print(f"\n{'='*60}")
        print(f"[multi-seed] Running seed {seed} (run_id={seed_run_id})")
        print(f"{'='*60}")

        if remote:
            exit_code = run_remote(args, seed=seed)
        else:
            exit_code = run_local(args, seed=seed)

        last_exit_code = exit_code
        leaderboard_path = Path(args.leaderboard)
        metrics = _read_leaderboard_metrics(leaderboard_path)
        seed_results.append({
            "seed": seed,
            "val_return": metrics["val_return"],
            "val_sortino": metrics["val_sortino"],
            "leaderboard_path": str(leaderboard_path),
            "exit_code": exit_code,
        })

    # Restore original run_id for the multiseed leaderboard name.
    args.run_id = base_run_id
    multiseed_leaderboard = Path(f"pufferlib_market/{base_run_id}_multiseed.csv")
    _print_variance_report(seed_results, multiseed_leaderboard)

    return last_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
