"""
GPU job pool for RL training experiments.

Runs one training trial per available GPU (serial on 1 GPU, parallel on N).
Jobs are stored in a JSONL queue file; workers claim jobs atomically via
fcntl.flock and write results to a shared leaderboard CSV.

Usage
-----
# 1) Generate a queue of jobs and launch workers:
python -m pufferlib_market.gpu_pool run \\
    --queue /tmp/pool_queue.jsonl \\
    --train-data pufferlib_market/data/stocks12_daily_train.bin \\
    --val-data   pufferlib_market/data/stocks12_daily_val.bin \\
    --leaderboard sweepresults/pool_leaderboard.csv \\
    --checkpoint-dir pufferlib_market/checkpoints/pool \\
    --preset stocks12_seedsweep \\
    --time-budget 300 \\
    [--gpu-ids 0,1,2]   # default: all detected GPUs

# 2) Add more jobs to a running pool:
python -m pufferlib_market.gpu_pool add \\
    --queue /tmp/pool_queue.jsonl \\
    --preset stocks12_tp05_family

# 3) Show queue status:
python -m pufferlib_market.gpu_pool status --queue /tmp/pool_queue.jsonl
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Re-use TrialConfig + run_trial from autoresearch_rl
# ---------------------------------------------------------------------------
from pufferlib_market.autoresearch_rl import (
    TrialConfig,
    build_config,
    run_trial,
    STOCK_EXPERIMENTS,
    H100_STOCK_EXPERIMENTS,
    EXPERIMENTS as CRYPTO_EXPERIMENTS,
)

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Built-in job presets
# ---------------------------------------------------------------------------

def _stocks12_seedsweep(seeds: range = range(1, 51)) -> list[dict]:
    """50-seed sweep over the proven tp05_s123 config."""
    base = {
        "hidden_size": 1024,
        "lr": 3e-4,
        "anneal_lr": True,
        "ent_coef": 0.05,
        "trade_penalty": 0.05,
        "weight_decay": 0.01,
        "fill_slippage_bps": 5.0,
        "num_envs": 128,
        "use_bf16": True,
        "cuda_graph_ppo": True,
    }
    return [{**base, "seed": s, "description": f"tp05_seed_{s}"} for s in seeds]


def _stocks12_tp05_family() -> list[dict]:
    """Mutations around the tp05_s123 best config for stocks12."""
    base = {
        "hidden_size": 1024,
        "lr": 3e-4,
        "anneal_lr": True,
        "ent_coef": 0.05,
        "trade_penalty": 0.05,
        "weight_decay": 0.01,
        "fill_slippage_bps": 5.0,
        "seed": 123,
        "use_bf16": True,
        "cuda_graph_ppo": True,
    }
    mutations = [
        # lr variants
        {"lr": 1e-4, "description": "tp05_lr1e4_s123"},
        {"lr": 5e-4, "description": "tp05_lr5e4_s123"},
        # ent variants
        {"ent_coef": 0.03, "description": "tp05_ent003_s123"},
        {"ent_coef": 0.08, "description": "tp05_ent008_s123"},
        # weight decay variants
        {"weight_decay": 0.005, "description": "tp05_wd005_s123"},
        {"weight_decay": 0.05, "description": "tp05_wd05_s123"},
        # trade penalty variants
        {"trade_penalty": 0.03, "description": "tp03_s123"},
        {"trade_penalty": 0.08, "description": "tp08_s123"},
        {"trade_penalty": 0.10, "description": "tp10_s123"},
        # entropy annealing
        {"anneal_ent": True, "ent_coef": 0.08, "ent_coef_end": 0.02,
         "description": "tp05_ent_anneal_s123"},
        # obs norm
        {"obs_norm": True, "description": "tp05_obsnorm_s123"},
        # cosine LR
        {"lr_schedule": "cosine", "lr_warmup_frac": 0.02, "description": "tp05_coslr_s123"},
        # NorMuon optimizer
        {"optimizer": "muon", "description": "tp05_muon_s123"},
        {"optimizer": "muon", "description": "tp05_normuon_s123",
         "_normuon": True},  # picked up by _make_cmd
        # larger hidden
        {"hidden_size": 2048, "description": "tp05_h2048_s123"},
        # more envs
        {"num_envs": 256, "description": "tp05_envs256_s123"},
        # resmlp arch
        {"arch": "resmlp", "description": "tp05_resmlp_s123"},
        # attention arch (cross-symbol)
        {"arch": "attn", "description": "tp05_attn_s123"},
        # ReLU² activation
        {"activation": "relu2", "description": "tp05_relu2_s123"},
    ]
    return [{**base, **m} for m in mutations]


def _crypto_seedsweep(seeds: range = range(1, 21)) -> list[dict]:
    """Seed sweep over the proven crypto slip_5bps config."""
    base = {
        "hidden_size": 1024,
        "lr": 3e-4,
        "anneal_lr": True,
        "ent_coef": 0.05,
        "fill_slippage_bps": 5.0,
        "num_envs": 128,
        "use_bf16": True,
        "cuda_graph_ppo": True,
    }
    return [{**base, "seed": s, "description": f"crypto_slip5_seed_{s}"} for s in seeds]


PRESETS: dict[str, list[dict]] = {
    "stocks12_seedsweep": _stocks12_seedsweep(),
    "stocks12_tp05_family": _stocks12_tp05_family(),
    "crypto_seedsweep": _crypto_seedsweep(),
}


# ---------------------------------------------------------------------------
# Queue file helpers (JSONL, one job per line)
# ---------------------------------------------------------------------------

def _lock_file(fh) -> None:
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)


def _unlock_file(fh) -> None:
    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _read_queue(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        lines = fh.read().strip().splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def _write_queue(path: Path, jobs: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        for job in jobs:
            fh.write(json.dumps(job) + "\n")
    os.replace(tmp, path)


def add_jobs_to_queue(queue_path: Path, overrides_list: list[dict]) -> int:
    """Append jobs to the queue file.  Returns number of jobs added."""
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    # Open for reading+writing with exclusive lock
    queue_path.touch()
    with open(queue_path, "r+") as fh:
        _lock_file(fh)
        try:
            fh.seek(0)
            raw = fh.read().strip().splitlines()
            existing = [json.loads(ln) for ln in raw if ln.strip()]
            existing_ids = {j.get("id") for j in existing}
            new_jobs = []
            for ov in overrides_list:
                job = {
                    "id": str(uuid.uuid4())[:8],
                    "status": "pending",
                    "overrides": ov,
                }
                # Deduplicate by description
                desc = ov.get("description", "")
                if desc and any(j["overrides"].get("description") == desc for j in existing):
                    continue
                new_jobs.append(job)
            all_jobs = existing + new_jobs
            fh.seek(0)
            fh.truncate()
            for j in all_jobs:
                fh.write(json.dumps(j) + "\n")
        finally:
            _unlock_file(fh)
    return len(new_jobs)


def claim_next_job(queue_path: Path, worker_id: str) -> Optional[dict]:
    """Atomically claim the next pending job.  Returns job dict or None."""
    queue_path.touch()
    with open(queue_path, "r+") as fh:
        _lock_file(fh)
        try:
            fh.seek(0)
            raw = fh.read().strip().splitlines()
            jobs = [json.loads(ln) for ln in raw if ln.strip()]
            claimed = None
            for job in jobs:
                if job.get("status") == "pending":
                    job["status"] = "running"
                    job["worker_id"] = worker_id
                    job["claimed_at"] = time.time()
                    claimed = job
                    break
            if claimed is not None:
                fh.seek(0)
                fh.truncate()
                for j in jobs:
                    fh.write(json.dumps(j) + "\n")
        finally:
            _unlock_file(fh)
    return claimed


def mark_job_done(queue_path: Path, job_id: str, status: str = "done") -> None:
    """Update a job's status to done/failed in the queue file."""
    queue_path.touch()
    with open(queue_path, "r+") as fh:
        _lock_file(fh)
        try:
            fh.seek(0)
            raw = fh.read().strip().splitlines()
            jobs = [json.loads(ln) for ln in raw if ln.strip()]
            for job in jobs:
                if job.get("id") == job_id:
                    job["status"] = status
                    job["finished_at"] = time.time()
                    break
            fh.seek(0)
            fh.truncate()
            for j in jobs:
                fh.write(json.dumps(j) + "\n")
        finally:
            _unlock_file(fh)


# ---------------------------------------------------------------------------
# Leaderboard helpers
# ---------------------------------------------------------------------------

_LEADERBOARD_LOCK = multiprocessing.Lock()
_LEADERBOARD_FIELDNAMES = [
    "description", "val_return", "val_sortino", "val_wr",
    "robust_score", "gpu_id", "elapsed_s", "job_id", "timestamp",
]


def append_leaderboard_row(lb_path: Path, row: dict) -> None:
    """Append a result row to the shared leaderboard CSV (file-locked)."""
    lb_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not lb_path.exists()
    lb_path.touch()
    with open(lb_path, "a", newline="") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(fh, fieldnames=_LEADERBOARD_FIELDNAMES, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def detect_gpus() -> list[int]:
    """Return list of GPU indices available on this machine."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        return [int(x) for x in visible.split(",") if x.strip().isdigit()]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return [int(x.strip()) for x in result.stdout.strip().splitlines() if x.strip().isdigit()]
    except Exception:
        pass
    return [0]  # fallback: assume GPU 0


def gpu_free_memory_mb(gpu_id: int) -> int:
    """Return free VRAM in MB for the given GPU index."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# Worker loop: claims jobs and runs them on a specific GPU
# ---------------------------------------------------------------------------

def worker_loop(
    gpu_id: int,
    queue_path: Path,
    leaderboard_path: Path,
    train_data: str,
    val_data: str,
    checkpoint_dir: str,
    time_budget: int,
    holdout_data: Optional[str],
    holdout_n_windows: int,
    holdout_fill_buffer_bps: float,
    stocks_mode: bool,
) -> None:
    """Worker process: polls the queue and runs one trial at a time on gpu_id."""
    worker_id = f"gpu{gpu_id}_{os.getpid()}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    # Override env for child subprocesses via os.environ — run_trial uses sys.executable
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[worker gpu={gpu_id}] started, pid={os.getpid()}")

    ckpt_base = Path(checkpoint_dir) / f"gpu{gpu_id}"
    ckpt_base.mkdir(parents=True, exist_ok=True)

    idle_count = 0
    while True:
        job = claim_next_job(queue_path, worker_id)
        if job is None:
            idle_count += 1
            if idle_count >= 6:  # 30s with no work → exit
                print(f"[worker gpu={gpu_id}] queue empty, exiting")
                break
            time.sleep(5)
            continue
        idle_count = 0

        overrides = job["overrides"]
        desc = overrides.get("description", job["id"])
        job_ckpt_dir = str(ckpt_base / desc)

        print(f"\n[worker gpu={gpu_id}] running job {job['id']} — {desc}")
        t0 = time.time()

        config = build_config(overrides)
        # Apply NorMuon if requested via _normuon flag
        if overrides.get("_normuon"):
            config.optimizer = "muon"
            # The normuon flag is passed to train.py via --muon-norm-update flag (added below)
            # We store it in overrides for now — handled in _make_cmd

        try:
            result = run_trial(
                config=config,
                train_data=train_data,
                val_data=val_data,
                time_budget=time_budget,
                checkpoint_dir=job_ckpt_dir,
                holdout_data=holdout_data,
                holdout_n_windows=holdout_n_windows,
                holdout_fill_buffer_bps=holdout_fill_buffer_bps,
                best_trial_val_return=-float("inf"),
                best_trial_combined_score=-float("inf"),
                best_trial_rank_score=-float("inf"),
            )
            elapsed = time.time() - t0
            row = {
                "description": desc,
                "val_return": result.get("val_return"),
                "val_sortino": result.get("val_sortino"),
                "val_wr": result.get("val_wr"),
                "robust_score": result.get("rank_score"),
                "gpu_id": gpu_id,
                "elapsed_s": f"{elapsed:.0f}",
                "job_id": job["id"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            append_leaderboard_row(leaderboard_path, row)
            print(
                f"[worker gpu={gpu_id}] done {desc} in {elapsed:.0f}s — "
                f"val_ret={result.get('val_return')} sortino={result.get('val_sortino')}"
            )
            mark_job_done(queue_path, job["id"], "done")
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"[worker gpu={gpu_id}] ERROR in {desc}: {exc}")
            mark_job_done(queue_path, job["id"], "failed")


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def cmd_add(args: argparse.Namespace) -> None:
    queue_path = Path(args.queue)
    overrides_list: list[dict] = []

    if args.preset:
        for p in args.preset.split(","):
            p = p.strip()
            if p not in PRESETS:
                print(f"Unknown preset '{p}'. Available: {', '.join(PRESETS)}")
                sys.exit(1)
            overrides_list.extend(PRESETS[p])

    if args.jobs_file:
        with open(args.jobs_file) as fh:
            overrides_list.extend(json.load(fh))

    n = add_jobs_to_queue(queue_path, overrides_list)
    print(f"Added {n} jobs to {queue_path}")


def cmd_status(args: argparse.Namespace) -> None:
    queue_path = Path(args.queue)
    jobs = _read_queue(queue_path)
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j.get("status", "?")] = counts.get(j.get("status", "?"), 0) + 1
    print(f"Queue: {queue_path} ({len(jobs)} total)")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


def cmd_run(args: argparse.Namespace) -> None:
    queue_path = Path(args.queue)

    # Build and enqueue jobs
    overrides_list: list[dict] = []
    if args.preset:
        for p in args.preset.split(","):
            p = p.strip()
            if p not in PRESETS:
                print(f"Unknown preset '{p}'. Available: {', '.join(PRESETS)}")
                sys.exit(1)
            overrides_list.extend(PRESETS[p])
    if args.jobs_file:
        with open(args.jobs_file) as fh:
            overrides_list.extend(json.load(fh))

    if overrides_list:
        n = add_jobs_to_queue(queue_path, overrides_list)
        print(f"Added {n} new jobs to {queue_path}")

    # Determine GPUs to use
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = detect_gpus()
    print(f"Using GPUs: {gpu_ids}")

    leaderboard_path = Path(args.leaderboard)

    # Spawn one worker process per GPU
    worker_kwargs = dict(
        queue_path=queue_path,
        leaderboard_path=leaderboard_path,
        train_data=args.train_data,
        val_data=args.val_data,
        checkpoint_dir=args.checkpoint_dir,
        time_budget=args.time_budget,
        holdout_data=args.holdout_data or None,
        holdout_n_windows=args.holdout_n_windows,
        holdout_fill_buffer_bps=args.holdout_fill_buffer_bps,
        stocks_mode=args.stocks,
    )

    if len(gpu_ids) == 1:
        # Serial mode — run directly in this process (saves overhead)
        worker_loop(gpu_id=gpu_ids[0], **worker_kwargs)
    else:
        procs = []
        for gid in gpu_ids:
            p = multiprocessing.Process(target=worker_loop, kwargs={"gpu_id": gid, **worker_kwargs})
            p.start()
            procs.append(p)
        try:
            for p in procs:
                p.join()
        except KeyboardInterrupt:
            print("\nInterrupt — stopping workers")
            for p in procs:
                p.terminate()
            for p in procs:
                p.join()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pufferlib_market.gpu_pool",
        description="GPU job pool for RL training experiments",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── add ──
    p_add = sub.add_parser("add", help="Add jobs to queue")
    p_add.add_argument("--queue", required=True, help="Path to JSONL queue file")
    p_add.add_argument("--preset", default="", help=f"Comma-separated preset names: {', '.join(PRESETS)}")
    p_add.add_argument("--jobs-file", default="", help="JSON file with list of override dicts")

    # ── status ──
    p_st = sub.add_parser("status", help="Show queue status")
    p_st.add_argument("--queue", required=True)

    # ── run ──
    p_run = sub.add_parser("run", help="Start workers and process the queue")
    p_run.add_argument("--queue", required=True, help="Path to JSONL queue file")
    p_run.add_argument("--train-data", required=True)
    p_run.add_argument("--val-data", required=True)
    p_run.add_argument("--leaderboard", required=True, help="Output CSV leaderboard path")
    p_run.add_argument("--checkpoint-dir", default="pufferlib_market/checkpoints/pool")
    p_run.add_argument("--preset", default="", help=f"Comma-separated preset names: {', '.join(PRESETS)}")
    p_run.add_argument("--jobs-file", default="", help="JSON file with list of override dicts")
    p_run.add_argument("--gpu-ids", default="", help="Comma-separated GPU IDs (default: all detected)")
    p_run.add_argument("--time-budget", type=int, default=300, help="Seconds per trial (default 300)")
    p_run.add_argument("--holdout-data", default="", help="Holdout data for multi-window eval")
    p_run.add_argument("--holdout-n-windows", type=int, default=50)
    p_run.add_argument("--holdout-fill-buffer-bps", type=float, default=5.0)
    p_run.add_argument("--stocks", action="store_true", help="Use stocks mode (disables shorts)")

    args = parser.parse_args()

    if args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
