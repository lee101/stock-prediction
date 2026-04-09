"""
GPU job pool for RL training experiments.

Runs one training trial per available GPU (serial on 1 GPU, parallel on N).
Jobs are stored in a JSONL queue file; workers claim jobs atomically via
fcntl.flock and write results to a shared leaderboard CSV.

Usage
-----
# 1) Generate a queue of jobs and launch workers:
python -m pufferlib_market.gpu_pool run \\
    --queue tmp/pool_queue.jsonl \\
    --train-data pufferlib_market/data/stocks12_daily_train.bin \\
    --val-data   pufferlib_market/data/stocks12_daily_val.bin \\
    --leaderboard sweepresults/pool_leaderboard.csv \\
    --checkpoint-dir pufferlib_market/checkpoints/pool \\
    --preset stocks12_seedsweep \\
    --time-budget 300 \\
    [--gpu-ids 0,1,2]   # default: all detected GPUs

# 2) Add more jobs to a running pool:
python -m pufferlib_market.gpu_pool add \\
    --queue tmp/pool_queue.jsonl \\
    --preset stocks12_tp05_family

# 3) Show queue status:
python -m pufferlib_market.gpu_pool status --queue tmp/pool_queue.jsonl
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
    """50-seed sweep over the EXACT tp05_s123 config (v2cfg).

    CRITICAL: s123 config is slip=0, wd=0, bf16=True, cuda_graph=True, periods=252.
    Adding slip=5 or wd=0.01 models to the ensemble HURTS (47.30%→28.15%),
    because slip-trained models trade differently (wider edges, incompatible timing).
    DO NOT add fill_slippage_bps or weight_decay here.
    """
    base = {
        "hidden_size": 1024,
        "lr": 3e-4,
        "anneal_lr": True,
        "ent_coef": 0.05,
        "trade_penalty": 0.05,
        "weight_decay": 0.0,       # MUST be 0 — matches s123
        "fill_slippage_bps": 0.0,  # MUST be 0 — matches s123
        "num_envs": 128,
        "use_bf16": True,
        "cuda_graph_ppo": True,
        "no_cuda_graph": False,
        "periods_per_year": 252.0,
        "max_steps": 720,
    }
    return [{**base, "seed": s, "description": f"v2cfg_s{s}"} for s in seeds]


def _stocks12_tp05_family() -> list[dict]:
    """Mutations around the EXACT tp05_s123 best config for stocks12 (v2cfg).

    Base uses s123 config: slip=0, wd=0, bf16=True, cuda_graph=True, periods=252.
    """
    base = {
        "hidden_size": 1024,
        "lr": 3e-4,
        "anneal_lr": True,
        "ent_coef": 0.05,
        "trade_penalty": 0.05,
        "weight_decay": 0.0,       # MUST be 0 — matches s123
        "fill_slippage_bps": 0.0,  # MUST be 0 — matches s123
        "seed": 123,
        "use_bf16": True,
        "cuda_graph_ppo": True,
        "no_cuda_graph": False,
        "periods_per_year": 252.0,
        "max_steps": 720,
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


def _crypto70_autoresearch(seeds: list[int] = [42, 123, 7]) -> list[dict]:
    """Autoresearch configs for crypto70 daily dataset — sweeps trade_penalty, LR, slippage.

    Uses daily-appropriate settings: periods_per_year=365, max_steps=180 (6mo episodes).
    """
    base = {
        "hidden_size": 1024,
        "anneal_lr": True,
        "ent_coef": 0.05,
        "num_envs": 128,
        "use_bf16": True,
        "cuda_graph_ppo": False,
        "no_cuda_graph": True,  # more robust when GPU is shared with other processes
        # Daily crypto settings
        "periods_per_year": 365.0,
        "max_steps": 180,
    }
    configs = []
    for seed in seeds:
        for tp in [0.03, 0.05, 0.08]:
            for slip in [0.0, 5.0]:
                for lr in [3e-4]:
                    desc = f"c70_tp{int(tp*100):02d}_slip{int(slip)}_lr{lr:.0e}_s{seed}"
                    configs.append({**base,
                        "seed": seed, "lr": lr,
                        "trade_penalty": tp, "fill_slippage_bps": slip,
                        "description": desc,
                    })
        # muon optimizer variant at best config
        configs.append({**base, "seed": seed, "lr": 0.02,
            "trade_penalty": 0.05, "fill_slippage_bps": 5.0,
            "optimizer": "muon",
            "description": f"c70_tp05_slip5_muon_s{seed}",
        })
    return configs


def _crypto15_robust_champion(seeds: list[int] = [42, 123, 7, 1, 2, 3, 5, 10]) -> list[dict]:
    """Multi-config seed sweep on crypto15 daily data.

    crypto15 has 15 live Binance symbols (BTC/ETH/SOL/LTC/AVAX/DOGE/LINK/ADA+more)
    and 1375 bars (~3.8 years). Matches live Binance deployment symbols.

    Uses the simple crypto70 winning config (no obs_norm) — obs_norm breaks evaluate.py
    because it adds obs_mean/obs_std buffers the evaluator doesn't handle.
    Crypto70 results: tp05+slip5+s123=4.96x, tp03+slip5+s123=3.92x, tp08+slip5+s42=2.99x.
    """
    daily_base = {
        "hidden_size": 1024,
        "lr": 3e-4,
        "fill_slippage_bps": 5.0,
        "anneal_lr": True,
        "ent_coef": 0.05,
        "num_envs": 128,
        "use_bf16": True,
        "no_cuda_graph": True,
        "periods_per_year": 365.0,
        "max_steps": 180,
    }
    configs = []
    for tp in [0.05, 0.03, 0.08]:
        for s in seeds:
            desc = f"c15_tp{int(tp*100):02d}_slip5_s{s}"
            configs.append({**daily_base, "trade_penalty": tp, "seed": s,
                            "description": desc})
    return configs


def _stocks12_seedsweep_ext(seeds: range = range(37, 51)) -> list[dict]:
    """Continue the tp05 seed sweep for stocks12, seeds 37-50."""
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
        "no_cuda_graph": True,
    }
    return [{**base, "seed": s, "description": f"tp05_seed_{s}"} for s in seeds]


def _crypto70_daily_long_sweep(seeds: range | list[int] = range(1, 101)) -> list[dict]:
    """Seed sweep of proven tp05_slip5 config on crypto70 daily, 30-min per seed on H100.

    crypto70 = 48 Binance USDT pairs, daily bars, train 2019-2025, val 2025-09-01 to 2026-03-31.
    Best OOS result (5-min runs): s19=+439%/180d ann, 28/60 seeds genuine.
    Longer training (30 min on H100 ≈ 100-200M steps) should improve further.
    """
    base = {
        "hidden_size": 1024,
        "lr": 3e-4,
        "anneal_lr": True,
        "ent_coef": 0.05,
        "trade_penalty": 0.05,
        "fill_slippage_bps": 5.0,
        "num_envs": 128,
        "use_bf16": True,
        "no_cuda_graph": True,
        "periods_per_year": 365.0,
        "max_steps": 180,
        "time_budget_override": 1800,  # 30 min per seed
    }
    return [{**base, "seed": s, "description": f"c70_long_tp05_s{s}"} for s in seeds]


def _crypto40_hourly_tp05_sweep(seeds: range | list[int] = range(1, 61)) -> list[dict]:
    """Seed sweep of tp05_slip5 on crypto40 hourly dataset.

    crypto40 = 25 Binance USDT pairs, hourly bars.
    train: 18448 steps (~2.1 years), val: 4901 steps (~204 days).
    Val gives 6+ non-overlapping 30-day windows → better OOS eval than daily.
    Max steps = 720 hours = 30 days per episode.
    """
    base = {
        "hidden_size": 1024,
        "lr": 3e-4,
        "anneal_lr": True,
        "ent_coef": 0.05,
        "trade_penalty": 0.05,
        "fill_slippage_bps": 5.0,
        "num_envs": 128,
        "use_bf16": True,
        "no_cuda_graph": True,
        "periods_per_year": 8760.0,
        "max_steps": 720,
        "time_budget_override": 1800,  # 30 min per seed
    }
    return [{**base, "seed": s, "description": f"c40h_tp05_s{s}"} for s in seeds]


def _stocks15_tp05_sweep(seeds: list[int] = [123, 15, 36, 42, 7, 1, 2, 3, 5, 10]) -> list[dict]:
    """tp05 seed sweep for stocks15 (15 symbols with 2012 data, 4840 bars).

    stocks15 = AAPL, MSFT, NVDA, GOOG, META, TSLA, SPY, QQQ, JPM, V, AMZN, COST, WMT, HD, BRK-B
    Starting with seeds that worked well for stocks12 (123, 15, 36) plus broader search.
    """
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
        "no_cuda_graph": True,
        "periods_per_year": 252.0,
        "max_steps": 252,
    }
    return [{**base, "seed": s, "description": f"s15_tp05_s{s}"} for s in seeds]


def _stocks12_fidelity_longrun() -> list[dict]:
    """Longer, higher-fidelity stocks12 runs around the current best seeds."""
    base = {
        "hidden_size": 1024,
        "lr": 3e-4,
        "anneal_lr": True,
        "ent_coef": 0.05,
        "weight_decay": 0.0,
        "fill_slippage_bps": 0.0,
        "num_envs": 128,
        "use_bf16": False,
        "no_tf32": True,
        "no_cuda_graph": True,
        "periods_per_year": 252.0,
        "max_steps": 252,
        "time_budget_override": 1800,
    }
    candidates = [
        {"seed": 123, "trade_penalty": 0.05, "description": "stocks12_fidelity_tp05_s123"},
        {"seed": 15, "trade_penalty": 0.05, "description": "stocks12_fidelity_tp05_s15"},
        {"seed": 36, "trade_penalty": 0.05, "description": "stocks12_fidelity_tp05_s36"},
        {"seed": 123, "trade_penalty": 0.10, "description": "stocks12_fidelity_tp10_s123"},
        {"seed": 15, "trade_penalty": 0.10, "description": "stocks12_fidelity_tp10_s15"},
        {"seed": 36, "trade_penalty": 0.10, "description": "stocks12_fidelity_tp10_s36"},
    ]
    return [{**base, **cfg} for cfg in candidates]


PRESETS: dict[str, list[dict]] = {
    "stocks12_seedsweep": _stocks12_seedsweep(),
    "stocks12_seedsweep_ext": _stocks12_seedsweep_ext(),
    "stocks12_tp05_family": _stocks12_tp05_family(),
    "stocks12_fidelity_longrun": _stocks12_fidelity_longrun(),
    "stocks15_tp05_sweep": _stocks15_tp05_sweep(),
    "crypto_seedsweep": _crypto_seedsweep(),
    "crypto70_autoresearch": _crypto70_autoresearch(),
    "crypto15_robust_champion": _crypto15_robust_champion(),
    "crypto70_daily_long_sweep": _crypto70_daily_long_sweep(),
    "crypto40_hourly_tp05_sweep": _crypto40_hourly_tp05_sweep(),
}


BASELINE_PROFILES: dict[str, dict[str, float | str]] = {
    "none": {
        "baseline_val_return_floor": -float("inf"),
        "baseline_combined_floor": -float("inf"),
        "projection_clip_abs": 6.0,
        "description": "Disable hard baseline pruning.",
    },
    "stocks_daily_candidate": {
        "baseline_val_return_floor": 0.35,
        "baseline_combined_floor": 1.0,
        "projection_clip_abs": 6.0,
        "description": (
            "Conservative daily-stock quick-eval floor derived from the canonical "
            "stocks leaderboard, intentionally below the 0.48-0.57 val_return and "
            "1.39-1.44 combined scores of the established winners."
        ),
    },
}


def resolve_baseline_prune_settings(
    *,
    baseline_profile: str,
    stocks_mode: bool,
    baseline_val_return_floor: float | None,
    baseline_combined_floor: float | None,
    projection_clip_abs: float | None,
) -> dict[str, float | str]:
    """Resolve pool-level hard-prune settings, with stock defaults in auto mode."""
    selected_profile = baseline_profile or "auto"
    if selected_profile == "auto":
        selected_profile = "stocks_daily_candidate" if stocks_mode else "none"
    if selected_profile not in BASELINE_PROFILES:
        raise ValueError(
            f"Unknown baseline profile '{selected_profile}'. "
            f"Available: auto, {', '.join(sorted(BASELINE_PROFILES))}"
        )

    settings = dict(BASELINE_PROFILES[selected_profile])
    if baseline_val_return_floor is not None:
        settings["baseline_val_return_floor"] = float(baseline_val_return_floor)
    if baseline_combined_floor is not None:
        settings["baseline_combined_floor"] = float(baseline_combined_floor)
    if projection_clip_abs is not None:
        settings["projection_clip_abs"] = float(projection_clip_abs)
    settings["profile_name"] = selected_profile
    return settings


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
    holdout_eval_steps: int,
    holdout_fill_buffer_bps: float,
    stocks_mode: bool,
    baseline_profile: str,
    baseline_val_return_floor: float | None,
    baseline_combined_floor: float | None,
    projection_clip_abs: float | None,
) -> None:
    """Worker process: polls the queue and runs one trial at a time on gpu_id."""
    worker_id = f"gpu{gpu_id}_{os.getpid()}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    # Override env for child subprocesses via os.environ — run_trial uses sys.executable
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[worker gpu={gpu_id}] started, pid={os.getpid()}")
    baseline_settings = resolve_baseline_prune_settings(
        baseline_profile=baseline_profile,
        stocks_mode=stocks_mode,
        baseline_val_return_floor=baseline_val_return_floor,
        baseline_combined_floor=baseline_combined_floor,
        projection_clip_abs=projection_clip_abs,
    )
    print(
        f"[worker gpu={gpu_id}] prune baseline={baseline_settings['profile_name']} "
        f"val_floor={baseline_settings['baseline_val_return_floor']} "
        f"combined_floor={baseline_settings['baseline_combined_floor']} "
        f"clip_abs={baseline_settings['projection_clip_abs']}"
    )

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
                holdout_eval_steps=holdout_eval_steps,
                holdout_fill_buffer_bps=holdout_fill_buffer_bps,
                best_trial_val_return=-float("inf"),
                best_trial_combined_score=-float("inf"),
                best_trial_rank_score=-float("inf"),
                baseline_val_return_floor=float(baseline_settings["baseline_val_return_floor"]),
                baseline_combined_floor=float(baseline_settings["baseline_combined_floor"]),
                projection_clip_abs=float(baseline_settings["projection_clip_abs"]),
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
        holdout_eval_steps=args.holdout_eval_steps,
        holdout_fill_buffer_bps=args.holdout_fill_buffer_bps,
        stocks_mode=args.stocks,
        baseline_profile=args.baseline_profile,
        baseline_val_return_floor=args.baseline_val_return_floor,
        baseline_combined_floor=args.baseline_combined_floor,
        projection_clip_abs=args.projection_clip_abs,
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
    p_run.add_argument("--holdout-eval-steps", type=int, default=0,
                       help="Window length in steps for holdout eval (0=disabled, e.g. 90 for daily stocks)")
    p_run.add_argument("--holdout-fill-buffer-bps", type=float, default=5.0)
    p_run.add_argument("--stocks", action="store_true", help="Use stocks mode (disables shorts)")
    p_run.add_argument(
        "--baseline-profile",
        choices=["auto", *sorted(BASELINE_PROFILES)],
        default="auto",
        help="Hard early-prune floor profile. 'auto' enables the stock daily floor only in --stocks mode.",
    )
    p_run.add_argument(
        "--baseline-val-return-floor",
        type=float,
        default=None,
        help="Override the hard val_return floor used for early pruning.",
    )
    p_run.add_argument(
        "--baseline-combined-floor",
        type=float,
        default=None,
        help="Override the hard combined-score floor used for early pruning.",
    )
    p_run.add_argument(
        "--projection-clip-abs",
        type=float,
        default=None,
        help="Clip absolute combined scores before line/polynomial projection.",
    )

    args = parser.parse_args()

    if args.cmd == "add":
        cmd_add(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
