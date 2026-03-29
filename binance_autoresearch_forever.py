#!/usr/bin/env python3
"""Binance crypto autoresearch forever loop.

Spins up RunPod 4090s (or runs locally), trains models on both daily and hourly
Binance data, evaluates against production baseline, saves top-K checkpoints
and logs to R2, and writes results to numbered progress files.

Usage:
    # Local GPU (auto-detect):
    python scripts/binance_autoresearch_forever.py

    # RunPod 4090:
    python scripts/binance_autoresearch_forever.py --gpu-type 4090

    # Dry run:
    python scripts/binance_autoresearch_forever.py --dry-run

    # Single round (no forever loop):
    python scripts/binance_autoresearch_forever.py --once
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.runpod_client import (
    DEFAULT_GPU_FALLBACKS,
    GPU_ALIASES,
    HOURLY_RATES,
    PodConfig,
    RunPodClient,
    build_gpu_fallback_types,
    is_capacity_error,
    parse_gpu_fallback_types,
    resolve_gpu_type,
)
from src.checkpoint_manager import TopKCheckpointManager

log = logging.getLogger("binance_forever")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Data tracks
# ---------------------------------------------------------------------------

@dataclass
class DataTrack:
    name: str
    train: str
    val: str
    periods_per_year: float
    max_steps: int
    fee_rate: float = 0.001
    description: str = ""
    focused_descriptions: str = ""  # comma-separated descriptions for focused sweeps

CRYPTO_DAILY = DataTrack(
    name="crypto_daily",
    train="pufferlib_market/data/crypto29_daily_train.bin",
    val="pufferlib_market/data/crypto29_daily_val.bin",
    periods_per_year=365.0,
    max_steps=90,
    description="29-symbol Binance daily bars",
)

CRYPTO_HOURLY = DataTrack(
    name="crypto_hourly",
    train="pufferlib_market/data/crypto34_hourly_train.bin",
    val="pufferlib_market/data/crypto34_hourly_val.bin",
    periods_per_year=8760.0,
    max_steps=720,
    description="34-symbol Binance hourly bars",
    focused_descriptions=",".join(
        f"c34h_{v}_s{s}"
        for v in [
            "tp01_slip5_wd01", "tp01_slip5_wd05",
            "tp03_slip5_wd01", "tp03_slip5_wd05",
            "tp05_slip8_wd01", "tp05_slip8_wd05",
        ]
        for s in [7, 19, 33, 42, 80, 99]
    ),
)

MIXED_DAILY = DataTrack(
    name="mixed_daily",
    train="pufferlib_market/data/mixed40_daily_train.bin",
    val="pufferlib_market/data/mixed40_daily_val.bin",
    periods_per_year=365.0,
    max_steps=90,
    description="40-symbol mixed daily (crypto+stocks)",
)

MIXED_HOURLY = DataTrack(
    name="mixed_hourly",
    train="pufferlib_market/data/mixed23_latest_train_20260320.bin",
    val="pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin",
    periods_per_year=8760.0,
    max_steps=720,
    description="23-symbol mixed hourly",
)

ALL_TRACKS = [CRYPTO_DAILY, CRYPTO_HOURLY, MIXED_DAILY, MIXED_HOURLY]

# ---------------------------------------------------------------------------
# Production baseline
# ---------------------------------------------------------------------------

@dataclass
class ProductionBaseline:
    val_return: float = 1.914       # +191.4% robust_reg_tp005_ent C sim binary fills 5bps
    val_sortino: float = 23.94
    win_rate: float = 0.59
    description: str = "robust_reg_tp005_ent (pufferlib, h1024, 23-sym mixed hourly)"
    checkpoint: str = "pufferlib_market/checkpoints/mixed23_a40_sweep/robust_reg_tp005_ent/best.pt"

    def combined_score(self) -> float:
        return 0.5 * self.val_return + 0.5 * self.val_sortino

    def to_dict(self) -> dict:
        return {
            "val_return": self.val_return,
            "val_sortino": self.val_sortino,
            "win_rate": self.win_rate,
            "combined_score": self.combined_score(),
            "description": self.description,
            "checkpoint": self.checkpoint,
        }

# ---------------------------------------------------------------------------
# Trial result
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    description: str
    track: str
    val_return: Optional[float] = None
    val_sortino: Optional[float] = None
    win_rate: Optional[float] = None
    holdout_robust_score: Optional[float] = None
    best_checkpoint: Optional[str] = None
    training_time_s: float = 0.0
    gpu_type: str = ""
    config: dict = field(default_factory=dict)
    early_stopped: bool = False
    error: Optional[str] = None

    def combined_score(self) -> float:
        r = self.val_return or 0.0
        s = self.val_sortino or 0.0
        return 0.5 * r + 0.5 * s

    def beats_baseline(self, baseline: ProductionBaseline) -> bool:
        if self.val_return is None or self.val_sortino is None:
            return False
        return self.combined_score() > baseline.combined_score()

# ---------------------------------------------------------------------------
# Progress file management
# ---------------------------------------------------------------------------

def _find_next_progress_number() -> int:
    existing = list(REPO.glob("binanceprogress*.md"))
    max_num = 0
    for p in existing:
        name = p.stem
        if name == "binanceprogress_failed":
            continue
        digits = "".join(c for c in name.replace("binanceprogress", "") if c.isdigit())
        if digits:
            max_num = max(max_num, int(digits))
    return max_num + 1


def write_success_progress(result: TrialResult, baseline: ProductionBaseline, round_num: int) -> Path:
    num = _find_next_progress_number()
    path = REPO / f"binanceprogress{num}.md"
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    content = f"""# Binance Autoresearch Success #{num} ({now})

## Round {round_num} -- {result.description}

**Track**: {result.track}
**GPU**: {result.gpu_type}
**Training time**: {result.training_time_s:.0f}s
**Early stopped**: {result.early_stopped}

## Results

| Metric | Trial | Production Baseline |
|--------|-------|-------------------|
| Return | {_fmt(result.val_return)} | {_fmt(baseline.val_return)} |
| Sortino | {_fmt(result.val_sortino)} | {_fmt(baseline.val_sortino)} |
| Win Rate | {_fmt(result.win_rate)} | {_fmt(baseline.win_rate)} |
| Combined | {result.combined_score():.4f} | {baseline.combined_score():.4f} |
| Holdout Robust | {_fmt(result.holdout_robust_score)} | -- |

**BEATS PRODUCTION**: YES

## Checkpoint
`{result.best_checkpoint or 'N/A'}`

## Config
```json
{json.dumps(result.config, indent=2)}
```
"""
    path.write_text(content)
    log.info("wrote success progress: %s", path)
    return path


def append_failure_progress(result: TrialResult, baseline: ProductionBaseline, round_num: int) -> Path:
    path = REPO / "binanceprogress_failed.md"
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"""
---

## Round {round_num} -- {result.description} ({now})

**Track**: {result.track} | **GPU**: {result.gpu_type} | **Time**: {result.training_time_s:.0f}s | **Early stopped**: {result.early_stopped}

| Metric | Trial | Baseline |
|--------|-------|----------|
| Return | {_fmt(result.val_return)} | {_fmt(baseline.val_return)} |
| Sortino | {_fmt(result.val_sortino)} | {_fmt(baseline.val_sortino)} |
| Combined | {result.combined_score():.4f} | {baseline.combined_score():.4f} |

"""
    if result.error:
        entry += f"**Error**: {result.error}\n\n"
    if result.config:
        entry += f"Config: `{json.dumps(result.config)}`\n"

    if not path.exists():
        header = "# Binance Autoresearch -- Failed Trials\n\nTrials that did not beat the production baseline.\n"
        path.write_text(header + entry)
    else:
        with open(path, "a") as f:
            f.write(entry)
    return path


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:+.4f}"

# ---------------------------------------------------------------------------
# R2 sync
# ---------------------------------------------------------------------------

def sync_to_r2(checkpoint_dir: Path, log_file: Path, run_id: str) -> None:
    try:
        from src.r2_client import R2Client
        client = R2Client()
        prefix = f"binance_autoresearch/{run_id}"

        if checkpoint_dir.exists():
            keys = client.sync_dir_to_r2(str(checkpoint_dir), f"{prefix}/checkpoints", skip_existing=True)
            log.info("R2: uploaded %d checkpoint files", len(keys))

        if log_file.exists():
            client.upload_file(str(log_file), f"{prefix}/trial_log.jsonl")
            log.info("R2: uploaded trial log")

    except Exception as e:
        log.warning("R2 sync failed: %s", e)

# ---------------------------------------------------------------------------
# SSH helpers for RunPod
# ---------------------------------------------------------------------------

_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]


def _ssh_run(host: str, port: int, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    full = ["ssh", *_SSH_OPTS, "-p", str(port), f"root@{host}", cmd]
    return subprocess.run(full, check=check, capture_output=True, text=True)


def _rsync_to_pod(host: str, port: int, local_dir: Path, remote_dir: str) -> None:
    cmd = [
        "rsync", "-az", "--delete",
        "--exclude", "__pycache__/", "--exclude", ".git/",
        "--exclude", "pufferlib_market/data/", "--exclude", "pufferlib_market/checkpoints/",
        "--exclude", ".venv*/", "--exclude", "*.pyc",
        "-e", f"ssh {' '.join(_SSH_OPTS)} -p {port}",
        f"{local_dir}/", f"root@{host}:{remote_dir}/",
    ]
    subprocess.run(cmd, check=True)


def _rsync_data(host: str, port: int, local_path: Path, remote_dir: str) -> None:
    rel = local_path.relative_to(REPO) if local_path.is_relative_to(REPO) else Path(local_path.name)
    remote_parent = str(Path(remote_dir) / rel.parent)
    _ssh_run(host, port, f"mkdir -p {remote_parent}")
    cmd = [
        "rsync", "-az",
        "-e", f"ssh {' '.join(_SSH_OPTS)} -p {port}",
        str(local_path), f"root@{host}:{remote_dir}/{rel}",
    ]
    subprocess.run(cmd, check=True)


def _rsync_from_pod(host: str, port: int, remote_path: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-az",
        "-e", f"ssh {' '.join(_SSH_OPTS)} -p {port}",
        f"root@{host}:{remote_path}/", f"{local_dir}/",
    ]
    subprocess.run(cmd, check=False)


def _bootstrap_pod(host: str, port: int, remote_dir: str) -> None:
    bootstrap = (
        f"set -euo pipefail && cd {remote_dir} && "
        f"pip install uv -q && "
        f"uv venv .venv313 --python python3.13 2>/dev/null || uv venv .venv313 && "
        f"source .venv313/bin/activate && "
        f"uv pip install -e . -q && "
        f"{{ [ -d PufferLib ] && uv pip install -e PufferLib/ -q || true; }} && "
        f"cd pufferlib_market && python setup.py build_ext --inplace -q && cd .."
    )
    _ssh_run(host, port, bootstrap)

# ---------------------------------------------------------------------------
# Run a single autoresearch batch on a pod or locally
# ---------------------------------------------------------------------------

def run_autoresearch_batch(
    track: DataTrack,
    *,
    time_budget: int = 300,
    max_trials: int = 15,
    run_id: str = "",
    gpu_type: str = "",
    ssh_host: str = "",
    ssh_port: int = 0,
    remote_dir: str = "/workspace/stock-prediction",
    local: bool = False,
    descriptions: str = "",
) -> tuple[Path, Path]:
    """Run autoresearch_rl and return (leaderboard_path, checkpoint_dir)."""
    if not run_id:
        run_id = f"binance_forever_{track.name}_{time.strftime('%Y%m%d_%H%M%S')}"

    leaderboard = REPO / "analysis" / f"{run_id}_leaderboard.csv"
    checkpoint_dir = REPO / "pufferlib_market" / "checkpoints" / run_id

    cmd_parts = [
        sys.executable, "-u", "-m", "pufferlib_market.autoresearch_rl",
        "--train-data", track.train,
        "--val-data", track.val,
        "--time-budget", str(time_budget),
        "--max-trials", str(max_trials),
        "--periods-per-year", str(track.periods_per_year),
        "--max-steps-override", str(track.max_steps),
        "--fee-rate-override", str(track.fee_rate),
        "--leaderboard", str(leaderboard),
        "--checkpoint-root", str(checkpoint_dir),
        "--holdout-n-windows", "20",
        "--holdout-fill-buffer-bps", "5",
        "--poly-prune",
    ]
    if descriptions:
        cmd_parts += ["--descriptions", descriptions]

    if local:
        log.info("running locally: %s", run_id)
        leaderboard.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd_parts, cwd=str(REPO))
    else:
        remote_lb = f"pufferlib_market/{run_id}_leaderboard.csv"
        remote_ckpt = f"pufferlib_market/checkpoints/{run_id}"
        remote_cmd_parts = [
            "python", "-u", "-m", "pufferlib_market.autoresearch_rl",
            "--train-data", track.train,
            "--val-data", track.val,
            "--time-budget", str(time_budget),
            "--max-trials", str(max_trials),
            "--periods-per-year", str(track.periods_per_year),
            "--max-steps-override", str(track.max_steps),
            "--fee-rate-override", str(track.fee_rate),
            "--leaderboard", remote_lb,
            "--checkpoint-root", remote_ckpt,
            "--holdout-n-windows", "20",
            "--holdout-fill-buffer-bps", "5",
            "--poly-prune",
        ]
        if descriptions:
            remote_cmd_parts += ["--descriptions", descriptions]

        wandb_key = os.environ.get("WANDB_API_KEY", "")
        wandb_export = f"export WANDB_API_KEY={wandb_key} && " if wandb_key else ""
        shell_cmd = (
            f"cd {remote_dir} && source .venv313/bin/activate && "
            f"export PYTHONPATH={remote_dir}:${{PYTHONPATH:-}} && "
            f"{wandb_export}"
            f"{' '.join(remote_cmd_parts)}"
        )
        log.info("running on pod %s:%d: %s", ssh_host, ssh_port, run_id)
        _ssh_run(ssh_host, ssh_port, shell_cmd, check=False)

        # download results
        leaderboard.parent.mkdir(parents=True, exist_ok=True)
        _scp_from_pod(ssh_host, ssh_port, f"{remote_dir}/{remote_lb}", leaderboard)
        _rsync_from_pod(ssh_host, ssh_port, f"{remote_dir}/{remote_ckpt}", checkpoint_dir)

    return leaderboard, checkpoint_dir


def _scp_from_pod(host: str, port: int, remote_path: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", *_SSH_OPTS, "-P", str(port), f"root@{host}:{remote_path}", str(local_path)]
    return subprocess.run(cmd, check=False).returncode == 0

# ---------------------------------------------------------------------------
# Read leaderboard and find best result
# ---------------------------------------------------------------------------

def read_best_from_leaderboard(leaderboard: Path, checkpoint_dir: Path) -> TrialResult:
    result = TrialResult(description="none", track="")
    if not leaderboard.exists():
        result.error = "leaderboard not found"
        return result

    try:
        with open(leaderboard) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        result.error = str(e)
        return result

    if not rows:
        result.error = "empty leaderboard"
        return result

    def score(r: dict) -> float:
        vr = _safe_float(r.get("val_return"))
        vs = _safe_float(r.get("val_sortino"))
        if vr is None and vs is None:
            return -float("inf")
        return 0.5 * (vr or 0.0) + 0.5 * (vs or 0.0)

    rows.sort(key=score, reverse=True)
    best = rows[0]

    result.description = best.get("description", best.get("model", "unknown"))
    result.val_return = _safe_float(best.get("val_return"))
    result.val_sortino = _safe_float(best.get("val_sortino"))
    result.win_rate = _safe_float(best.get("val_wr"))
    result.holdout_robust_score = _safe_float(best.get("holdout_robust_score"))

    # find checkpoint
    ckpt_path = best.get("checkpoint_path", "")
    if ckpt_path and Path(ckpt_path).exists():
        result.best_checkpoint = ckpt_path
    else:
        desc = result.description
        candidates = list(checkpoint_dir.glob(f"*{desc}*/best.pt"))
        if candidates:
            result.best_checkpoint = str(candidates[0])
        else:
            pts = sorted(checkpoint_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pts:
                result.best_checkpoint = str(pts[0])

    return result


def _safe_float(v) -> Optional[float]:
    if v in (None, "", "None", "nan"):
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None

# ---------------------------------------------------------------------------
# Pod lifecycle
# ---------------------------------------------------------------------------

class PodManager:
    def __init__(self, gpu_type: str = "4090", gpu_fallback_types: list[str] | None = None):
        self.gpu_type = gpu_type
        self.gpu_fallback_types = None if gpu_fallback_types is None else list(gpu_fallback_types)
        self.client: Optional[RunPodClient] = None
        self.pod_id: Optional[str] = None
        self.ssh_host: str = ""
        self.ssh_port: int = 0
        self._bootstrapped: bool = False
        self.active_gpu_type: str = resolve_gpu_type(gpu_type)

    def ensure_ready(self) -> tuple[str, int]:
        if self.ssh_host and self.ssh_port:
            try:
                _ssh_run(self.ssh_host, self.ssh_port, "echo ok")
                return self.ssh_host, self.ssh_port
            except Exception:
                log.warning("pod connection lost, reprovisioning")
                self._bootstrapped = False

        if self.client is None:
            self.client = RunPodClient()

        if self.pod_id:
            try:
                self.client.terminate_pod(self.pod_id)
            except Exception:
                pass

        candidates = build_gpu_fallback_types(self.gpu_type, self.gpu_fallback_types)
        last_error: Exception | None = None
        for idx, candidate in enumerate(candidates):
            name = f"binance-forever-{time.strftime('%H%M%S')}"
            log.info("provisioning pod: %s (%s)", name, candidate)
            try:
                config = PodConfig(name=name, gpu_type=candidate)
                pod = self.client.create_pod(config)
                self.pod_id = pod.id

                pod = self.client.wait_for_pod(pod.id)
                self.ssh_host = pod.ssh_host
                self.ssh_port = pod.ssh_port
                self.active_gpu_type = pod.gpu_type or candidate
                if idx > 0:
                    log.warning(
                        "requested %s unavailable, fell back to %s",
                        resolve_gpu_type(self.gpu_type),
                        self.active_gpu_type,
                    )
                log.info("pod ready: %s:%d", self.ssh_host, self.ssh_port)
                return self.ssh_host, self.ssh_port
            except Exception as e:
                last_error = e
                if self.pod_id:
                    try:
                        self.client.terminate_pod(self.pod_id)
                    except Exception:
                        pass
                    self.pod_id = None
                if is_capacity_error(e) and idx < len(candidates) - 1:
                    log.warning("pod provisioning failed for %s: %s", candidate, e)
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("RunPod provisioning failed without an exception")

    def bootstrap(self, remote_dir: str = "/workspace/stock-prediction") -> None:
        if self._bootstrapped:
            return
        log.info("syncing code to pod")
        _rsync_to_pod(self.ssh_host, self.ssh_port, REPO, remote_dir)
        log.info("bootstrapping pod")
        _bootstrap_pod(self.ssh_host, self.ssh_port, remote_dir)
        self._bootstrapped = True

    def upload_data(self, track: DataTrack, remote_dir: str = "/workspace/stock-prediction") -> None:
        for data_rel in (track.train, track.val):
            local = REPO / data_rel
            if local.exists():
                log.info("uploading %s", data_rel)
                _rsync_data(self.ssh_host, self.ssh_port, local, remote_dir)
            else:
                log.warning("data file not found: %s", local)

    def terminate(self) -> None:
        if self.client and self.pod_id:
            try:
                self.client.terminate_pod(self.pod_id)
                log.info("pod terminated: %s", self.pod_id)
            except Exception as e:
                log.warning("failed to terminate pod: %s", e)
            self.pod_id = None
            self.ssh_host = ""
            self.ssh_port = 0
            self._bootstrapped = False

    def hourly_rate(self) -> float:
        resolved = self.active_gpu_type or resolve_gpu_type(self.gpu_type)
        return HOURLY_RATES.get(resolved, 0.0)

# ---------------------------------------------------------------------------
# Main forever loop
# ---------------------------------------------------------------------------

def run_forever(
    *,
    gpu_type: str = "4090",
    gpu_fallback_types: list[str] | None = None,
    time_budget: int = 300,
    max_trials_per_round: int = 15,
    tracks: list[DataTrack] | None = None,
    local: bool = False,
    once: bool = False,
    dry_run: bool = False,
    descriptions: str = "",
    remote_dir: str = "/workspace/stock-prediction",
    r2_sync: bool = True,
):
    if tracks is None:
        tracks = [CRYPTO_DAILY, CRYPTO_HOURLY]

    baseline = ProductionBaseline()
    log_file = REPO / "analysis" / "binance_autoresearch_forever_log.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    pod_mgr: Optional[PodManager] = None
    if not local:
        pod_mgr = PodManager(gpu_type=gpu_type, gpu_fallback_types=gpu_fallback_types)

    round_num = 0
    track_idx = 0
    _shutdown = False

    def _handle_signal(signum, frame):
        nonlocal _shutdown
        log.info("shutdown signal received, finishing current round")
        _shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if dry_run:
        _print_plan(gpu_type, gpu_fallback_types, time_budget, max_trials_per_round, tracks, local)
        return

    try:
        while not _shutdown:
            round_num += 1
            track = tracks[track_idx % len(tracks)]
            track_idx += 1
            run_id = f"binance_forever_{track.name}_{time.strftime('%Y%m%d_%H%M%S')}"

            log.info("=== round %d: %s (%s) ===", round_num, track.name, track.description)

            t0 = time.time()
            ssh_host, ssh_port = "", 0

            try:
                if pod_mgr:
                    ssh_host, ssh_port = pod_mgr.ensure_ready()
                    pod_mgr.bootstrap(remote_dir)
                    pod_mgr.upload_data(track, remote_dir)

                round_descriptions = descriptions or track.focused_descriptions
                leaderboard, checkpoint_dir = run_autoresearch_batch(
                    track,
                    time_budget=time_budget,
                    max_trials=max_trials_per_round,
                    run_id=run_id,
                    gpu_type=pod_mgr.active_gpu_type if pod_mgr else "local",
                    ssh_host=ssh_host,
                    ssh_port=ssh_port,
                    remote_dir=remote_dir,
                    local=local,
                    descriptions=round_descriptions,
                )

                training_time = time.time() - t0
                result = read_best_from_leaderboard(leaderboard, checkpoint_dir)
                result.track = track.name
                result.gpu_type = pod_mgr.active_gpu_type if pod_mgr else "local"
                result.training_time_s = training_time

                # log result
                log_entry = {
                    "round": round_num,
                    "run_id": run_id,
                    "track": track.name,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "description": result.description,
                    "val_return": result.val_return,
                    "val_sortino": result.val_sortino,
                    "win_rate": result.win_rate,
                    "holdout_robust_score": result.holdout_robust_score,
                    "combined_score": result.combined_score(),
                    "baseline_combined": baseline.combined_score(),
                    "beats_baseline": result.beats_baseline(baseline),
                    "training_time_s": training_time,
                    "checkpoint": result.best_checkpoint,
                    "error": result.error,
                }
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                if result.beats_baseline(baseline):
                    log.info("NEW BEST: %s combined=%.4f > baseline=%.4f",
                             result.description, result.combined_score(), baseline.combined_score())
                    write_success_progress(result, baseline, round_num)
                    baseline.val_return = result.val_return or baseline.val_return
                    baseline.val_sortino = result.val_sortino or baseline.val_sortino
                    baseline.win_rate = result.win_rate or baseline.win_rate
                    baseline.description = result.description
                    baseline.checkpoint = result.best_checkpoint or baseline.checkpoint
                else:
                    log.info("no improvement: %s combined=%.4f <= baseline=%.4f",
                             result.description, result.combined_score(), baseline.combined_score())
                    append_failure_progress(result, baseline, round_num)

                # R2 sync
                if r2_sync and os.environ.get("R2_ENDPOINT"):
                    sync_to_r2(checkpoint_dir, log_file, run_id)

                # top-K checkpoint management
                if result.best_checkpoint and result.val_sortino is not None:
                    try:
                        r2_prefix = f"binance_autoresearch/{run_id}" if r2_sync else None
                        mgr = TopKCheckpointManager(
                            checkpoint_dir, max_keep=5, mode="max", r2_prefix=r2_prefix,
                        )
                        mgr.register(result.best_checkpoint, result.val_sortino)
                    except Exception as e:
                        log.warning("checkpoint manager error: %s", e)

            except Exception as e:
                log.error("round %d failed: %s", round_num, e)
                traceback.print_exc()
                result = TrialResult(
                    description=f"round_{round_num}_error",
                    track=track.name,
                    error=str(e),
                    training_time_s=time.time() - t0,
                    gpu_type=pod_mgr.active_gpu_type if pod_mgr else "local",
                )
                append_failure_progress(result, baseline, round_num)

            if once:
                log.info("--once mode, exiting after round %d", round_num)
                break

            if _shutdown:
                break

            # brief pause between rounds
            log.info("round %d complete, next round in 10s", round_num)
            time.sleep(10)

    finally:
        if pod_mgr:
            pod_mgr.terminate()
        log.info("forever loop stopped after %d rounds", round_num)


def _print_plan(
    gpu_type: str,
    gpu_fallback_types: list[str] | None,
    time_budget: int,
    max_trials: int,
    tracks: list[DataTrack],
    local: bool,
):
    resolved = resolve_gpu_type(gpu_type)
    rate = HOURLY_RATES.get(resolved, 0.0)
    fallback_chain = build_gpu_fallback_types(gpu_type, gpu_fallback_types)
    round_time_min = (max_trials * time_budget + 1800) / 60  # +30min overhead

    print("Binance Autoresearch Forever -- Dry Run Plan")
    print(f"  Mode: {'LOCAL' if local else 'RunPod'}")
    print(f"  GPU: {resolved} @ ${rate:.2f}/hr")
    if not local:
        print(f"  GPU fallback order: {', '.join(fallback_chain)}")
    print(f"  Time budget: {time_budget}s per trial")
    print(f"  Max trials per round: {max_trials}")
    print(f"  Est. round time: {round_time_min:.0f} min")
    print(f"  Est. cost per round: ${rate * round_time_min / 60:.2f}")
    print()
    print("  Tracks (alternating):")
    for t in tracks:
        exists_train = "OK" if (REPO / t.train).exists() else "MISSING"
        exists_val = "OK" if (REPO / t.val).exists() else "MISSING"
        print(f"    {t.name}: {t.description}")
        print(f"      train: {t.train} [{exists_train}]")
        print(f"      val:   {t.val} [{exists_val}]")
        print(f"      periods/yr: {t.periods_per_year}, max_steps: {t.max_steps}, fee: {t.fee_rate}")
    print()
    print("  Workflow per round:")
    print("    1. Provision/reuse RunPod pod" if not local else "    1. Use local GPU")
    print("    2. rsync code + data")
    print("    3. autoresearch_rl (N trials with poly early stopping)")
    print("    4. Download leaderboard + checkpoints")
    print("    5. Compare best vs production baseline")
    print("    6. Write to binanceprogress{N}.md (success) or binanceprogress_failed.md")
    print("    7. Sync top-K checkpoints + logs to R2")
    print("    8. Loop forever (alternating tracks)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binance crypto autoresearch forever loop")
    p.add_argument("--gpu-type", default="4090", choices=[""] + sorted(GPU_ALIASES.keys()),
                   help="RunPod GPU type (default: 4090)")
    p.add_argument(
        "--gpu-fallback-types",
        default="",
        help=(
            "comma-separated RunPod fallback GPU aliases "
            f"(default: {','.join(DEFAULT_GPU_FALLBACKS)}; use 'none' to disable)"
        ),
    )
    p.add_argument("--time-budget", type=int, default=300, help="seconds per trial")
    p.add_argument("--max-trials", type=int, default=15, help="trials per round")
    p.add_argument("--local", action="store_true", help="run on local GPU")
    p.add_argument("--once", action="store_true", help="run one round then exit")
    p.add_argument("--dry-run", action="store_true", help="print plan and exit")
    p.add_argument("--tracks", default="crypto_daily,crypto_hourly",
                   help="comma-separated track names: crypto_daily, crypto_hourly, mixed_daily, mixed_hourly")
    p.add_argument("--descriptions", default="", help="comma-separated experiment descriptions to run")
    p.add_argument("--no-r2", action="store_true", help="disable R2 sync")
    p.add_argument("--remote-dir", default="/workspace/stock-prediction")
    return p.parse_args()


def main():
    args = parse_args()

    track_map = {t.name: t for t in ALL_TRACKS}
    tracks = []
    for name in args.tracks.split(","):
        name = name.strip()
        if name in track_map:
            tracks.append(track_map[name])
        else:
            log.error("unknown track: %s (available: %s)", name, ", ".join(track_map.keys()))
            sys.exit(1)

    run_forever(
        gpu_type=args.gpu_type,
        gpu_fallback_types=parse_gpu_fallback_types(args.gpu_fallback_types),
        time_budget=args.time_budget,
        max_trials_per_round=args.max_trials,
        tracks=tracks,
        local=args.local,
        once=args.once,
        dry_run=args.dry_run,
        descriptions=args.descriptions,
        remote_dir=args.remote_dir,
        r2_sync=not args.no_r2,
    )


if __name__ == "__main__":
    main()
