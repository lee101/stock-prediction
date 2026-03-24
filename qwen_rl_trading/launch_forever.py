#!/usr/bin/env python3
"""Qwen GRPO autoresearch forever loop on RunPod 4090s.

Usage:
    python -m qwen_rl_trading.launch_forever --gpu-type 4090
    python -m qwen_rl_trading.launch_forever --local
    python -m qwen_rl_trading.launch_forever --once --dry-run
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.binance_autoresearch_forever import (
    PodManager,
    ProductionBaseline,
    TrialResult,
    append_failure_progress,
    sync_to_r2,
    write_success_progress,
    _ssh_run,
    _rsync_to_pod,
    _rsync_data,
    _rsync_from_pod,
    _scp_from_pod,
    _fmt,
)
from src.checkpoint_manager import TopKCheckpointManager
from src.runpod_client import HOURLY_RATES, resolve_gpu_type

log = logging.getLogger("qwen_forever")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_SIZES = ["0.6B", "1.8B", "3B", "7B"]

QWEN_BOOTSTRAP_EXTRA = (
    "uv pip install trl>=0.15 peft>=0.14 bitsandbytes>=0.45 outlines>=0.1 -q"
)


def _bootstrap_qwen_pod(host: str, port: int, remote_dir: str) -> None:
    """Bootstrap pod with Qwen RL deps on top of base env."""
    bootstrap = (
        f"set -euo pipefail && cd {remote_dir} && "
        f"pip install uv -q && "
        f"uv venv .venv313 --python python3.13 2>/dev/null || uv venv .venv313 && "
        f"source .venv313/bin/activate && "
        f"uv pip install -e . -q && "
        f"{QWEN_BOOTSTRAP_EXTRA}"
    )
    _ssh_run(host, port, bootstrap)


def _upload_qwen_data(host: str, port: int, remote_dir: str) -> None:
    """Upload crypto CSVs and forecast cache."""
    crypto_dir = REPO / "trainingdatahourly" / "crypto"
    forecast_dir = REPO / "binanceneural" / "forecast_cache"
    sft_data = REPO / "rl-trainingbinance" / "trading_plans_train.jsonl"

    for local_path in [crypto_dir, forecast_dir]:
        if local_path.exists():
            rel = local_path.relative_to(REPO)
            _ssh_run(host, port, f"mkdir -p {remote_dir}/{rel}")
            cmd = [
                "rsync", "-az",
                "-e", f"ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p {port}",
                f"{local_path}/", f"root@{host}:{remote_dir}/{rel}/",
            ]
            log.info("uploading %s", rel)
            subprocess.run(cmd, check=False)

    if sft_data.exists():
        rel = sft_data.relative_to(REPO)
        _ssh_run(host, port, f"mkdir -p {remote_dir}/{rel.parent}")
        cmd = [
            "rsync", "-az",
            "-e", f"ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p {port}",
            str(sft_data), f"root@{host}:{remote_dir}/{rel}",
        ]
        subprocess.run(cmd, check=False)


def run_qwen_batch_remote(
    host: str, port: int, remote_dir: str,
    model_sizes: list[str], time_budget: int, max_trials: int,
    descriptions: str = "",
) -> tuple[Path, Path]:
    """Run Qwen autoresearch on remote pod."""
    run_id = f"qwen_forever_{time.strftime('%Y%m%d_%H%M%S')}"
    remote_lb = f"qwen_rl_trading/{run_id}_leaderboard.csv"
    remote_ckpt = f"qwen_rl_trading/checkpoints/{run_id}"

    cmd_parts = [
        "python", "-u", "-m", "qwen_rl_trading.autoresearch_qwen",
        "--time-budget", str(time_budget),
        "--max-trials", str(max_trials),
        "--leaderboard", remote_lb,
        "--checkpoint-root", remote_ckpt,
    ]
    if descriptions:
        cmd_parts += ["--descriptions", descriptions]

    shell_cmd = (
        f"cd {remote_dir} && source .venv313/bin/activate && "
        f"export PYTHONPATH={remote_dir}:${{PYTHONPATH:-}} && "
        f"{' '.join(cmd_parts)}"
    )
    log.info("running on pod %s:%d: %s", host, port, run_id)
    _ssh_run(host, port, shell_cmd, check=False)

    # download results
    local_lb = REPO / "analysis" / f"{run_id}_leaderboard.csv"
    local_ckpt = REPO / "qwen_rl_trading" / "checkpoints" / run_id
    local_lb.parent.mkdir(parents=True, exist_ok=True)
    _scp_from_pod(host, port, f"{remote_dir}/{remote_lb}", local_lb)
    _rsync_from_pod(host, port, f"{remote_dir}/{remote_ckpt}", local_ckpt)

    return local_lb, local_ckpt


def run_qwen_batch_local(
    model_sizes: list[str], time_budget: int, max_trials: int,
    descriptions: str = "",
) -> tuple[Path, Path]:
    """Run Qwen autoresearch locally."""
    from .autoresearch_qwen import run_autoresearch
    from .sweep_configs import EXPERIMENTS

    run_id = f"qwen_forever_{time.strftime('%Y%m%d_%H%M%S')}"
    leaderboard = REPO / "qwen_rl_trading" / f"{run_id}_leaderboard.csv"
    checkpoint_root = REPO / "qwen_rl_trading" / "checkpoints" / run_id

    # filter experiments to requested model sizes
    exps = [e for e in EXPERIMENTS if e.get("model_size", "0.6B") in model_sizes]

    run_autoresearch(
        exps,
        time_budget=time_budget,
        max_trials=max_trials,
        checkpoint_root=checkpoint_root,
        leaderboard_path=leaderboard,
        descriptions=descriptions,
    )
    return leaderboard, checkpoint_root


def read_best_from_qwen_leaderboard(leaderboard: Path) -> TrialResult:
    """Parse leaderboard CSV and return best result."""
    import csv
    result = TrialResult(description="none", track="qwen_rl")
    if not leaderboard.exists():
        result.error = "leaderboard not found"
        return result

    try:
        with open(leaderboard) as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        result.error = str(e)
        return result

    if not rows:
        result.error = "empty leaderboard"
        return result

    def _safe(v):
        if v in (None, "", "None", "nan"):
            return None
        try:
            f = float(v)
            return None if f != f else f
        except (TypeError, ValueError):
            return None

    def score(r):
        vr = _safe(r.get("val_mean_return"))
        vs = _safe(r.get("val_mean_sortino"))
        if vr is None and vs is None:
            return -float("inf")
        return 0.5 * (vr or 0) + 0.5 * (vs or 0)

    rows.sort(key=score, reverse=True)
    best = rows[0]

    result.description = best.get("description", "unknown")
    result.val_return = _safe(best.get("val_mean_return"))
    result.val_sortino = _safe(best.get("val_mean_sortino"))
    result.best_checkpoint = best.get("best_checkpoint")
    result.config = {k: best.get(k) for k in ["model_size", "lora_r", "group_size", "lr", "kl_coef"]}
    return result


def run_forever(
    *,
    gpu_type: str = "4090",
    time_budget: int = 600,
    max_trials_per_round: int = 10,
    model_sizes: list[str] | None = None,
    local: bool = False,
    once: bool = False,
    dry_run: bool = False,
    descriptions: str = "",
    remote_dir: str = "/workspace/stock-prediction",
    r2_sync: bool = True,
):
    if model_sizes is None:
        model_sizes = ["0.6B", "1.8B"]

    baseline = ProductionBaseline()
    log_file = REPO / "analysis" / "qwen_autoresearch_forever_log.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    pod_mgr: Optional[PodManager] = None
    if not local:
        pod_mgr = PodManager(gpu_type=gpu_type)

    round_num = 0
    _shutdown = False

    def _handle_signal(signum, frame):
        nonlocal _shutdown
        log.info("shutdown signal, finishing current round")
        _shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if dry_run:
        resolved = resolve_gpu_type(gpu_type)
        rate = HOURLY_RATES.get(resolved, 0.0)
        round_time_min = (max_trials_per_round * time_budget + 1800) / 60
        print("Qwen RL Autoresearch Forever -- Dry Run")
        print(f"  Mode: {'LOCAL' if local else 'RunPod'}")
        print(f"  GPU: {resolved} @ ${rate:.2f}/hr")
        print(f"  Time budget: {time_budget}s per trial")
        print(f"  Max trials per round: {max_trials_per_round}")
        print(f"  Model sizes: {model_sizes}")
        print(f"  Est. round time: {round_time_min:.0f} min")
        print(f"  Est. cost per round: ${rate * round_time_min / 60:.2f}")
        return

    try:
        while not _shutdown:
            round_num += 1
            run_id = f"qwen_round{round_num}_{time.strftime('%Y%m%d_%H%M%S')}"
            log.info("=== round %d: model_sizes=%s ===", round_num, model_sizes)

            t0 = time.time()
            try:
                if local:
                    leaderboard, checkpoint_dir = run_qwen_batch_local(
                        model_sizes, time_budget, max_trials_per_round, descriptions,
                    )
                else:
                    ssh_host, ssh_port = pod_mgr.ensure_ready()
                    log.info("syncing code to pod")
                    _rsync_to_pod(ssh_host, ssh_port, REPO, remote_dir)
                    log.info("bootstrapping qwen deps")
                    _bootstrap_qwen_pod(ssh_host, ssh_port, remote_dir)
                    log.info("uploading data")
                    _upload_qwen_data(ssh_host, ssh_port, remote_dir)

                    leaderboard, checkpoint_dir = run_qwen_batch_remote(
                        ssh_host, ssh_port, remote_dir,
                        model_sizes, time_budget, max_trials_per_round, descriptions,
                    )

                training_time = time.time() - t0
                result = read_best_from_qwen_leaderboard(leaderboard)
                result.track = "qwen_rl"
                result.gpu_type = gpu_type if pod_mgr else "local"
                result.training_time_s = training_time

                # log
                log_entry = {
                    "round": round_num, "run_id": run_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "description": result.description,
                    "val_return": result.val_return,
                    "val_sortino": result.val_sortino,
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
                    log.info("NEW BEST: %s combined=%.4f", result.description, result.combined_score())
                    write_success_progress(result, baseline, round_num)
                    baseline.val_return = result.val_return or baseline.val_return
                    baseline.val_sortino = result.val_sortino or baseline.val_sortino
                    baseline.description = result.description
                    baseline.checkpoint = result.best_checkpoint or baseline.checkpoint
                else:
                    log.info("no improvement: %s combined=%.4f <= baseline=%.4f",
                             result.description, result.combined_score(), baseline.combined_score())
                    append_failure_progress(result, baseline, round_num)

                if r2_sync and os.environ.get("R2_ENDPOINT"):
                    sync_to_r2(checkpoint_dir, log_file, run_id)

                if result.best_checkpoint and result.val_sortino is not None:
                    try:
                        r2_prefix = f"qwen_autoresearch/{run_id}" if r2_sync else None
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
                    description=f"qwen_round_{round_num}_error",
                    track="qwen_rl",
                    error=str(e),
                    training_time_s=time.time() - t0,
                    gpu_type=gpu_type if pod_mgr else "local",
                )
                append_failure_progress(result, baseline, round_num)

            if once:
                log.info("--once mode, exiting")
                break
            if _shutdown:
                break
            log.info("round %d complete, next in 10s", round_num)
            time.sleep(10)

    finally:
        if pod_mgr:
            pod_mgr.terminate()
        log.info("forever loop stopped after %d rounds", round_num)


def main():
    p = argparse.ArgumentParser(description="Qwen GRPO autoresearch forever loop")
    p.add_argument("--gpu-type", default="4090")
    p.add_argument("--time-budget", type=int, default=600)
    p.add_argument("--max-trials", type=int, default=10)
    p.add_argument("--model-sizes", default="0.6B,1.8B", help="comma-separated model sizes")
    p.add_argument("--local", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--descriptions", default="")
    p.add_argument("--no-r2-sync", action="store_true")
    p.add_argument("--remote-dir", default="/workspace/stock-prediction")
    args = p.parse_args()

    run_forever(
        gpu_type=args.gpu_type,
        time_budget=args.time_budget,
        max_trials_per_round=args.max_trials,
        model_sizes=args.model_sizes.split(","),
        local=args.local,
        once=args.once,
        dry_run=args.dry_run,
        descriptions=args.descriptions,
        remote_dir=args.remote_dir,
        r2_sync=not args.no_r2_sync,
    )


if __name__ == "__main__":
    main()
