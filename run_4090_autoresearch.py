#!/usr/bin/env python3
"""End-to-end RunPod RTX 4090 autoresearch job manager.

Provisions a RunPod RTX 4090 pod, syncs code + training data, launches the RL
autoresearch pipeline, polls until completion, downloads results (checkpoints +
leaderboard + logs), then **auto-terminates the pod**.

Usage:
    # Dry run — show steps, no real provisioning
    python run_4090_autoresearch.py --dry-run

    # Run stocks11_2012 autoresearch (default, 500 trials)
    python run_4090_autoresearch.py --dataset stocks11 --max-trials 500

    # Run stocks15_2012 autoresearch
    python run_4090_autoresearch.py --dataset stocks15 --max-trials 500

    # Run both sequentially (one pod each)
    python run_4090_autoresearch.py --dataset both --max-trials 500
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.runpod_client import (
    HOURLY_RATES,
    Pod,
    PodConfig,
    RunPodClient,
    TRAINING_DOCKER_IMAGE,
    resolve_gpu_type,
)
from src.remote_training_pipeline import (
    build_remote_autoresearch_plan,
    render_remote_pipeline_script,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REMOTE_DIR = "/workspace/stock-prediction"
POD_PROVISION_TIMEOUT = 600   # 10 min to become RUNNING
POLL_INTERVAL = 60            # seconds between progress checks

# Per-dataset parameters
DATASET_CONFIGS: dict[str, dict] = {
    "stocks11": {
        "train":           "pufferlib_market/data/stocks11_daily_train_2012.bin",
        "val":             "pufferlib_market/data/stocks11_daily_val_2012.bin",
        "leaderboard":     "autoresearch_stocks11_2012_leaderboard.csv",
        "checkpoint_root": "pufferlib_market/checkpoints/autoresearch_stock",
        "time_budget":     175,   # 200ts × 53,240 samples ÷ 92k sps ≈ 115s + headroom
    },
    "stocks15": {
        "train":           "pufferlib_market/data/stocks15_daily_train_2012.bin",
        "val":             "pufferlib_market/data/stocks15_daily_val_2012.bin",
        "leaderboard":     "autoresearch_stocks15_2012_leaderboard.csv",
        "checkpoint_root": "pufferlib_market/checkpoints/autoresearch_stocks15",
        "time_budget":     255,   # 200ts × 72,600 samples ÷ 92k sps ≈ 158s + headroom
    },
}

log = logging.getLogger("run_4090_autoresearch")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path)),
        ],
    )


# ---------------------------------------------------------------------------
# SSH / rsync helpers (port-aware)
# ---------------------------------------------------------------------------

def _ssh_opts(pod: Pod) -> list[str]:
    return [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=30",
        "-p", str(pod.ssh_port),
    ]


def _ssh_host(pod: Pod) -> str:
    return f"root@{pod.ssh_host}"


def _run_ssh(
    pod: Pod,
    cmd: str,
    *,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    ssh_cmd = ["ssh"] + _ssh_opts(pod) + [_ssh_host(pod), cmd]
    log.info("SSH [%s:%d]: %s", pod.ssh_host, pod.ssh_port, cmd[:140])
    return subprocess.run(ssh_cmd, check=check, capture_output=capture, text=True)


def _run_rsync(
    srcs: list[str],
    dst: str,
    *,
    pod: Pod,
    excludes: Sequence[str] = (),
    flags: str = "-azR",
) -> None:
    ssh_e = "ssh " + " ".join(_ssh_opts(pod))
    cmd = ["rsync", flags, "--progress", "-e", ssh_e]
    for ex in excludes:
        cmd += ["--exclude", ex]
    cmd += srcs + [dst]
    log.info("RSYNC %s -> %s", srcs, dst[:80])
    subprocess.run(cmd, check=True, cwd=str(REPO), text=True)


# ---------------------------------------------------------------------------
# Pod provisioning
# ---------------------------------------------------------------------------

def provision_pod(
    client: RunPodClient,
    gpu_type: str,
    pod_name: str,
    *,
    volume_gb: int = 120,
    disk_gb: int = 40,
) -> Pod:
    full_name = resolve_gpu_type(gpu_type)
    log.info("Provisioning pod: name=%s  gpu=%s", pod_name, full_name)
    config = PodConfig(
        name=pod_name,
        gpu_type=full_name,
        image=TRAINING_DOCKER_IMAGE,
        volume_size=volume_gb,
        container_disk=disk_gb,
    )
    pod = client.create_pod(config)
    log.info("Pod created: id=%s  (waiting for RUNNING…)", pod.id)
    pod = client.wait_for_pod(pod.id, timeout=POD_PROVISION_TIMEOUT)
    log.info("Pod RUNNING: %s:%d", pod.ssh_host, pod.ssh_port)
    return pod


# ---------------------------------------------------------------------------
# Code + data sync
# ---------------------------------------------------------------------------

def rsync_to_pod(pod: Pod, train_data: str, val_data: str) -> None:
    """Sync code and the two data bins to the pod."""
    remote_root = f"{_ssh_host(pod)}:{REMOTE_DIR}/"

    # 1. Code sync — exclude heavy data dirs; they'll be pushed separately
    _run_rsync(
        [
            "launch_stocks_autoresearch_remote.py",
            "launch_stocks15_2012_remote.py",
            "src/",
            "pufferlib_market/",
            "AGENTS.md",
        ],
        remote_root,
        pod=pod,
        excludes=[
            "pufferlib_market/data/",
            "pufferlib_market/checkpoints/",
            "__pycache__",
            "*.pyc",
            "*.so",
            ".venv",
            ".venv313",
            ".venv312",
            "chronos-forecasting/",
            "trainingdata/",
            "trainingdatahourly/",
            "sweepresults/",
        ],
    )

    # 2. Data bins — create remote dir first, then push each file
    _run_ssh(pod, f"mkdir -p {REMOTE_DIR}/pufferlib_market/data/")
    for data_rel in [train_data, val_data]:
        local = str(REPO / data_rel)
        remote = f"{_ssh_host(pod)}:{REMOTE_DIR}/{data_rel}"
        ssh_e = "ssh " + " ".join(_ssh_opts(pod))
        subprocess.run(
            ["rsync", "-az", "--progress", "-e", ssh_e, local, remote],
            check=True, text=True,
        )

    log.info("Code + data synced to pod")


# ---------------------------------------------------------------------------
# Pod environment setup
# ---------------------------------------------------------------------------

def setup_pod_env(pod: Pod) -> None:
    """Create venv (inheriting system PyTorch) and build C extension."""
    setup_cmd = " && ".join([
        f"cd {REMOTE_DIR}",
        # Create venv reusing the Docker image's PyTorch/CUDA
        "python -m venv --system-site-packages .venv",
        "source .venv/bin/activate",
        # Minimal extra deps (numpy usually already present)
        "pip install -q numpy 2>&1 | tail -2 || true",
        # Build C trading env extension
        "cd pufferlib_market && python setup.py build_ext --inplace 2>&1 | tail -10",
    ])
    _run_ssh(pod, f"bash -lc {repr(setup_cmd)}")
    log.info("Pod environment ready")


# ---------------------------------------------------------------------------
# Autoresearch launch
# ---------------------------------------------------------------------------

def launch_autoresearch(
    pod: Pod,
    *,
    dataset: str,
    max_trials: int,
    run_id: str,
) -> tuple[str, str]:
    """Build pipeline script, push to pod, launch nohup.

    Returns (remote_log_path, remote_pid_path).
    """
    cfg = DATASET_CONFIGS[dataset]

    plan = build_remote_autoresearch_plan(
        run_id=run_id,
        train_data_path=cfg["train"],
        val_data_path=cfg["val"],
        time_budget=int(cfg["time_budget"]),
        max_trials=max_trials,
        descriptions=[],
        rank_metric="holdout_robust_score",
        periods_per_year=252.0,
        max_steps_override=252,
        max_timesteps_per_sample=200,
        fee_rate_override=0.001,
        holdout_data=cfg["val"],
        holdout_eval_steps=90,
        holdout_n_windows=20,
        holdout_fee_rate=0.001,
        holdout_fill_buffer_bps=5.0,
        leaderboard_path=cfg["leaderboard"],
        checkpoint_root=cfg["checkpoint_root"],
        stocks_mode=True,
        start_from=326,
        seed_only=True,
        init_best_config="stock_trade_pen_05_s123",
        lock_best_config=True,
    )

    pipeline_script = render_remote_pipeline_script(
        remote_dir=REMOTE_DIR,
        remote_env=".venv",
        plan=plan,
    )

    remote_script = plan.remote_script_path
    remote_log    = plan.remote_log_path
    remote_pid    = plan.remote_pid_path

    # Push pipeline script via heredoc, then launch nohup
    bootstrap = "\n".join([
        "set -euo pipefail",
        f"cd {REMOTE_DIR}",
        f"mkdir -p $(dirname {remote_script})",
        f"mkdir -p $(dirname {remote_log})",
        # Write script (use a delimiter unlikely to appear in the script content)
        f"cat > {remote_script} <<'__AUTORESEARCH_PIPELINE__'",
        pipeline_script.rstrip("\n"),
        "__AUTORESEARCH_PIPELINE__",
        f"chmod +x {remote_script}",
        f"nohup bash {remote_script} > {remote_log} 2>&1 &",
        f"echo $! > {remote_pid}",
        f"echo LAUNCHED:$(cat {remote_pid})",
    ])

    result = _run_ssh(pod, f"bash -c {repr(bootstrap)}", capture=True)
    launched_line = next(
        (l for l in (result.stdout or "").splitlines() if l.startswith("LAUNCHED:")),
        "",
    )
    pid = launched_line.split(":", 1)[-1] if launched_line else "?"
    log.info(
        "Autoresearch launched: dataset=%s  PID=%s  log=%s",
        dataset, pid, remote_log,
    )
    return remote_log, remote_pid


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------

def poll_until_done(
    pod: Pod,
    *,
    remote_pid_path: str,
    remote_leaderboard: str,
    poll_interval: int = POLL_INTERVAL,
    max_wait_hours: float = 32.0,
) -> None:
    """Poll every `poll_interval` seconds until the autoresearch process exits."""
    deadline = time.monotonic() + max_wait_hours * 3600
    last_rows = -1
    check_num = 0

    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        check_num += 1

        # Is the process still alive?
        alive_res = _run_ssh(
            pod,
            f"PID=$(cat {remote_pid_path} 2>/dev/null || echo '') "
            f"&& [ -n \"$PID\" ] && kill -0 \"$PID\" 2>/dev/null && echo ALIVE || echo DONE",
            check=False, capture=True,
        )
        status = (alive_res.stdout or "").strip().splitlines()[-1] if alive_res.stdout else "UNKNOWN"

        # Leaderboard row count
        rows_res = _run_ssh(
            pod,
            f"wc -l < {remote_leaderboard} 2>/dev/null || echo 0",
            check=False, capture=True,
        )
        try:
            rows = int((rows_res.stdout or "0").strip())
        except ValueError:
            rows = 0

        if rows != last_rows or check_num % 10 == 0:
            elapsed_min = int((time.monotonic() - (deadline - max_wait_hours * 3600)) / 60)
            log.info(
                "[+%dm] status=%s  leaderboard_rows=%d",
                elapsed_min, status, rows,
            )
            last_rows = rows

        if status == "DONE":
            log.info("Autoresearch process finished. Total leaderboard rows: %d", rows)
            return

    log.warning(
        "Polling timed out after %.1fh. Proceeding with result download anyway.",
        max_wait_hours,
    )


# ---------------------------------------------------------------------------
# Result download
# ---------------------------------------------------------------------------

def download_results(pod: Pod, *, dataset: str, output_dir: Path) -> None:
    """Pull leaderboard, checkpoints, and logs from the pod."""
    cfg = DATASET_CONFIGS[dataset]
    output_dir.mkdir(parents=True, exist_ok=True)

    ssh_e = "ssh " + " ".join(_ssh_opts(pod))

    # Leaderboard CSV
    leaderboard_local = output_dir / cfg["leaderboard"]
    subprocess.run(
        ["rsync", "-az", "-e", ssh_e,
         f"{_ssh_host(pod)}:{REMOTE_DIR}/{cfg['leaderboard']}",
         str(leaderboard_local)],
        check=False, text=True,
    )
    log.info("Leaderboard: %s", leaderboard_local)

    # Checkpoints
    ckpt_local = output_dir / "checkpoints" / dataset
    ckpt_local.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-az", "-e", ssh_e,
         f"{_ssh_host(pod)}:{REMOTE_DIR}/{cfg['checkpoint_root']}/",
         str(ckpt_local) + "/"],
        check=False, text=True,
    )
    log.info("Checkpoints: %s", ckpt_local)

    # Pipeline logs (analysis/remote_runs/)
    logs_local = output_dir / "pod_logs" / dataset
    logs_local.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-az", "-e", ssh_e,
         f"{_ssh_host(pod)}:{REMOTE_DIR}/analysis/remote_runs/",
         str(logs_local) + "/"],
        check=False, text=True,
    )
    log.info("Logs: %s", logs_local)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(
    output_dir: Path,
    *,
    run_id: str,
    dataset: str,
    pod_id: str,
    pod_ip: str,
    max_trials: int,
    gpu_type: str,
    start_time: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    elapsed_h = (time.monotonic() - start_time) / 3600
    rate = HOURLY_RATES.get(resolve_gpu_type(gpu_type), 0.0)
    manifest = {
        "run_id": run_id,
        "dataset": dataset,
        "pod_id": pod_id,
        "pod_ip": pod_ip,
        "gpu_type": gpu_type,
        "max_trials": max_trials,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_hours": round(elapsed_h, 3),
        "cost_estimate_usd": round(rate * elapsed_h, 4),
    }
    path = output_dir / f"manifest_{run_id}_{dataset}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("Manifest: %s", path)
    return path


# ---------------------------------------------------------------------------
# Single-dataset pipeline
# ---------------------------------------------------------------------------

def run_dataset(
    client: RunPodClient,
    *,
    dataset: str,
    gpu_type: str,
    max_trials: int,
    run_id: str,
    output_dir: Path,
) -> bool:
    """Provision → sync → setup → launch → poll → download → terminate.

    Returns True on success.
    """
    cfg = DATASET_CONFIGS[dataset]
    pod_name = f"autoresearch-{dataset}-{run_id}"
    pod: Optional[Pod] = None
    start_time = time.monotonic()

    # max wait = trials × budget + 20% headroom
    max_hours = max_trials * cfg["time_budget"] / 3600 * 1.2
    log.info(
        "=== %s  |  %d trials × %ds  |  max %.1fh ===",
        dataset, max_trials, cfg["time_budget"], max_hours,
    )

    try:
        pod = provision_pod(client, gpu_type, pod_name)

        log.info("Syncing code + data…")
        rsync_to_pod(pod, cfg["train"], cfg["val"])

        log.info("Setting up pod environment…")
        setup_pod_env(pod)

        log.info("Launching autoresearch…")
        remote_log, remote_pid = launch_autoresearch(
            pod, dataset=dataset, max_trials=max_trials, run_id=run_id,
        )

        log.info("Polling (max %.1fh, check every %ds)…", max_hours, POLL_INTERVAL)
        poll_until_done(
            pod,
            remote_pid_path=remote_pid,
            remote_leaderboard=f"{REMOTE_DIR}/{cfg['leaderboard']}",
            max_wait_hours=max_hours,
        )

        log.info("Downloading results…")
        download_results(pod, dataset=dataset, output_dir=output_dir)

        write_manifest(
            output_dir,
            run_id=run_id,
            dataset=dataset,
            pod_id=pod.id,
            pod_ip=pod.ssh_host,
            max_trials=max_trials,
            gpu_type=gpu_type,
            start_time=start_time,
        )
        return True

    except Exception as exc:
        log.error("Pipeline failed for %s: %s", dataset, exc, exc_info=True)
        return False

    finally:
        if pod is not None:
            log.info("Terminating pod %s…", pod.id)
            try:
                client.terminate_pod(pod.id)
                log.info("Pod %s terminated.", pod.id)
            except Exception as exc:
                log.warning(
                    "WARN: Failed to terminate pod %s: %s  — terminate manually!",
                    pod.id, exc,
                )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end RunPod RTX 4090 autoresearch job manager. "
            "Provisions pod → syncs data → runs autoresearch → downloads results → "
            "auto-terminates pod."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["stocks11", "stocks15", "both"],
        default="stocks11",
        help="Which dataset to run. 'both' runs stocks11 then stocks15 on separate pods.",
    )
    parser.add_argument(
        "--max-trials", type=int, default=500,
        help="Number of autoresearch trials per dataset.",
    )
    parser.add_argument(
        "--gpu-type", default="4090",
        choices=["4090", "5090", "a40", "a100", "l40s"],
        help="RunPod GPU alias (default: 4090 @ $0.34/hr).",
    )
    parser.add_argument(
        "--output-dir", default="autoresearch_4090_results",
        help="Local directory to save leaderboards, checkpoints, and logs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print pipeline steps without provisioning any pods.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = time.strftime("4090_%Y%m%d_%H%M%S")

    output_dir = REPO / args.output_dir
    _setup_logging(output_dir / f"run_{run_id}.log")

    datasets = ["stocks11", "stocks15"] if args.dataset == "both" else [args.dataset]
    gpu_full = resolve_gpu_type(args.gpu_type)
    rate = HOURLY_RATES.get(gpu_full, 0.0)

    log.info("=== RunPod Autoresearch Job ===")
    log.info("run_id:    %s", run_id)
    log.info("datasets:  %s", datasets)
    log.info("trials:    %d per dataset", args.max_trials)
    log.info("gpu:       %s ($%.2f/hr)", gpu_full, rate)
    log.info("output:    %s", output_dir)

    if args.dry_run:
        for ds in datasets:
            cfg = DATASET_CONFIGS[ds]
            max_h = args.max_trials * cfg["time_budget"] / 3600 * 1.2
            cost = rate * max_h
            log.info(
                "[dry-run] %s: %d trials × %ds → max %.1fh → ~$%.2f",
                ds, args.max_trials, cfg["time_budget"], max_h, cost,
            )
            log.info("[dry-run]   train: %s", cfg["train"])
            log.info("[dry-run]   val:   %s", cfg["val"])
            log.info("[dry-run]   leaderboard: %s", cfg["leaderboard"])
        return 0

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        log.error("RUNPOD_API_KEY environment variable is not set")
        return 1

    client = RunPodClient(api_key=api_key)

    all_ok = True
    for ds in datasets:
        ok = run_dataset(
            client,
            dataset=ds,
            gpu_type=args.gpu_type,
            max_trials=args.max_trials,
            run_id=run_id,
            output_dir=output_dir,
        )
        all_ok = all_ok and ok

    log.info("=== All done. Success=%s ===", all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
