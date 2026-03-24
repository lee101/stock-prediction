#!/usr/bin/env python3
"""Launch parallel RunPod pods for large-scale RL seed sweeps.

Splits a seed range across N pods so they all train in parallel.
Each pod runs pufferlib_market/seed_sweep_rl.py for its slice of seeds.
Results are pulled back via SCP and merged into a single leaderboard CSV.

Usage:
    # 4 H100 pods: crypto70 daily, seeds 1-100, 30 min/seed
    python scripts/launch_runpod_scale.py \\
        --dataset crypto70_daily \\
        --seeds 1 100 \\
        --time-budget 1800 \\
        --n-pods 4 \\
        --gpu h100

    # 4 H100 pods: crypto40 hourly, seeds 1-60, 30 min/seed
    python scripts/launch_runpod_scale.py \\
        --dataset crypto40_hourly \\
        --seeds 1 60 \\
        --time-budget 1800 \\
        --n-pods 4 \\
        --gpu h100

    # Local multi-GPU (no RunPod):
    python scripts/launch_runpod_scale.py \\
        --dataset crypto40_hourly --seeds 1 60 --time-budget 1800 --local

    # Dry run (print plan without launching):
    python scripts/launch_runpod_scale.py --dataset crypto70_daily --seeds 1 100 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import sys
import threading
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.runpod_client import RunPodClient, PodConfig, resolve_gpu_type, build_gpu_fallback_types
from pufferlib_market.gpu_pool_rl import (
    PoolPod, bootstrap_pod, sync_data_files, ssh_exec, scp_from,
    _detect_remote_h100, HOURLY_RATES, GPU_ALIASES,
)

REMOTE_DIR = "/workspace/stock-prediction"

# Data files needed per dataset
DATASET_DATA_FILES = {
    "crypto70_daily": [
        "pufferlib_market/data/crypto70_daily_train.bin",
        "pufferlib_market/data/crypto70_daily_val.bin",
    ],
    "crypto40_daily": [
        "pufferlib_market/data/crypto40_daily_train.bin",
        "pufferlib_market/data/crypto40_daily_val.bin",
    ],
    "crypto40_hourly": [
        "pufferlib_market/data/crypto40_hourly_train.bin",
        "pufferlib_market/data/crypto40_hourly_val.bin",
    ],
    "crypto34_hourly": [
        "pufferlib_market/data/crypto34_hourly_train.bin",
        "pufferlib_market/data/crypto34_hourly_val.bin",
    ],
}


def split_seeds(seeds: list[int], n_pods: int) -> list[list[int]]:
    """Split seeds into n_pods roughly equal slices."""
    chunk = math.ceil(len(seeds) / n_pods)
    return [seeds[i:i+chunk] for i in range(0, len(seeds), chunk) if seeds[i:i+chunk]]


def run_sweep_on_pod(
    pod: PoolPod,
    *,
    dataset: str,
    seeds: list[int],
    time_budget: int,
    run_id: str,
    results: dict,
    dry_run: bool = False,
) -> None:
    """Thread target: run seed sweep on a single pod, pull back CSV."""
    pod_name = pod.name
    remote_leaderboard = f"{REMOTE_DIR}/sweepresults/{run_id}_{pod_name}.csv"
    remote_ckpt = f"{REMOTE_DIR}/pufferlib_market/checkpoints/{run_id}_{pod_name}"
    local_leaderboard = REPO / "sweepresults" / f"{run_id}_{pod_name}.csv"
    local_leaderboard.parent.mkdir(parents=True, exist_ok=True)

    seed_args = f"{seeds[0]} {seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1]+1)) else ""
    seed_list_args = " ".join(str(s) for s in seeds) if not seed_args else ""

    # Detect H100 for tuned settings (exclusive GPU → cuda-graph safe)
    is_h100 = _detect_remote_h100(pod)
    num_envs = 256 if is_h100 else 128

    if seed_args:
        seeds_flag = f"--seeds {seed_args}"
    else:
        seeds_flag = f"--seed-list {seed_list_args}"

    cuda_graph_flag = "--cuda-graph" if is_h100 else ""

    train_cmd = " ".join(filter(None, [
        "python", "-u", "-m", "pufferlib_market.seed_sweep_rl",
        "--dataset", shlex.quote(dataset),
        seeds_flag,
        "--time-budget", str(time_budget),
        "--num-envs", str(num_envs),
        cuda_graph_flag,
        "--leaderboard", shlex.quote(f"sweepresults/{run_id}_{pod_name}.csv"),
        "--checkpoint-root", shlex.quote(f"pufferlib_market/checkpoints/{run_id}_{pod_name}"),
    ]))

    script = f"""
set -euo pipefail
cd {shlex.quote(REMOTE_DIR)}
source .venv313/bin/activate
export PYTHONPATH="$PWD:$PWD/PufferLib:${{PYTHONPATH:-}}"
mkdir -p sweepresults pufferlib_market/checkpoints
{train_cmd}
echo "SWEEP_OK"
"""

    print(f"[{pod_name}] seeds={seeds[0]}-{seeds[-1]} ({len(seeds)} seeds)  "
          f"h100={is_h100}  budget={time_budget}s/seed", flush=True)
    if dry_run:
        print(f"[{pod_name}] DRY RUN — would run:\n{train_cmd}", flush=True)
        results[pod_name] = {"status": "dry_run", "seeds": seeds}
        return

    t0 = time.time()
    result = ssh_exec(pod, script)
    elapsed = time.time() - t0
    ok = "SWEEP_OK" in result.stdout

    print(f"[{pod_name}] done in {elapsed/60:.1f}min  ok={ok}", flush=True)
    if not ok:
        print(f"[{pod_name}] stderr tail: {result.stderr[-500:]}", flush=True)

    # Pull leaderboard back
    try:
        scp_from(pod, remote_leaderboard, str(local_leaderboard))
        print(f"[{pod_name}] leaderboard downloaded: {local_leaderboard}", flush=True)
    except Exception as exc:
        print(f"[{pod_name}] warn: could not download leaderboard: {exc}", flush=True)

    # Pull top checkpoints (best.pt files only to save bandwidth)
    try:
        ckpt_script = f"""
cd {shlex.quote(REMOTE_DIR)}
find {shlex.quote(f"pufferlib_market/checkpoints/{run_id}_{pod_name}")} -name "best.pt" 2>/dev/null || true
"""
        ckpt_result = ssh_exec(pod, ckpt_script)
        ckpt_paths = [p.strip() for p in ckpt_result.stdout.strip().splitlines() if p.strip()]
        local_ckpt_root = REPO / "pufferlib_market" / "checkpoints" / f"{run_id}_{pod_name}"
        for rp in ckpt_paths:
            rel = Path(rp).relative_to(f"pufferlib_market/checkpoints/{run_id}_{pod_name}")
            local_p = local_ckpt_root / rel
            local_p.parent.mkdir(parents=True, exist_ok=True)
            scp_from(pod, f"{REMOTE_DIR}/{rp}", str(local_p))
        if ckpt_paths:
            print(f"[{pod_name}] downloaded {len(ckpt_paths)} best.pt files", flush=True)
    except Exception as exc:
        print(f"[{pod_name}] warn: checkpoint download failed: {exc}", flush=True)

    results[pod_name] = {
        "status": "ok" if ok else "failed",
        "seeds": seeds,
        "elapsed_s": elapsed,
        "leaderboard": str(local_leaderboard),
    }


def merge_leaderboards(run_id: str, n_pods: int) -> Path:
    """Merge per-pod CSV files into a single sorted leaderboard."""
    out_path = REPO / "sweepresults" / f"{run_id}_merged.csv"
    all_rows: list[dict] = []
    for i in range(n_pods):
        for p in (REPO / "sweepresults").glob(f"{run_id}_pod*.csv"):
            if not p.exists():
                continue
            with open(p) as fh:
                rows = list(csv.DictReader(fh))
                all_rows.extend(rows)

    if not all_rows:
        print("No leaderboard rows found to merge.")
        return out_path

    # Deduplicate by seed+dataset
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for row in all_rows:
        key = (row.get("dataset", ""), row.get("seed", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    # Sort by val_return descending
    def _val_ret(r: dict) -> float:
        try:
            return float(r.get("val_return") or 0)
        except (ValueError, TypeError):
            return 0.0

    deduped.sort(key=_val_ret, reverse=True)

    if deduped:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(deduped[0].keys()))
            writer.writeheader()
            writer.writerows(deduped)
        print(f"\nMerged {len(deduped)} results → {out_path}")

        # Print top 10
        print("\nTop 10 by val_return:")
        for i, row in enumerate(deduped[:10]):
            print(f"  {i+1:2d}. seed={row.get('seed'):>4s}  "
                  f"val_ret={row.get('val_return','?'):>8s}  "
                  f"val_sort={row.get('val_sortino','?'):>6s}  "
                  f"holdout_med={row.get('holdout_median_return_pct','?'):>8s}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch parallel RunPod seed sweeps")
    parser.add_argument("--dataset", required=True,
                        choices=["crypto70_daily", "crypto40_daily", "crypto40_hourly", "crypto34_hourly"],
                        help="Dataset to train on")
    parser.add_argument("--seeds", nargs=2, type=int, metavar=("START", "END"), default=[1, 60])
    parser.add_argument("--time-budget", type=int, default=1800,
                        help="Training budget per seed in seconds (default 1800=30min)")
    parser.add_argument("--n-pods", type=int, default=4,
                        help="Number of parallel RunPod pods (default 4)")
    parser.add_argument("--gpu", type=str, default="h100",
                        help="GPU type: h100, a100, 4090 (default h100)")
    parser.add_argument("--local", action="store_true",
                        help="Run locally instead of RunPod (single GPU, sequential)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without launching anything")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run identifier (default: auto-generated)")
    args = parser.parse_args()

    seeds = list(range(args.seeds[0], args.seeds[1] + 1))
    run_id = args.run_id or f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Cost estimate
    gpu_type = resolve_gpu_type(args.gpu)
    rate = HOURLY_RATES.get(gpu_type, 0.0)
    setup_overhead = 1800  # 30 min
    pod_time_s = setup_overhead + math.ceil(len(seeds) / args.n_pods) * args.time_budget
    est_cost_per_pod = rate * pod_time_s / 3600
    total_cost = est_cost_per_pod * args.n_pods

    print(f"=== RunPod Scale Sweep ===")
    print(f"  dataset:     {args.dataset}")
    print(f"  seeds:       {seeds[0]}-{seeds[-1]} ({len(seeds)} total)")
    print(f"  time/seed:   {args.time_budget}s ({args.time_budget/60:.0f} min)")
    print(f"  n_pods:      {args.n_pods}")
    print(f"  gpu:         {gpu_type}")
    print(f"  run_id:      {run_id}")
    print(f"  seeds/pod:   ~{math.ceil(len(seeds)/args.n_pods)}")
    print(f"  wall time:   ~{pod_time_s/60:.0f} min")
    print(f"  cost/pod:    ~${est_cost_per_pod:.2f}")
    print(f"  TOTAL COST:  ~${total_cost:.2f}")
    print()

    if args.local:
        # Run locally
        from pufferlib_market.seed_sweep_rl import run_sweep
        run_sweep(
            dataset_key=args.dataset,
            seeds=seeds,
            time_budget=args.time_budget,
            leaderboard=REPO / "sweepresults" / f"{run_id}.csv",
            checkpoint_root=REPO / "pufferlib_market" / "checkpoints" / run_id,
            overrides={},
        )
        return

    if args.dry_run:
        print("DRY RUN — would launch these pod assignments:")
        for i, chunk in enumerate(split_seeds(seeds, args.n_pods)):
            print(f"  pod{i}: seeds {chunk[0]}-{chunk[-1]} ({len(chunk)} seeds)")
        print(f"\nTotal wall time: ~{pod_time_s/60:.0f} min at ${rate:.2f}/hr = ${total_cost:.2f}")
        return

    # Launch RunPod pods
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY not set in environment")
        sys.exit(1)

    client = RunPodClient(api_key=api_key)
    seed_chunks = split_seeds(seeds, args.n_pods)
    pods: list[PoolPod] = []
    data_files = DATASET_DATA_FILES.get(args.dataset, [])

    print(f"Provisioning {len(seed_chunks)} pods...", flush=True)
    for i, chunk in enumerate(seed_chunks):
        pod_name = f"{run_id}_pod{i}"
        config = PodConfig(
            name=pod_name,
            gpu_type=gpu_type,
            gpu_count=1,
            volume_size=80,
            container_disk=50,
        )
        print(f"  Provisioning {pod_name} ({gpu_type})...", flush=True)
        try:
            raw_pod = client.create_pod(config)
            pool_pod = PoolPod(
                pod_id=raw_pod.id,
                name=pod_name,
                gpu_type=gpu_type,
                gpu_count=1,
                status="provisioning",
                ssh_host=raw_pod.ssh_host or "",
                ssh_port=raw_pod.ssh_port or 22,
            )
            pods.append(pool_pod)
            print(f"  Created {pod_name} (id={raw_pod.id})", flush=True)
        except Exception as exc:
            print(f"  ERROR provisioning pod {i}: {exc}", flush=True)
            # Continue with fewer pods
            seed_chunks_remaining = seed_chunks[i:]
            if len(pods) == 0:
                print("No pods launched, aborting.")
                sys.exit(1)
            break

    # Wait for pods to be ready
    print(f"\nWaiting for {len(pods)} pods to start...", flush=True)
    ready_pods: list[PoolPod] = []
    for pod in pods:
        try:
            raw = client.wait_for_pod(pod.pod_id, timeout=600)
            pod.ssh_host = raw.ssh_host or pod.ssh_host
            pod.ssh_port = raw.ssh_port or pod.ssh_port
            pod.status = "ready"
            ready_pods.append(pod)
            print(f"  {pod.name} ready at {pod.ssh_host}:{pod.ssh_port}", flush=True)
        except Exception as exc:
            print(f"  {pod.name} failed to start: {exc}", flush=True)

    if not ready_pods:
        print("No pods ready. Aborting.")
        sys.exit(1)

    # Bootstrap + sync data to each pod
    print(f"\nBootstrapping {len(ready_pods)} pods...", flush=True)
    booted: list[PoolPod] = []
    for pod in ready_pods:
        try:
            bootstrap_pod(pod, repo_root=REPO, remote_dir=REMOTE_DIR)
            sync_data_files(pod, repo_root=REPO, data_files=data_files, remote_dir=REMOTE_DIR)
            booted.append(pod)
        except Exception as exc:
            print(f"  {pod.name} bootstrap failed: {exc}", flush=True)

    if not booted:
        print("All bootstraps failed. Aborting.")
        sys.exit(1)

    # Re-assign seeds to only booted pods
    seed_chunks = split_seeds(seeds, len(booted))
    print(f"\nStarting sweeps on {len(booted)} pods in parallel...", flush=True)

    results: dict = {}
    threads = []
    for pod, chunk in zip(booted, seed_chunks):
        t = threading.Thread(
            target=run_sweep_on_pod,
            kwargs=dict(
                pod=pod,
                dataset=args.dataset,
                seeds=chunk,
                time_budget=args.time_budget,
                run_id=run_id,
                results=results,
                dry_run=False,
            ),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Merge results
    print("\nMerging results...", flush=True)
    merged = merge_leaderboards(run_id, len(booted))

    # Terminate pods
    print("\nTerminating pods...", flush=True)
    for pod in booted:
        try:
            client.terminate_pod(pod.pod_id)
            print(f"  Terminated {pod.name}", flush=True)
        except Exception as exc:
            print(f"  {pod.name} termination failed: {exc}", flush=True)

    print(f"\nDone. Merged leaderboard: {merged}")


if __name__ == "__main__":
    main()
