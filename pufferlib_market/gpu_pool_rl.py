"""GPU pool scheduler for RL trading experiments.

Manages a dynamic pool of RunPod GPU pods with configurable limits per GPU type.
Queues RL autoresearch experiments, assigns them to available pods, and handles
pod lifecycle.

Usage:
  # Show pool status
  python -m pufferlib_market.gpu_pool_rl status

  # Run an RL experiment on a GPU
  python -m pufferlib_market.gpu_pool_rl run \\
      --experiment daily_crypto10 --gpu a100 \\
      --time-budget 3600 \\
      --train-data pufferlib_market/data/crypto10_daily_train.bin \\
      --val-data pufferlib_market/data/crypto10_daily_val.bin

  # Teardown idle pods
  python -m pufferlib_market.gpu_pool_rl cleanup

  # Change pool limits
  python -m pufferlib_market.gpu_pool_rl set-limit --gpu a100 --count 2
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_a, **_k) -> bool:
        return False

load_dotenv()

try:
    from src.runpod_client import RunPodClient, Pod, PodConfig
    from src.runpod_client import HOURLY_RATES as _RC_HOURLY_RATES
    from src.runpod_client import GPU_ALIASES as _RC_GPU_ALIASES
except ImportError:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "cutellm"))
        from runpod_client import RunPodClient, Pod, PodConfig  # type: ignore[no-redef]
        from runpod_client import HOURLY_RATES as _RC_HOURLY_RATES  # type: ignore[no-redef]
        from runpod_client import GPU_ALIASES as _RC_GPU_ALIASES  # type: ignore[no-redef]
    except ImportError:
        RunPodClient = None  # type: ignore[assignment,misc]
        Pod = None  # type: ignore[assignment,misc]
        PodConfig = None  # type: ignore[assignment,misc]
        _RC_HOURLY_RATES = None  # type: ignore[assignment]
        _RC_GPU_ALIASES = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_STATE_FILE = REPO_ROOT / "analysis" / "rl_training_pool.json"

# Fallback GPU aliases/rates used when src.runpod_client is unavailable.
_FALLBACK_GPU_ALIASES: dict[str, str] = {
    "a100": "NVIDIA A100 80GB PCIe",
    "a100-sxm": "NVIDIA A100-SXM4-80GB",
    "h100": "NVIDIA H100 80GB HBM3",
    "h100-sxm": "NVIDIA H100 SXM",
    "4090": "NVIDIA GeForce RTX 4090",
    "5090": "NVIDIA GeForce RTX 5090",
    "l40s": "NVIDIA L40S",
    "l40": "NVIDIA L40",
    "a40": "NVIDIA A40",
    "6000-ada": "NVIDIA RTX 6000 Ada Generation",
    "rtx6000": "NVIDIA RTX 6000 Ada Generation",
}
_FALLBACK_HOURLY_RATES: dict[str, float] = {
    "NVIDIA A100 80GB PCIe": 1.19,
    "NVIDIA A100-SXM4-80GB": 1.39,
    "NVIDIA H100 80GB HBM3": 1.99,
    "NVIDIA H100 SXM": 2.69,
    "NVIDIA GeForce RTX 4090": 0.34,
    "NVIDIA GeForce RTX 5090": 0.69,
    "NVIDIA L40S": 0.79,
    "NVIDIA L40": 0.69,
    "NVIDIA A40": 0.35,
    "NVIDIA RTX 6000 Ada Generation": 0.74,
}

GPU_ALIASES: dict[str, str] = (
    _RC_GPU_ALIASES if isinstance(_RC_GPU_ALIASES, dict) else _FALLBACK_GPU_ALIASES
)
HOURLY_RATES: dict[str, float] = (
    _RC_HOURLY_RATES if isinstance(_RC_HOURLY_RATES, dict) else _FALLBACK_HOURLY_RATES
)

DEFAULT_POOL_LIMITS: dict[str, int] = {
    "NVIDIA A40": 2,
    "NVIDIA RTX 6000 Ada Generation": 1,
    "NVIDIA A100 80GB PCIe": 1,
    "NVIDIA H100 80GB HBM3": 0,
}

SETUP_OVERHEAD_SECS = 1800  # code sync + bootstrap + teardown


def estimate_cost(gpu_type: str, num_seeds: int = 1, time_budget_secs: int = 300) -> float:
    """Estimate total cost for a training run including setup overhead.

    Args:
        gpu_type: Full GPU display name (e.g. "NVIDIA A100 80GB PCIe") or alias (e.g. "a100").
        num_seeds: Number of sequential seeds / trials.
        time_budget_secs: Seconds per seed.

    Returns:
        Estimated cost in USD.
    """
    resolved = GPU_ALIASES.get(gpu_type.lower(), gpu_type)
    rate = HOURLY_RATES.get(resolved, HOURLY_RATES.get(gpu_type, 0.0))
    total_secs = SETUP_OVERHEAD_SECS + num_seeds * time_budget_secs
    return rate * (total_secs / 3600)


def _print_cost_estimate(gpu_type: str, time_budget_secs: int) -> float:
    """Print a cost-estimate block and return the estimated USD amount."""
    rate = HOURLY_RATES.get(gpu_type, 0.0)
    setup_min = SETUP_OVERHEAD_SECS // 60
    training_min = time_budget_secs // 60
    total_min = (SETUP_OVERHEAD_SECS + time_budget_secs) // 60
    est_cost = estimate_cost(gpu_type, num_seeds=1, time_budget_secs=time_budget_secs)
    print("Cost estimate:")
    print(f"  GPU: {gpu_type} @ ${rate:.2f}/hr")
    print(f"  Setup overhead: {setup_min}min")
    print(f"  Training: 1 seed x {training_min}min = {training_min}min")
    print(f"  Total: {total_min}min -> ~${est_cost:.2f}")
    return est_cost


def check_running_cost(state: "PoolState", budget_limit: float) -> None:
    """Warn if any busy pod has been running past the budget limit duration."""
    if budget_limit <= 0:
        return
    now = datetime.now(timezone.utc)
    for name, pod in state.pods.items():
        if pod.status != "busy" or not pod.created_at:
            continue
        rate = HOURLY_RATES.get(pod.gpu_type, 0.0)
        if rate <= 0:
            continue
        try:
            started = datetime.fromisoformat(pod.created_at)
            hours_running = (now - started).total_seconds() / 3600
            cost_so_far = rate * hours_running
            if cost_so_far > budget_limit:
                print(
                    f"  WARNING: pod '{name}' has been running {hours_running:.1f}h "
                    f"(${cost_so_far:.2f} spent, budget ${budget_limit:.2f}). "
                    f"Consider stopping it."
                )
        except (ValueError, OverflowError):
            pass


@dataclass
class PoolPod:
    pod_id: str
    name: str
    gpu_type: str
    gpu_count: int
    status: str  # "provisioning", "ready", "busy", "stopped", "dead"
    ssh_host: str = ""
    ssh_port: int = 0
    created_at: str = ""
    current_experiment: str = ""


@dataclass
class PoolState:
    pods: dict[str, PoolPod] = field(default_factory=dict)
    limits: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_POOL_LIMITS))
    total_experiments_run: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_gpu(alias: str) -> str:
    return GPU_ALIASES.get(alias.lower(), alias)


def load_pool_state() -> PoolState:
    if POOL_STATE_FILE.exists():
        data = json.loads(POOL_STATE_FILE.read_text())
        pods = {}
        for name, info in data.get("pods", {}).items():
            pods[name] = PoolPod(**info)
        return PoolState(
            pods=pods,
            limits=data.get("limits", dict(DEFAULT_POOL_LIMITS)),
            total_experiments_run=data.get("total_experiments_run", 0),
        )
    return PoolState()


def save_pool_state(state: PoolState) -> None:
    POOL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "pods": {name: asdict(pod) for name, pod in state.pods.items()},
        "limits": state.limits,
        "total_experiments_run": state.total_experiments_run,
    }
    POOL_STATE_FILE.write_text(json.dumps(data, indent=2) + "\n")


def _count_gpu_type(state: PoolState, gpu_type: str) -> int:
    return sum(
        1 for p in state.pods.values()
        if p.gpu_type == gpu_type and p.status not in ("dead", "stopped")
    )


def _find_available_pod(state: PoolState, gpu_type: str) -> Optional[PoolPod]:
    for pod in state.pods.values():
        if pod.gpu_type == gpu_type and pod.status == "ready":
            return pod
    return None


def refresh_pod_status(state: PoolState, client: "RunPodClient") -> None:
    """Sync pool state with actual RunPod API state."""
    try:
        live_pods = {p.id: p for p in client.list_pods()}
    except Exception as exc:
        print(f"Warning: could not list pods: {exc}")
        return

    for name, pool_pod in list(state.pods.items()):
        live = live_pods.get(pool_pod.pod_id)
        if live is None:
            pool_pod.status = "dead"
            continue
        if live.status != "RUNNING":
            pool_pod.status = "stopped"
            continue
        # Pod exists and is running — refresh SSH details
        try:
            full = client.get_pod(pool_pod.pod_id)
            pool_pod.ssh_host = full.ssh_host
            pool_pod.ssh_port = full.ssh_port
            if pool_pod.status == "provisioning" and full.ssh_host and full.ssh_port:
                pool_pod.status = "ready"
        except Exception:
            pass  # keep existing status on transient errors

    dead = [n for n, p in state.pods.items() if p.status == "dead"]
    for n in dead:
        del state.pods[n]
        print(f"  Removed dead pod: {n}")


def provision_pod(
    state: PoolState,
    client: "RunPodClient",
    gpu_type: str,
    gpu_count: int = 1,
    name: Optional[str] = None,
) -> PoolPod:
    """Create a new pod in the pool."""
    limit = state.limits.get(gpu_type, 0)
    current = _count_gpu_type(state, gpu_type)
    if current >= limit:
        raise RuntimeError(
            f"Pool limit reached: {current}/{limit} {gpu_type} pods. "
            "Increase limit with set-limit or cleanup existing pods."
        )

    if name is None:
        short = next((k for k, v in GPU_ALIASES.items() if v == gpu_type), "gpu")
        idx = current + 1
        name = f"rl-{short}-{idx}"

    config = PodConfig(
        name=name,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        volume_size=200,
        container_disk=60,
    )
    print(f"Creating pod '{name}' ({gpu_count}x {gpu_type})...")
    pod = client.create_pod(config)
    pool_pod = PoolPod(
        pod_id=pod.id,
        name=name,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        status="provisioning",
        created_at=_utc_now(),
    )
    state.pods[name] = pool_pod
    save_pool_state(state)
    print(f"  Pod created: {pod.id}")

    # Wait for SSH to become available
    try:
        ready_pod = client.wait_for_pod(pod.id, timeout=600)
        pool_pod.ssh_host = ready_pod.ssh_host
        pool_pod.ssh_port = ready_pod.ssh_port
        pool_pod.status = "ready"
        save_pool_state(state)
        print(f"  Pod ready: ssh -p {pool_pod.ssh_port} root@{pool_pod.ssh_host}")
    except TimeoutError:
        pool_pod.status = "provisioning"
        save_pool_state(state)
        print("  Warning: pod created but SSH not ready yet")

    return pool_pod


def get_or_create_pod(
    state: PoolState,
    client: "RunPodClient",
    gpu_type: str,
    gpu_count: int = 1,
) -> PoolPod:
    """Return an available ready pod, or provision a new one within limits."""
    pod = _find_available_pod(state, gpu_type)
    if pod:
        return pod
    return provision_pod(state, client, gpu_type, gpu_count)


def ssh_exec(pod: PoolPod, script: str) -> subprocess.CompletedProcess:
    """Run a bash script on the pod over SSH.  No timeout — training may be long."""
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=30",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=10",
        "-p", str(pod.ssh_port),
        f"root@{pod.ssh_host}",
        f"bash -lc {shlex.quote(script)}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def scp_from(pod: PoolPod, remote: str, local: str) -> None:
    """Download a file from the pod."""
    cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-P", str(pod.ssh_port),
        f"root@{pod.ssh_host}:{remote}",
        local,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"scp_from failed: {result.stderr.strip()}")


def sync_data_files(
    pod: PoolPod,
    *,
    repo_root: Path,
    data_files: list[str],
    remote_dir: str = "/workspace/stock-prediction",
) -> None:
    """Rsync specific data files to the pod."""
    if not data_files:
        return
    ssh_exec(pod, f"mkdir -p {shlex.quote(remote_dir)}/pufferlib_market/data")
    for df in data_files:
        local_path = repo_root / df
        if not local_path.exists():
            print(f"  [warn] data file not found locally: {local_path}")
            continue
        remote_path = f"root@{pod.ssh_host}:{remote_dir}/{df}"
        cmd = [
            "rsync", "-az",
            "-e", f"ssh -o StrictHostKeyChecking=no -p {pod.ssh_port}",
            str(local_path),
            remote_path,
        ]
        print(f"  Syncing {df} ({local_path.stat().st_size / 1e6:.1f}MB)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode not in (0, 23):
            raise RuntimeError(f"rsync data failed: {result.stderr.strip()}")
    print(f"  Data sync complete ({len(data_files)} files)")


def bootstrap_pod(
    pod: PoolPod,
    *,
    repo_root: Path,
    remote_dir: str = "/workspace/stock-prediction",
) -> None:
    """Rsync code + install deps + build C extension on pod."""
    print(f"  Bootstrapping {pod.name}...")

    rsync_cmd = [
        "rsync", "-az", "--delete",
        "--exclude", "__pycache__/",
        "--exclude", ".git/",
        "--exclude", "pufferlib_market/data/",
        "--exclude", "pufferlib_market/checkpoints/",
        "-e", f"ssh -o StrictHostKeyChecking=no -p {pod.ssh_port}",
        f"{repo_root}/",
        f"root@{pod.ssh_host}:{remote_dir}/",
    ]
    result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode not in (0, 23):
        raise RuntimeError(f"rsync failed (exit {result.returncode}): {result.stderr.strip()}")

    setup_script = f"""
set -euo pipefail
cd {shlex.quote(remote_dir)}
pip install uv --quiet
uv venv .venv313 --python python3.13 2>/dev/null || uv venv .venv313
source .venv313/bin/activate
uv pip install -e . --quiet
# Install vendored PufferLib so pufferlib_market.train can import it.
if [ -d PufferLib ]; then
    uv pip install -e PufferLib/ --quiet
fi
# Build the C trading environment extension.
cd pufferlib_market && python setup.py build_ext --inplace && cd ..
echo "BOOTSTRAP_OK"
"""
    result = ssh_exec(pod, setup_script)
    if "BOOTSTRAP_OK" not in result.stdout:
        raise RuntimeError(
            f"Bootstrap failed on {pod.name}:\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
    print(f"  Bootstrap complete on {pod.name}")


def _detect_remote_h100(pod: PoolPod) -> bool:
    """Return True when the pod's first GPU is an H100 (via nvidia-smi over SSH)."""
    result = ssh_exec(pod, "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1")
    if result.returncode == 0:
        first_line = result.stdout.strip().lower()
        return "h100" in first_line
    return False


def run_rl_experiment_on_pod(
    pod: PoolPod,
    *,
    run_id: str,
    train_data: str,
    val_data: str,
    time_budget: int = 300,
    max_trials: int = 50,
    wandb_project: str = "stock",
    wandb_api_key: str = "",
    checkpoint_dir: str = "",
    leaderboard: str = "",
    remote_dir: str = "/workspace/stock-prediction",
    descriptions: str = "",
) -> dict:
    """Run autoresearch on pod, pull back leaderboard + top checkpoints."""
    remote_leaderboard = leaderboard or f"pufferlib_market/{run_id}_leaderboard.csv"
    remote_checkpoint_dir = checkpoint_dir or f"pufferlib_market/checkpoints/{run_id}"

    wandb_export = ""
    if wandb_api_key:
        wandb_export = f"export WANDB_API_KEY={shlex.quote(wandb_api_key)}"
    if wandb_project:
        wandb_export += f"\nexport WANDB_PROJECT={shlex.quote(wandb_project)}"

    extra_args = []
    if descriptions:
        extra_args.extend(["--descriptions", shlex.quote(descriptions)])

    # Auto-detect H100 and add scale-up overrides when running on one.
    is_h100 = _detect_remote_h100(pod)
    if is_h100:
        print(f"  [{pod.name}] H100 detected — adding --num-envs 256 --cuda-graph-ppo")
        extra_args.extend(["--num-envs", "256", "--cuda-graph-ppo"])

    train_cmd = " ".join([
        "python", "-u", "-m", "pufferlib_market.autoresearch_rl",
        "--train-data", shlex.quote(train_data),
        "--val-data", shlex.quote(val_data),
        "--time-budget", str(int(time_budget)),
        "--max-trials", str(int(max_trials)),
        "--leaderboard", shlex.quote(remote_leaderboard),
        "--checkpoint-root", shlex.quote(remote_checkpoint_dir),
        *extra_args,
    ])

    experiment_script = f"""
set -euo pipefail
cd {shlex.quote(remote_dir)}
source .venv313/bin/activate
export PYTHONPATH="$PWD:$PWD/PufferLib:${{PYTHONPATH:-}}"
mkdir -p {shlex.quote(remote_checkpoint_dir)}
{wandb_export}
{train_cmd}
echo "EXPERIMENT_OK"
"""

    print(f"  [{pod.name}] Starting autoresearch run_id={run_id} (budget={time_budget}s)...")
    result = ssh_exec(pod, experiment_script)

    log_text = (result.stdout or "") + "\n" + (result.stderr or "")
    metrics: dict = {
        "run_id": run_id,
        "pod": pod.name,
        "status": "completed" if "EXPERIMENT_OK" in result.stdout else "failed",
        "returncode": result.returncode,
    }

    if metrics["status"] == "failed":
        print(f"  [{pod.name}] Experiment may have failed (rc={result.returncode})")
        print(f"  Last stderr: {result.stderr[-500:]}")

    # Pull leaderboard CSV back to local
    local_leaderboard = str(REPO_ROOT / remote_leaderboard)
    Path(local_leaderboard).parent.mkdir(parents=True, exist_ok=True)
    try:
        scp_from(pod, f"{remote_dir}/{remote_leaderboard}", local_leaderboard)
        metrics["leaderboard_path"] = local_leaderboard
        print(f"  [{pod.name}] Downloaded leaderboard: {local_leaderboard}")
    except RuntimeError as exc:
        print(f"  [{pod.name}] Warning: could not download leaderboard: {exc}")

    # Pull top-5 checkpoints back
    local_ckpt_dir = REPO_ROOT / remote_checkpoint_dir
    local_ckpt_dir.mkdir(parents=True, exist_ok=True)
    fetch_script = f"""
set -euo pipefail
cd {shlex.quote(remote_dir)}
# List best.pt and top-5 by modification time in checkpoint dir
find {shlex.quote(remote_checkpoint_dir)} -name "*.pt" | sort -t/ -k3 | tail -5 || true
ls {shlex.quote(remote_checkpoint_dir)}/best.pt 2>/dev/null || true
"""
    ckpt_result = ssh_exec(pod, fetch_script)
    ckpt_paths = [
        line.strip() for line in ckpt_result.stdout.splitlines() if line.strip().endswith(".pt")
    ]
    downloaded = 0
    for remote_ckpt in ckpt_paths[:5]:
        ckpt_name = Path(remote_ckpt).name
        local_ckpt = str(local_ckpt_dir / ckpt_name)
        try:
            scp_from(pod, f"{remote_dir}/{remote_ckpt}", local_ckpt)
            downloaded += 1
        except RuntimeError:
            pass
    if downloaded:
        metrics["checkpoints_downloaded"] = downloaded
        metrics["checkpoint_dir"] = str(local_ckpt_dir)
        print(f"  [{pod.name}] Downloaded {downloaded} checkpoints to {local_ckpt_dir}")

    print(f"  [{pod.name}] Done: status={metrics['status']}")
    return metrics


def cmd_status(args: argparse.Namespace) -> None:
    state = load_pool_state()

    # If the pool state file does not exist there is nothing to query — skip the
    # network call and print a helpful message instead.
    if not POOL_STATE_FILE.exists():
        print("No pool state found.")
        print(f"Limits: {json.dumps(state.limits)}")
        return

    client = _require_client()
    refresh_pod_status(state, client)
    save_pool_state(state)

    if not state.pods:
        print("Pool is empty (no pods).")
        print(f"Limits: {json.dumps(state.limits)}")
        return

    print(f"{'Name':<25} {'GPU':<30} {'Status':<12} {'Experiment':<30} {'Cost':>8}  {'SSH'}")
    print("-" * 145)
    total_cost = 0.0
    for name, pod in sorted(state.pods.items()):
        rate = HOURLY_RATES.get(pod.gpu_type, 0)
        hours = 0.0
        if pod.created_at:
            try:
                dt = datetime.fromisoformat(pod.created_at)
                hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
            except ValueError:
                pass
        cost = rate * hours
        total_cost += cost
        cost_str = f"${cost:.2f}" if rate > 0 else "N/A"
        ssh = f"ssh -p {pod.ssh_port} root@{pod.ssh_host}" if pod.ssh_host else "N/A"
        exp = pod.current_experiment or "-"
        print(f"{name:<25} {pod.gpu_type:<30} {pod.status:<12} {exp:<30} {cost_str:>8}  {ssh}")

    print(f"\nLimits: {json.dumps(state.limits)}")
    print(f"Total experiments run: {state.total_experiments_run}")
    print(f"Estimated total cost so far: ${total_cost:.2f}")


def cmd_run(args: argparse.Namespace) -> None:
    dry_run: bool = getattr(args, "dry_run", False)

    if dry_run:
        gpu_type = _resolve_gpu(args.gpu)
        budget_limit: float = getattr(args, "budget_limit", 10.0)
        print("[dry-run] Would run experiment:")
        print(f"  experiment:   {args.experiment}")
        print(f"  gpu:          {gpu_type}")
        print(f"  time_budget:  {args.time_budget}s")
        print(f"  max_trials:   {args.max_trials}")
        print(f"  train_data:   {args.train_data}")
        print(f"  val_data:     {args.val_data}")
        print(f"  remote_dir:   {args.remote_dir}")
        print(f"  budget_limit: ${budget_limit:.2f}")
        est_cost = _print_cost_estimate(gpu_type, args.time_budget)
        if budget_limit > 0 and est_cost > budget_limit:
            print(f"[dry-run] WARNING: estimated cost ${est_cost:.2f} exceeds budget limit ${budget_limit:.2f}")
        else:
            print("[dry-run] Cost check passed.")
        print("[dry-run] No pod provisioned.")
        return

    client = _require_client()
    state = load_pool_state()
    refresh_pod_status(state, client)

    gpu_type = _resolve_gpu(args.gpu)
    budget_limit = getattr(args, "budget_limit", 10.0)

    est_cost = _print_cost_estimate(gpu_type, args.time_budget)
    if budget_limit > 0 and est_cost > budget_limit:
        print(f"Estimated cost: ${est_cost:.2f} exceeds budget limit ${budget_limit:.2f}")
        print(f"Use --budget-limit {est_cost + 0.5:.1f} to allow this, or --budget-limit 0 to disable limit.")
        raise SystemExit(1)

    check_running_cost(state, budget_limit)

    pod = get_or_create_pod(state, client, gpu_type, gpu_count=args.gpu_count)

    if not pod.ssh_host or not pod.ssh_port:
        print("Pod not SSH-ready, waiting...")
        ready = client.wait_for_pod(pod.pod_id, timeout=600)
        pod.ssh_host = ready.ssh_host
        pod.ssh_port = ready.ssh_port
        pod.status = "ready"

    remote_dir = args.remote_dir
    bootstrap_pod(pod, repo_root=REPO_ROOT, remote_dir=remote_dir)

    data_files = [args.train_data, args.val_data]
    sync_data_files(pod, repo_root=REPO_ROOT, data_files=data_files, remote_dir=remote_dir)

    pod.status = "busy"
    pod.current_experiment = args.experiment
    save_pool_state(state)

    try:
        metrics = run_rl_experiment_on_pod(
            pod,
            run_id=args.experiment,
            train_data=args.train_data,
            val_data=args.val_data,
            time_budget=args.time_budget,
            max_trials=args.max_trials,
            wandb_project=args.wandb_project,
            wandb_api_key=os.environ.get("WANDB_API_KEY", ""),
            remote_dir=remote_dir,
            descriptions=getattr(args, "descriptions", ""),
        )
    finally:
        pod.status = "ready"
        pod.current_experiment = ""
        state.total_experiments_run += 1
        save_pool_state(state)

    if args.stop_after:
        print(f"Stopping pod {pod.name}...")
        client.stop_pod(pod.pod_id)
        pod.status = "stopped"
        save_pool_state(state)

    print(f"\nMetrics: {json.dumps(metrics, indent=2)}")


def cmd_cleanup(args: argparse.Namespace) -> None:
    client = _require_client()
    state = load_pool_state()
    refresh_pod_status(state, client)

    to_remove = [
        name for name, pod in state.pods.items()
        if pod.status in ("dead", "stopped") or (args.all and pod.status == "ready")
    ]

    if not to_remove:
        print("No pods to clean up.")
        return

    for name in to_remove:
        pod = state.pods[name]
        if pod.status != "dead":
            try:
                client.terminate_pod(pod.pod_id)
                print(f"Terminated: {name}")
            except Exception as exc:
                print(f"Warning: could not terminate {name}: {exc}")
        del state.pods[name]

    save_pool_state(state)
    print(f"Cleaned up {len(to_remove)} pods.")


def cmd_set_limit(args: argparse.Namespace) -> None:
    state = load_pool_state()
    gpu_type = _resolve_gpu(args.gpu)
    state.limits[gpu_type] = args.count
    save_pool_state(state)
    print(f"Set limit: {gpu_type} = {args.count}")
    print(f"All limits: {json.dumps(state.limits, indent=2)}")


def _require_client() -> "RunPodClient":
    if RunPodClient is None:
        print("Error: RunPodClient not available — install src/runpod_client.py or cutellm.", file=sys.stderr)
        sys.exit(1)
    try:
        return RunPodClient()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU pool scheduler for RL trading experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # status
    sub.add_parser("status", help="Show pool status")

    # run
    run_p = sub.add_parser("run", help="Run an RL experiment on a GPU pod")
    run_p.add_argument("--experiment", "-e", required=True, help="Experiment / run ID")
    run_p.add_argument("--gpu", default="4090", help="GPU type alias (4090, a40, 6000-ada, a100, …)")
    run_p.add_argument("--gpu-count", type=int, default=1)
    run_p.add_argument("--train-data", required=True, help="Path to training .bin on pod")
    run_p.add_argument("--val-data", required=True, help="Path to validation .bin on pod")
    run_p.add_argument("--time-budget", type=int, default=300, help="Autoresearch time budget (seconds)")
    run_p.add_argument("--max-trials", type=int, default=50, help="Max autoresearch trials")
    run_p.add_argument("--wandb-project", default="stock", help="W&B project name")
    run_p.add_argument("--remote-dir", default="/workspace/stock-prediction")
    run_p.add_argument("--stop-after", action="store_true", help="Stop pod after experiment")
    run_p.add_argument(
        "--budget-limit", type=float, default=10.0,
        help="Max USD to spend on this run (setup + training). 0 = no limit.",
    )
    run_p.add_argument(
        "--descriptions", default="",
        help="Comma-separated subset of experiment descriptions to run",
    )
    run_p.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and print what would be done without provisioning a pod",
    )

    # cleanup
    clean_p = sub.add_parser("cleanup", help="Remove dead/stopped pods")
    clean_p.add_argument("--all", action="store_true", help="Also stop and remove ready pods")

    # set-limit
    limit_p = sub.add_parser("set-limit", help="Set max pods per GPU type")
    limit_p.add_argument("--gpu", required=True, help="GPU type alias or full name")
    limit_p.add_argument("--count", type=int, required=True, help="Max pods of this type")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "cleanup":
        cmd_cleanup(args)
    elif args.command == "set-limit":
        cmd_set_limit(args)


if __name__ == "__main__":
    main()
