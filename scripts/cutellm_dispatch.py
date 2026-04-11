#!/usr/bin/env python3
"""Unified dispatch for both stock-prediction RL training and cutellm LLM training.

Uses the same RunPod infrastructure to route jobs to the right GPU type.

Stock-prediction RL: runs on RTX 5090 (single GPU), 5-min timeboxed
CuteLLM LLM: runs on 8xH100 (multi-GPU), 10-min training budget

Usage:
  # RL trading experiments:
  python scripts/cutellm_dispatch.py rl --config trade_pen_05 --dry-run
  python scripts/cutellm_dispatch.py rl --config trade_pen_05 --gpu-type 5090 --budget-limit 5

  # LLM experiments (cutellm):
  python scripts/cutellm_dispatch.py llm --experiment champion_int4_13L_ttt --dry-run
  python scripts/cutellm_dispatch.py llm --experiment best_record_seed1337 --gpu-type h100 --gpu-count 8

  # Status of current pods (both RL and LLM jobs):
  python scripts/cutellm_dispatch.py status

  # Cost summary:
  python scripts/cutellm_dispatch.py cost
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# cutellm project lives as a sibling at /nvme0n1-disk/code/cutellm
# or via the cutedsl symlink in the stock-prediction root.
CUTELLM_ROOT = Path("/nvme0n1-disk/code/cutellm")
if not CUTELLM_ROOT.exists():
    _symlink = REPO / "cutedsl"
    if _symlink.exists():
        CUTELLM_ROOT = _symlink.resolve()

from src.runpod_client import (  # noqa: E402
    GPU_ALIASES,
    HOURLY_RATES,
    TRAINING_GPU_TYPES,
    Pod,
    PodConfig,
    RunPodClient,
    resolve_gpu_type,
)
from src.runpod_remote_utils import SSH_OPTIONS, render_subprocess_error  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RL_GPU_TYPE = "5090"
DEFAULT_LLM_GPU_TYPE = "a100"
DEFAULT_LLM_GPU_COUNT = 1
DEFAULT_LLM_MAX_WALLCLOCK = 600   # seconds (10-min competition budget)
DEFAULT_RL_TIME_BUDGET = 300      # seconds per RL trial (5-min timeboxed)
DEFAULT_RL_MAX_TRIALS = 50
DEFAULT_BUDGET_LIMIT = 5.0        # USD guard across both project types
SETUP_OVERHEAD_SECS = 1800        # code sync + bootstrap + teardown

CUTELLM_DEFAULT_EXPERIMENTS_JSON = (
    CUTELLM_ROOT / "parameter-golf" / "autoresearch" / "default_experiments.json"
)
CUTELLM_PROGRESS_MD = CUTELLM_ROOT / "PROGRESS.md"
CUTELLM_REMOTE_DIR = "/workspace/cutellm"

_SSH_OPTS = list(SSH_OPTIONS)

DISPATCH_HOURLY_RATES: dict[str, float] = {
    "NVIDIA A100 80GB PCIe": 1.64,
    "NVIDIA A100-SXM4-80GB": 1.64,
    "NVIDIA H100 80GB HBM3": 3.89,
    "NVIDIA H100 SXM": 3.89,
    "NVIDIA GeForce RTX 5090": 1.25,
}


# ---------------------------------------------------------------------------
# Shared cost helpers
# ---------------------------------------------------------------------------


def _resolve_hourly_rate(gpu_type: str) -> tuple[str, float]:
    resolved = GPU_ALIASES.get(gpu_type.lower(), gpu_type)
    rate = DISPATCH_HOURLY_RATES.get(resolved)
    if rate is None:
        rate = HOURLY_RATES.get(resolved, HOURLY_RATES.get(gpu_type, 0.0))
    return resolved, rate


def estimate_cost(gpu_type: str, total_secs: int, gpu_count: int = 1) -> float:
    """Estimate cost in USD including setup overhead.

    Args:
        gpu_type: Short alias (e.g. 'a100', '5090') or full display name.
        total_secs: Total training seconds (NOT including setup overhead).
        gpu_count: Number of GPUs (cost scales linearly).

    Returns:
        Estimated cost in USD.
    """
    _resolved, rate = _resolve_hourly_rate(gpu_type)
    wall_secs = SETUP_OVERHEAD_SECS + total_secs
    return rate * gpu_count * (wall_secs / 3600)


def _print_cost_line(gpu_type: str, total_secs: int, gpu_count: int = 1) -> float:
    resolved, rate = _resolve_hourly_rate(gpu_type)
    est = rate * gpu_count * ((SETUP_OVERHEAD_SECS + total_secs) / 3600)
    total_min = (SETUP_OVERHEAD_SECS + total_secs) // 60
    gpu_str = f"{gpu_count}x " if gpu_count > 1 else ""
    print(f"  GPU: {gpu_str}{resolved} @ ${rate:.2f}/hr")
    print(f"  Wall time: ~{total_min}min (incl. {SETUP_OVERHEAD_SECS // 60}min setup)")
    print(f"  Estimated cost: ~${est:.2f}")
    return est


def _enforce_budget(est_cost: float, budget_limit: float, *, raise_on_exceed: bool = True) -> None:
    """Warn (and optionally raise SystemExit) when est_cost exceeds budget_limit (limit > 0)."""
    if budget_limit > 0 and est_cost > budget_limit:
        if raise_on_exceed:
            print(f"[budget] Estimated ${est_cost:.2f} exceeds limit ${budget_limit:.2f}.")
            print(f"  Use --budget-limit {est_cost + 0.5:.1f} to allow, or --budget-limit 0 to disable.")
            raise SystemExit(1)
        else:
            print(f"  [budget warning] ${est_cost:.2f} > limit ${budget_limit:.2f}")
            print(f"  Use --budget-limit {est_cost + 0.5:.1f} to allow.")


# ---------------------------------------------------------------------------
# SSH / rsync helpers (shared by rl + llm dispatch)
# ---------------------------------------------------------------------------


def _ssh_run(
    ssh_host: str,
    ssh_port: int,
    remote_cmd: str,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess:
    cmd = [
        "ssh",
        *_SSH_OPTS,
        "-p", str(ssh_port),
        f"root@{ssh_host}",
        remote_cmd,
    ]
    print(f"[ssh] {remote_cmd[:120]}")
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise render_subprocess_error(
            description="SSH command failed",
            cmd=cmd,
            result=result,
        )
    return result


def _rsync_to_pod(
    ssh_host: str,
    ssh_port: int,
    local_dir: Path,
    remote_dir: str,
    *,
    excludes: list[str] | None = None,
) -> None:
    base_excludes = ["__pycache__/", ".git/", ".venv*/", "*.pyc"]
    all_excludes = base_excludes + (excludes or [])
    cmd = [
        "rsync", "-az", "--delete",
        *[arg for ex in all_excludes for arg in ("--exclude", ex)],
        "-e", f"ssh {' '.join(_SSH_OPTS)} -p {ssh_port}",
        f"{local_dir}/",
        f"root@{ssh_host}:{remote_dir}/",
    ]
    print(f"[rsync] {local_dir} -> {remote_dir}")
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise render_subprocess_error(
            description=f"rsync to pod failed for {local_dir}",
            cmd=cmd,
            result=result,
        )


def _scp_from_pod(
    ssh_host: str,
    ssh_port: int,
    remote_path: str,
    local_path: Path,
) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp", *_SSH_OPTS,
        "-P", str(ssh_port),
        f"root@{ssh_host}:{remote_path}",
        str(local_path),
    ]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        print(render_subprocess_error(
            description=f"scp from pod failed for {remote_path}",
            cmd=cmd,
            result=result,
        ))
    return result.returncode == 0


def _wait_for_pod_ready(client: RunPodClient, pod: Pod) -> Pod:
    if pod.ssh_host and pod.ssh_port:
        return pod
    print("[dispatch] Waiting for pod to be ready ...")
    return client.wait_for_pod(pod.id)


# ---------------------------------------------------------------------------
# LLM experiment helpers (cutellm)
# ---------------------------------------------------------------------------


def load_llm_experiments() -> list[dict]:
    """Load experiments from cutellm default_experiments.json if available."""
    try:
        data = json.loads(CUTELLM_DEFAULT_EXPERIMENTS_JSON.read_text())
        return data.get("experiments", [])
    except (json.JSONDecodeError, OSError):
        return []


def get_best_bpb() -> Optional[float]:
    """Read our current best val_bpb from PROGRESS.md, or None."""
    try:
        lines = CUTELLM_PROGRESS_MD.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if "Our best score" in line and "BPB" in line:
            # Extract the float from e.g. "**1.1748 BPB**"
            parts = line.split()
            for i, p in enumerate(parts):
                if "BPB" in p and i > 0:
                    try:
                        return float(parts[i - 1].strip("*"))
                    except ValueError:
                        pass
    return None


def _llm_dry_run(args: argparse.Namespace) -> None:
    """Print LLM dispatch plan without executing."""
    gpu_type = resolve_gpu_type(args.gpu_type or DEFAULT_LLM_GPU_TYPE)
    gpu_count = args.gpu_count or DEFAULT_LLM_GPU_COUNT
    max_wallclock = getattr(args, "max_wallclock", DEFAULT_LLM_MAX_WALLCLOCK)
    total_secs = max_wallclock

    best_bpb = get_best_bpb()
    experiments = load_llm_experiments()
    selected_exp = _select_llm_experiment(args, experiments)

    print("[dry-run] LLM Dispatch Plan (cutellm)")
    if best_bpb is not None:
        print(f"  Current best BPB: {best_bpb:.4f}")
    print(f"  Experiment:  {selected_exp.get('name', 'N/A') if selected_exp else 'N/A'}")
    if selected_exp:
        print(f"  Description: {selected_exp.get('description', '')[:80]}")
        train_script = selected_exp.get("train_script", "parameter-golf/train_gpt.py")
        print(f"  Script:      {train_script}")
    print(f"  GPU:         {gpu_count}x {gpu_type}")
    print(f"  Max wallclock: {max_wallclock}s ({max_wallclock // 60}min)")
    print(f"  CuteLLM root: {CUTELLM_ROOT}")
    print()
    _print_cost_line(args.gpu_type or DEFAULT_LLM_GPU_TYPE, total_secs, gpu_count)
    print()
    print("  Steps:")
    print(f"    1. Provision RunPod pod ({gpu_count}x {gpu_type})")
    print(f"    2. rsync {CUTELLM_ROOT} -> {CUTELLM_REMOTE_DIR}")
    print("    3. Bootstrap: pip install uv, uv pip install -e parameter-golf/")
    print("    4. Download data (fineweb10B_sp1024 if missing)")
    print(f"    5. torchrun --nproc_per_node={gpu_count} train_gpt.py")
    print("    6. Download train.log + final_model.int8.ptz")
    print("    7. Terminate pod")


def _select_llm_experiment(
    args: argparse.Namespace,
    experiments: list[dict],
) -> Optional[dict]:
    """Return the experiment config matching args.experiment, or first available."""
    name = getattr(args, "experiment", None)
    if not experiments:
        return None
    if name:
        for exp in experiments:
            if exp.get("name") == name:
                return exp
        # Not found by exact match — return None so caller can warn
        return None
    return experiments[0]


def dispatch_llm(args: argparse.Namespace) -> int:
    """Provision a pod and run a cutellm LLM training experiment."""
    if not CUTELLM_ROOT.exists():
        print(f"[error] cutellm root not found: {CUTELLM_ROOT}")
        print("  Expected at /nvme0n1-disk/code/cutellm or via cutedsl symlink.")
        return 1

    gpu_type = resolve_gpu_type(args.gpu_type or DEFAULT_LLM_GPU_TYPE)
    gpu_count = args.gpu_count or DEFAULT_LLM_GPU_COUNT
    max_wallclock = getattr(args, "max_wallclock", DEFAULT_LLM_MAX_WALLCLOCK)
    budget_limit = getattr(args, "budget_limit", DEFAULT_BUDGET_LIMIT)

    est_cost = estimate_cost(args.gpu_type or DEFAULT_LLM_GPU_TYPE, max_wallclock, gpu_count)
    print(f"[llm] Estimated cost: ${est_cost:.2f}")
    _enforce_budget(est_cost, budget_limit)

    experiments = load_llm_experiments()
    selected = _select_llm_experiment(args, experiments)
    exp_name = getattr(args, "experiment", None) or (selected and selected.get("name")) or "default"

    if getattr(args, "experiment", None) and not selected:
        print(f"[warning] Experiment '{args.experiment}' not found in default_experiments.json; using raw train_gpt.py")

    train_script = "parameter-golf/train_gpt.py"
    env_overrides: dict[str, str] = {}
    if selected:
        train_script = selected.get("train_script", train_script)
        env_overrides = {str(k): str(v) for k, v in selected.get("env_overrides", {}).items()}

    try:
        client = RunPodClient()
    except ValueError as exc:
        print(f"[error] {exc}")
        print("  Set RUNPOD_API_KEY and retry.")
        return 1

    pod_name = f"llm-dispatch-{exp_name[:30]}"
    print(f"[llm] Provisioning pod {pod_name!r} ({gpu_count}x {gpu_type})")
    config = PodConfig(name=pod_name, gpu_type=gpu_type, gpu_count=gpu_count)
    pod = client.create_pod(config)
    print(f"[llm] Created pod {pod.id}")

    pod = _wait_for_pod_ready(client, pod)
    ssh_host = pod.ssh_host
    ssh_port = pod.ssh_port

    try:
        print("[llm] Syncing cutellm repo ...")
        _rsync_to_pod(
            ssh_host, ssh_port, CUTELLM_ROOT, CUTELLM_REMOTE_DIR,
            excludes=[
                "parameter-golf/data/datasets/",
                "parameter-golf/data/tokenizers/",
                "analysis/",
                "runs/",
                "*.pt", "*.ptz", "*.bin",
            ],
        )

        bootstrap = (
            f"cd {CUTELLM_REMOTE_DIR}/parameter-golf && "
            f"pip install uv -q && "
            f"uv pip install -r requirements.txt -q 2>&1 | tail -3 && "
            f"if [ ! -f data/tokenizers/fineweb_1024_bpe.model ]; then "
            f"  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10; "
            f"fi && echo BOOTSTRAP_OK"
        )
        _ssh_run(ssh_host, ssh_port, bootstrap)

        env_str = " ".join(
            f"export {shlex.quote(k)}={shlex.quote(v)};"
            for k, v in env_overrides.items()
        )
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        wandb_export = f"export WANDB_API_KEY={wandb_key};" if wandb_key else ""

        if not Path(train_script).is_absolute():
            remote_script = f"{CUTELLM_REMOTE_DIR}/{train_script}"
        else:
            remote_script = train_script

        run_cmd = (
            f"cd {CUTELLM_REMOTE_DIR}/parameter-golf && "
            f"{wandb_export} "
            f"{env_str} "
            f"torchrun --standalone --nproc_per_node={gpu_count} {remote_script}"
        )
        result = _ssh_run(ssh_host, ssh_port, run_cmd, check=False)
        exit_code = result.returncode

        run_ts = time.strftime("%Y%m%d_%H%M%S")
        local_out = REPO / "analysis" / "llm_runs" / f"{exp_name}_{run_ts}"
        local_out.mkdir(parents=True, exist_ok=True)
        _scp_from_pod(ssh_host, ssh_port,
                      f"{CUTELLM_REMOTE_DIR}/parameter-golf/train.log",
                      local_out / "train.log")
        _scp_from_pod(ssh_host, ssh_port,
                      f"{CUTELLM_REMOTE_DIR}/parameter-golf/final_model.int8.ptz",
                      local_out / "final_model.int8.ptz")
        print(f"[llm] Artifacts saved to: {local_out}")
        return exit_code

    finally:
        print(f"[llm] Terminating pod {pod.id} ...")
        try:
            client.terminate_pod(pod.id)
        except Exception as exc:
            print(f"[warning] Could not terminate pod {pod.id}: {exc}")


# ---------------------------------------------------------------------------
# RL dispatch helpers (delegates to dispatch_rl_training logic)
# ---------------------------------------------------------------------------


# Preset configs: short names -> data file pairs + descriptions for autoresearch_rl
RL_PRESET_CONFIGS: dict[str, dict] = {
    "trade_pen_05": {
        "data_train": "pufferlib_market/data/crypto10_daily_train.bin",
        "data_val": "pufferlib_market/data/crypto10_daily_val.bin",
        "descriptions": "trade_pen_05",
        "note": "Best daily config: trade_penalty=0.05, 100% profitable",
    },
    "fdusd3_daily": {
        "data_train": "pufferlib_market/data/fdusd3_daily_train.bin",
        "data_val": "pufferlib_market/data/fdusd3_daily_val.bin",
        "descriptions": "",
        "note": "FDUSD 0-fee BTC/ETH/SOL daily RL",
    },
    "fdusd3_hourly": {
        "data_train": "pufferlib_market/data/fdusd3_hourly_train.bin",
        "data_val": "pufferlib_market/data/fdusd3_hourly_val.bin",
        "descriptions": "",
        "note": "FDUSD 0-fee BTC/ETH/SOL hourly RL",
    },
    "crypto10_daily": {
        "data_train": "pufferlib_market/data/crypto10_daily_train.bin",
        "data_val": "pufferlib_market/data/crypto10_daily_val.bin",
        "descriptions": "",
        "note": "crypto10 daily autoresearch sweep",
    },
}


def _rl_dry_run(args: argparse.Namespace) -> None:
    """Print RL dispatch plan without executing."""
    config_name = getattr(args, "config", None) or "crypto10_daily"
    preset = RL_PRESET_CONFIGS.get(config_name, RL_PRESET_CONFIGS["crypto10_daily"])
    data_train = getattr(args, "data_train", None) or preset["data_train"]
    data_val = getattr(args, "data_val", None) or preset["data_val"]
    gpu_type = args.gpu_type or DEFAULT_RL_GPU_TYPE
    time_budget = getattr(args, "time_budget", DEFAULT_RL_TIME_BUDGET)
    max_trials = getattr(args, "max_trials", DEFAULT_RL_MAX_TRIALS)
    budget_limit = getattr(args, "budget_limit", DEFAULT_BUDGET_LIMIT)
    total_secs = max_trials * time_budget

    print("[dry-run] RL Dispatch Plan (stock-prediction)")
    print(f"  Config:      {config_name}  -- {preset.get('note', '')}")
    print(f"  Data train:  {data_train}")
    print(f"  Data val:    {data_val}")
    print(f"  GPU:         {gpu_type}")
    print(f"  Trials:      {max_trials} x {time_budget}s")
    print()
    est_cost = _print_cost_line(gpu_type, total_secs)
    print()
    _enforce_budget(est_cost, budget_limit, raise_on_exceed=False)
    print("  Steps:")
    print(f"    1. Provision RunPod pod ({gpu_type})")
    print(f"    2. rsync {REPO} -> /workspace/stock-prediction")
    print("    3. Bootstrap: pip install uv, uv pip install -e .")
    print(f"    4. Upload data ({data_train}, {data_val})")
    print(f"    5. pufferlib_market.autoresearch_rl --max-trials {max_trials} --time-budget {time_budget}")
    print("    6. Download leaderboard CSV + checkpoints")
    print("    7. Terminate pod")


def dispatch_rl(args: argparse.Namespace) -> int:
    """Dispatch RL autoresearch via scripts/dispatch_rl_training.py."""
    config_name = getattr(args, "config", None) or "crypto10_daily"
    preset = RL_PRESET_CONFIGS.get(config_name, RL_PRESET_CONFIGS["crypto10_daily"])

    data_train = getattr(args, "data_train", None) or preset["data_train"]
    data_val = getattr(args, "data_val", None) or preset["data_val"]
    gpu_type = args.gpu_type or DEFAULT_RL_GPU_TYPE
    time_budget = getattr(args, "time_budget", DEFAULT_RL_TIME_BUDGET)
    max_trials = getattr(args, "max_trials", DEFAULT_RL_MAX_TRIALS)
    budget_limit = getattr(args, "budget_limit", DEFAULT_BUDGET_LIMIT)
    descriptions = preset.get("descriptions", "")

    dispatch_argv = [
        "--data-train", data_train,
        "--data-val", data_val,
        "--gpu-type", gpu_type,
        "--time-budget", str(time_budget),
        "--max-trials", str(max_trials),
        "--budget-limit", str(budget_limit),
        "--force-remote",
    ]
    if descriptions:
        dispatch_argv += ["--descriptions", descriptions]

    dispatch_script = REPO / "scripts" / "dispatch_rl_training.py"
    if not dispatch_script.exists():
        print(f"[error] dispatch_rl_training.py not found at {dispatch_script}")
        return 1

    import importlib.util
    spec = importlib.util.spec_from_file_location("dispatch_rl_training", dispatch_script)
    if spec is None or spec.loader is None:
        print("[error] Could not load dispatch_rl_training module")
        return 1
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.main(dispatch_argv)


# ---------------------------------------------------------------------------
# Status command: show all active pods (both RL and LLM)
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    """Show all active RunPod pods with cost estimates."""
    try:
        client = RunPodClient()
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1

    try:
        pods = client.list_pods()
    except Exception as exc:
        print(f"[error] Could not list pods: {exc}")
        return 1

    if not pods:
        print("No active pods.")
        return 0

    print(f"{'Name':<35} {'ID':<20} {'Status'}")
    print("-" * 70)
    for pod in pods:
        print(f"{pod.name:<35} {pod.id:<20} {pod.status}")

    print(f"\nTotal pods: {len(pods)}")
    print()
    print("Tip: Use 'cost' subcommand for detailed cost breakdown.")
    return 0


# ---------------------------------------------------------------------------
# Cost command: cumulative cost breakdown by project
# ---------------------------------------------------------------------------


def cmd_cost(args: argparse.Namespace) -> int:
    """Show cost summary by project type from pod names."""
    try:
        client = RunPodClient()
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1

    try:
        pods = client.list_pods()
    except Exception as exc:
        print(f"[error] Could not list pods: {exc}")
        return 1

    if not pods:
        print("No active pods — cost is $0.00.")
        return 0

    rl_pods = [p for p in pods if p.name.startswith(("rl-dispatch-", "rl-"))]
    llm_pods = [p for p in pods if p.name.startswith(("llm-dispatch-", "pgolf-", "llm-"))]
    other_pods = [p for p in pods if p not in rl_pods and p not in llm_pods]

    print("Cost breakdown by project (per-hour rates for running pods):")
    print()

    total_rate = 0.0
    for label, pod_list in [("RL (stock-prediction)", rl_pods),
                             ("LLM (cutellm)", llm_pods),
                             ("Other", other_pods)]:
        if not pod_list:
            continue
        pod_rate = sum(_resolve_hourly_rate(p.gpu_type)[1] for p in pod_list)
        total_rate += pod_rate
        print(f"  {label}: {len(pod_list)} pod(s), ${pod_rate:.2f}/hr")
        for pod in pod_list:
            rate = _resolve_hourly_rate(pod.gpu_type)[1]
            gpu = pod.gpu_type or "unknown GPU"
            print(f"    {pod.name:<35} {pod.id:<20} {pod.status:<10} ${rate:.2f}/hr  {gpu}")
        print()

    print(f"Total active rate: ${total_rate:.2f}/hr")
    return 0


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu-type", default="",
                   choices=[""] + sorted(TRAINING_GPU_TYPES.keys()),
                   metavar="TYPE",
                   help=f"GPU type alias: {', '.join(sorted(TRAINING_GPU_TYPES.keys()))}")
    p.add_argument("--budget-limit", type=float, default=DEFAULT_BUDGET_LIMIT,
                   help="Max USD to spend (0 = no limit)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without executing")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified dispatch for RL trading (stock-prediction) and LLM training (cutellm).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- rl subcommand ---
    rl_p = sub.add_parser(
        "rl",
        help="Dispatch RL autoresearch experiment (stock-prediction)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rl_p.add_argument(
        "--config", default="crypto10_daily",
        choices=list(RL_PRESET_CONFIGS.keys()),
        help="Preset RL config name",
    )
    rl_p.add_argument("--data-train", default="", metavar="PATH",
                      help="Override training data .bin (default: from config)")
    rl_p.add_argument("--data-val", default="", metavar="PATH",
                      help="Override validation data .bin (default: from config)")
    rl_p.add_argument("--time-budget", type=int, default=DEFAULT_RL_TIME_BUDGET,
                      help="Seconds per trial")
    rl_p.add_argument("--max-trials", type=int, default=DEFAULT_RL_MAX_TRIALS,
                      help="Max number of trials")
    _add_shared_args(rl_p)

    # --- llm subcommand ---
    llm_p = sub.add_parser(
        "llm",
        help="Dispatch LLM parameter-golf experiment (cutellm)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    llm_p.add_argument(
        "--experiment", default="",
        metavar="NAME",
        help="Experiment name from default_experiments.json (default: first entry)",
    )
    llm_p.add_argument("--gpu-count", type=int, default=DEFAULT_LLM_GPU_COUNT,
                       help="Number of GPUs (use 8 for full H100 competition run)")
    llm_p.add_argument("--max-wallclock", type=int, default=DEFAULT_LLM_MAX_WALLCLOCK,
                       help="Max wallclock seconds for training")
    _add_shared_args(llm_p)

    # --- status subcommand ---
    sub.add_parser("status", help="Show all active pods (both RL and LLM)")

    # --- cost subcommand ---
    sub.add_parser("cost", help="Show cumulative cost breakdown by project")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "status":
        return cmd_status(args)

    if args.command == "cost":
        return cmd_cost(args)

    if args.command == "rl":
        if args.dry_run:
            _rl_dry_run(args)
            return 0
        return dispatch_rl(args)

    if args.command == "llm":
        if args.dry_run:
            _llm_dry_run(args)
            return 0
        return dispatch_llm(args)

    print(f"[error] Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
