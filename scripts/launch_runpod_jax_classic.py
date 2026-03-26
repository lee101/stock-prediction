#!/usr/bin/env python3
"""Provision a RunPod pod, train the JAX classic stock model, sync artifacts, terminate the pod."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import requests


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.runpod_client import TRAINING_DOCKER_IMAGE, RunPodClient, resolve_gpu_type


RUNPOD_REST_URL = "https://rest.runpod.io/v1"
REMOTE_DIR = "/workspace/stock-prediction"
REMOTE_ENV = ".venv311jax"
DEFAULT_SYMBOLS = "NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE,ITUB,BTG,ABEV"
DEFAULT_PRELOAD = "unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt"
DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 4090"
DEFAULT_DOCKER_IMAGE = TRAINING_DOCKER_IMAGE
DOCKER_VALIDATE_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
SYNC_CODE_PATHS = (
    "binanceneural",
    "src",
    "hyperparamstore",
    "wandboard.py",
    "differentiable_loss_utils.py",
    "pyproject.toml",
    "uv.lock",
    "unified_hourly_experiment/train_jax_classic.py",
)
BOOTSTRAP_PACKAGES = (
    "numpy",
    "setuptools",
    "wheel",
    "torch",
    "pandas",
    "pyarrow",
    "loguru",
    "exchange-calendars",
    "tensorboard",
    "wandb",
    "jax[cuda12]==0.9.2",
    "flax>=0.12.6",
    "optax>=0.2.8",
)


def parse_symbols(text: str) -> list[str]:
    return [token.strip().upper() for token in str(text).split(",") if token.strip()]


def parse_horizons(text: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in str(text).split(",") if token.strip())
    if not values:
        raise ValueError("At least one forecast horizon is required")
    return values


def _run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        check=check,
        text=True,
        capture_output=capture,
    )


def _ssh_base(key_path: Path, ssh_port: int, ssh_host: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(ssh_port),
        f"root@{ssh_host}",
    ]


def run_ssh(
    key_path: Path,
    ssh_port: int,
    ssh_host: str,
    remote_cmd: str,
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _run(_ssh_base(key_path, ssh_port, ssh_host) + [remote_cmd], capture=capture, check=check)


def remote_path_exists(
    key_path: Path,
    ssh_port: int,
    ssh_host: str,
    remote_path: str,
) -> bool:
    result = run_ssh(
        key_path,
        ssh_port,
        ssh_host,
        f"test -e {shlex.quote(remote_path)}",
        capture=False,
        check=False,
    )
    return result.returncode == 0


def run_rsync(
    sources: list[str],
    destination: str,
    *,
    key_path: Path,
    ssh_port: int,
    delete: bool = False,
) -> None:
    cmd = [
        "rsync",
        "-rltDz",
        "-e",
        f"ssh -i {shlex.quote(str(key_path))} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {ssh_port}",
    ]
    if delete:
        cmd.append("--delete")
    cmd.extend(sources)
    cmd.append(destination)
    _run(cmd, cwd=REPO)


def rest_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def create_pod(api_key: str, *, gpu_type: str, name: str, image: str) -> str:
    client = RunPodClient(api_key=api_key)
    gpu_type_id = client.find_gpu_type_id(resolve_gpu_type(gpu_type))
    body = {
        "name": name,
        "computeType": "GPU",
        "gpuTypeIds": [gpu_type_id],
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "imageName": image,
        "interruptible": False,
        "containerDiskInGb": 40,
        "volumeInGb": 120,
        "volumeMountPath": "/workspace",
        "supportPublicIp": True,
        "ports": ["22/tcp", "8888/http"],
    }
    response = requests.post(f"{RUNPOD_REST_URL}/pods", headers=rest_headers(api_key), json=body, timeout=60)
    response.raise_for_status()
    return response.json()["id"]


def get_pod(api_key: str, pod_id: str) -> dict[str, Any]:
    response = requests.get(f"{RUNPOD_REST_URL}/pods/{pod_id}", headers=rest_headers(api_key), timeout=60)
    response.raise_for_status()
    return response.json()


def delete_pod(api_key: str, pod_id: str) -> None:
    requests.delete(f"{RUNPOD_REST_URL}/pods/{pod_id}", headers=rest_headers(api_key), timeout=60).raise_for_status()


def wait_for_public_ssh(api_key: str, pod_id: str, *, timeout_sec: int = 900) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        pod = get_pod(api_key, pod_id)
        port_mappings = pod.get("portMappings") or {}
        public_ip = pod.get("publicIp") or ""
        ssh_port = int(port_mappings.get("22") or 0)
        if pod.get("desiredStatus") == "RUNNING" and public_ip and ssh_port:
            return {
                "pod_id": pod_id,
                "public_ip": public_ip,
                "ssh_port": ssh_port,
                "cost_per_hr": pod.get("costPerHr"),
                "gpu_type": pod.get("machineType") or pod.get("gpuType") or "",
            }
        time.sleep(10)
    raise TimeoutError(f"Pod {pod_id} did not expose SSH within {timeout_sec}s")


def build_sync_manifest(
    *,
    symbols: list[str],
    horizons: tuple[int, ...],
    preload_path: str,
) -> dict[str, list[str]]:
    data_files = [f"trainingdatahourly/stocks/{symbol}.csv" for symbol in symbols]
    cache_files = [f"unified_hourly_experiment/forecast_cache/h{horizon}/{symbol}.parquet" for horizon in horizons for symbol in symbols]
    checkpoint_files = [
        preload_path,
        str(Path(preload_path).with_name("config.json")),
        str(Path(preload_path).with_name("training_meta.json")),
    ]
    checkpoint_files = [path for path in checkpoint_files if (REPO / path).exists()]
    code_paths = [path for path in SYNC_CODE_PATHS if (REPO / path).exists()]
    return {
        "code_paths": code_paths,
        "data_files": [path for path in data_files if (REPO / path).exists()],
        "cache_files": [path for path in cache_files if (REPO / path).exists()],
        "checkpoint_files": checkpoint_files,
    }


def build_bootstrap_command(*, remote_run_dir: str) -> str:
    package_args = " ".join(shlex.quote(pkg) for pkg in BOOTSTRAP_PACKAGES)
    lines = [
        "set -euo pipefail",
        f"cd {shlex.quote(REMOTE_DIR)}",
        "apt-get update >/dev/null",
        "apt-get install -y --no-install-recommends rsync python3.11 python3.11-venv python3-pip >/dev/null",
        "if ! command -v uv >/dev/null 2>&1; then python3.11 -m pip install -q uv; fi",
        f"mkdir -p {shlex.quote(remote_run_dir)}/env",
        f"uv venv {shlex.quote(REMOTE_ENV)} --python python3.11",
        f"source {shlex.quote(REMOTE_ENV)}/bin/activate",
        f"uv pip install {package_args}",
        "python -V > " + shlex.quote(f"{remote_run_dir}/env/python_version.txt"),
        "uv pip freeze > " + shlex.quote(f"{remote_run_dir}/env/uv_pip_freeze.txt"),
        "nvidia-smi > " + shlex.quote(f"{remote_run_dir}/env/nvidia_smi.txt"),
        "python - <<'PY' > " + shlex.quote(f"{remote_run_dir}/env/torch_cuda.txt"),
        "import torch",
        "print(torch.__version__)",
        "print(torch.cuda.is_available())",
        "print(torch.cuda.device_count())",
        "PY",
        "python - <<'PY' > " + shlex.quote(f"{remote_run_dir}/env/jax_devices.txt"),
        "import jax",
        "print(jax.__version__)",
        "print(jax.devices())",
        "PY",
    ]
    return "\n".join(lines)


def build_train_command(
    *,
    run_name: str,
    remote_run_dir: str,
    remote_log_path: str,
    symbols: list[str],
    horizons: tuple[int, ...],
    preload_path: str,
    validation_days: int,
    epochs: int,
    batch_size: int,
    sequence_length: int,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_group: str | None,
    wandb_tags: str,
    wandb_notes: str | None,
    wandb_mode: str,
    dry_train_steps: int | None,
) -> str:
    cmd = [
        "python",
        "unified_hourly_experiment/train_jax_classic.py",
        "--symbols",
        ",".join(symbols),
        "--forecast-horizons",
        ",".join(str(horizon) for horizon in horizons),
        "--run-name",
        run_name,
        "--checkpoint-root",
        "unified_hourly_experiment/checkpoints",
        "--log-dir",
        f"tensorboard_logs/binanceneural/{run_name}",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--sequence-length",
        str(sequence_length),
        "--validation-days",
        str(validation_days),
        "--preload",
        preload_path,
        "--market-order-entry",
        "--wandb-mode",
        wandb_mode,
    ]
    if wandb_project:
        cmd.extend(["--wandb-project", wandb_project])
    if wandb_entity:
        cmd.extend(["--wandb-entity", wandb_entity])
    if wandb_group:
        cmd.extend(["--wandb-group", wandb_group])
    if wandb_tags:
        cmd.extend(["--wandb-tags", wandb_tags])
    if wandb_notes:
        cmd.extend(["--wandb-notes", wandb_notes])
    if dry_train_steps is not None:
        cmd.extend(["--dry-train-steps", str(dry_train_steps)])

    quoted_cmd = " ".join(shlex.quote(part) for part in cmd)
    lines = [
        "set -euo pipefail",
        f"cd {shlex.quote(REMOTE_DIR)}",
        f"source {shlex.quote(REMOTE_ENV)}/bin/activate",
        'export PYTHONPATH="$PWD:${PYTHONPATH:-}"',
        f'export WANDB_DIR="{remote_run_dir}/wandb"',
        f'mkdir -p "{remote_run_dir}" "{remote_run_dir}/wandb"',
    ]
    if os.environ.get("WANDB_API_KEY"):
        lines.append(f'export WANDB_API_KEY="{os.environ["WANDB_API_KEY"]}"')
    lines.extend(
        [
            f'echo running > "{remote_run_dir}/status.txt"',
            f'{quoted_cmd} > {shlex.quote(remote_log_path)} 2>&1',
            'rc=$?',
            f'echo "$rc" > "{remote_run_dir}/exit_code.txt"',
            f'if [ "$rc" -eq 0 ]; then echo completed > "{remote_run_dir}/status.txt"; else echo failed > "{remote_run_dir}/status.txt"; fi',
            'exit "$rc"',
        ]
    )
    return "\n".join(lines)


def launch_remote_script(
    *,
    key_path: Path,
    ssh_host: str,
    ssh_port: int,
    remote_script_path: str,
    script_body: str,
    remote_log_path: str,
    remote_pid_path: str,
) -> str:
    bootstrap = "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(str(Path(remote_script_path).parent))}",
            f"mkdir -p {shlex.quote(str(Path(remote_log_path).parent))}",
            f"cat > {shlex.quote(remote_script_path)} <<'__RUNPOD_JAX_SCRIPT__'",
            script_body.rstrip("\n"),
            "__RUNPOD_JAX_SCRIPT__",
            f"chmod +x {shlex.quote(remote_script_path)}",
            f"nohup bash {shlex.quote(remote_script_path)} > {shlex.quote(remote_log_path)} 2>&1 &",
            f"echo $! > {shlex.quote(remote_pid_path)}",
            f"cat {shlex.quote(remote_pid_path)}",
        ]
    )
    result = run_ssh(key_path, ssh_port, ssh_host, f"bash -lc {shlex.quote(bootstrap)}", capture=True)
    return (result.stdout or "").strip().splitlines()[-1]


def write_remote_script(
    *,
    key_path: Path,
    ssh_host: str,
    ssh_port: int,
    remote_script_path: str,
    script_body: str,
) -> None:
    bootstrap = "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(str(Path(remote_script_path).parent))}",
            f"cat > {shlex.quote(remote_script_path)} <<'__RUNPOD_JAX_SCRIPT__'",
            script_body.rstrip("\n"),
            "__RUNPOD_JAX_SCRIPT__",
            f"chmod +x {shlex.quote(remote_script_path)}",
        ]
    )
    run_ssh(key_path, ssh_port, ssh_host, f"bash -lc {shlex.quote(bootstrap)}")


def poll_remote_pid(
    *,
    key_path: Path,
    ssh_host: str,
    ssh_port: int,
    remote_pid_path: str,
    remote_status_path: str,
    poll_interval_sec: int,
) -> str:
    while True:
        result = run_ssh(
            key_path,
            ssh_port,
            ssh_host,
            (
                f'PID=$(cat {shlex.quote(remote_pid_path)} 2>/dev/null || echo "") ; '
                'if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then echo running; '
                f'else cat {shlex.quote(remote_status_path)} 2>/dev/null || echo done; fi'
            ),
            capture=True,
            check=False,
        )
        status = (result.stdout or "unknown").strip().splitlines()[-1]
        if status not in {"running", "done", "unknown"}:
            return status
        if status == "done":
            return status
        time.sleep(poll_interval_sec)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def docker_validate(args: argparse.Namespace) -> None:
    symbols = parse_symbols(args.symbols)
    validation_symbols = symbols[: min(3, len(symbols))]
    run_name = f"{args.run_name}_docker"
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{REPO}:/workspace",
        "-w",
        "/workspace",
        DOCKER_VALIDATE_IMAGE,
        "bash",
        "-lc",
        " && ".join(
            [
                "python -m pip install -q uv",
                "uv venv /tmp/jax-smoke --python python3.11",
                "source /tmp/jax-smoke/bin/activate",
                f"uv pip install {' '.join(shlex.quote(pkg) for pkg in BOOTSTRAP_PACKAGES)}",
                'export PYTHONPATH="$PWD:${PYTHONPATH:-}"',
                (
                    "python unified_hourly_experiment/train_jax_classic.py "
                    f"--symbols {shlex.quote(','.join(validation_symbols))} "
                    f"--run-name {shlex.quote(run_name)} "
                    "--epochs 1 --batch-size 2 "
                    f"--sequence-length {args.sequence_length} "
                    "--validation-days 7 "
                    f"--preload {shlex.quote(args.preload)} "
                    "--dry-train-steps 1 --market-order-entry"
                ),
            ]
        ),
    ]
    _run(cmd, cwd=REPO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a detached JAX classic stock training run on RunPod.")
    parser.add_argument("--run-name", default=f"jax_classic_runpod_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--forecast-horizons", default="1")
    parser.add_argument("--preload", default=DEFAULT_PRELOAD)
    parser.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    parser.add_argument("--image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--validation-days", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--dry-train-steps", type=int, default=None)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default="runpod,jax,alpaca")
    parser.add_argument("--wandb-notes", default=None)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--key-path", type=Path, default=Path.home() / ".ssh" / "id_ed25519")
    parser.add_argument("--poll-interval-sec", type=int, default=60)
    parser.add_argument("--output-root", type=Path, default=Path("analysis") / "remote_runs")
    parser.add_argument("--docker-validate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--keep-pod", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = parse_symbols(args.symbols)
    horizons = parse_horizons(args.forecast_horizons)
    manifest = build_sync_manifest(symbols=symbols, horizons=horizons, preload_path=args.preload)
    local_run_dir = args.output_root / args.run_name
    local_run_dir.mkdir(parents=True, exist_ok=True)

    if args.docker_validate:
        docker_validate(args)

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_name": args.run_name,
        "symbols": symbols,
        "forecast_horizons": list(horizons),
        "preload": args.preload,
        "gpu_type": args.gpu_type,
        "image": args.image,
        "validation_days": args.validation_days,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "dry_train_steps": args.dry_train_steps,
        "manifest": manifest,
    }
    write_manifest(local_run_dir / "launch_manifest.json", payload)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY is required")

    pod_id = create_pod(api_key, gpu_type=args.gpu_type, name=args.run_name, image=args.image)
    pod = None
    start = time.monotonic()
    should_delete_pod = False
    try:
        pod = wait_for_public_ssh(api_key, pod_id)
        should_delete_pod = not args.keep_pod and not args.detach
        write_manifest(
            local_run_dir / "pod_manifest.json",
            {
                "pod_id": pod["pod_id"],
                "public_ip": pod["public_ip"],
                "ssh_port": pod["ssh_port"],
                "cost_per_hr": pod["cost_per_hr"],
                "gpu_type": pod["gpu_type"],
            },
        )

        run_ssh(args.key_path, pod["ssh_port"], pod["public_ip"], f"mkdir -p {shlex.quote(REMOTE_DIR)}")

        code_sources = [str(REPO / rel) for rel in manifest["code_paths"]]
        run_rsync(code_sources, f"root@{pod['public_ip']}:{REMOTE_DIR}/", key_path=args.key_path, ssh_port=pod["ssh_port"])

        if manifest["data_files"]:
            run_ssh(args.key_path, pod["ssh_port"], pod["public_ip"], f"mkdir -p {shlex.quote(f'{REMOTE_DIR}/trainingdatahourly/stocks')}")
            run_rsync(
                [str(REPO / rel) for rel in manifest["data_files"]],
                f"root@{pod['public_ip']}:{REMOTE_DIR}/trainingdatahourly/stocks/",
                key_path=args.key_path,
                ssh_port=pod["ssh_port"],
            )

        for horizon in horizons:
            cache_root = f"{REMOTE_DIR}/unified_hourly_experiment/forecast_cache/h{horizon}"
            run_ssh(args.key_path, pod["ssh_port"], pod["public_ip"], f"mkdir -p {shlex.quote(cache_root)}")
        if manifest["cache_files"]:
            run_rsync(
                [str(REPO / rel) for rel in manifest["cache_files"]],
                f"root@{pod['public_ip']}:{REMOTE_DIR}/",
                key_path=args.key_path,
                ssh_port=pod["ssh_port"],
            )

        if manifest["checkpoint_files"]:
            run_ssh(
                args.key_path,
                pod["ssh_port"],
                pod["public_ip"],
                f"mkdir -p {shlex.quote(f'{REMOTE_DIR}/{Path(args.preload).parent}')}",
            )
            run_rsync(
                [str(REPO / rel) for rel in manifest["checkpoint_files"]],
                f"root@{pod['public_ip']}:{REMOTE_DIR}/{Path(args.preload).parent}/",
                key_path=args.key_path,
                ssh_port=pod["ssh_port"],
            )

        remote_run_dir = f"{REMOTE_DIR}/analysis/remote_runs/{args.run_name}"
        remote_bootstrap_log = f"{remote_run_dir}/bootstrap.log"
        remote_bootstrap_script = f"{remote_run_dir}/bootstrap.sh"
        remote_train_script = f"{remote_run_dir}/train.sh"
        remote_train_driver_log = f"{remote_run_dir}/train_driver.log"
        remote_train_log = f"{remote_run_dir}/train.log"
        remote_train_pid = f"{remote_run_dir}/train.pid"
        remote_status_path = f"{remote_run_dir}/status.txt"
        remote_bootstrap_status = f"{remote_run_dir}/bootstrap.status.txt"

        bootstrap_cmd = build_bootstrap_command(remote_run_dir=remote_run_dir)
        write_remote_script(
            key_path=args.key_path,
            ssh_host=pod["public_ip"],
            ssh_port=pod["ssh_port"],
            remote_script_path=remote_bootstrap_script,
            script_body=bootstrap_cmd,
        )
        run_ssh(
            args.key_path,
            pod["ssh_port"],
            pod["public_ip"],
            (
                "bash -lc "
                + shlex.quote(
                    "\n".join(
                        [
                            "set -euo pipefail",
                            f"bash {shlex.quote(remote_bootstrap_script)} > {shlex.quote(remote_bootstrap_log)} 2>&1",
                            f"echo completed > {shlex.quote(remote_bootstrap_status)}",
                        ]
                    )
                )
            ),
        )

        train_cmd = build_train_command(
            run_name=args.run_name,
            remote_run_dir=remote_run_dir,
            remote_log_path=remote_train_log,
            symbols=symbols,
            horizons=horizons,
            preload_path=args.preload,
            validation_days=args.validation_days,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=args.wandb_group or args.run_name,
            wandb_tags=args.wandb_tags,
            wandb_notes=args.wandb_notes,
            wandb_mode=args.wandb_mode,
            dry_train_steps=args.dry_train_steps,
        )
        launch_remote_script(
            key_path=args.key_path,
            ssh_host=pod["public_ip"],
            ssh_port=pod["ssh_port"],
            remote_script_path=remote_train_script,
            script_body=train_cmd,
            remote_log_path=remote_train_driver_log,
            remote_pid_path=remote_train_pid,
        )

        if not args.detach:
            final_status = poll_remote_pid(
                key_path=args.key_path,
                ssh_host=pod["public_ip"],
                ssh_port=pod["ssh_port"],
                remote_pid_path=remote_train_pid,
                remote_status_path=remote_status_path,
                poll_interval_sec=args.poll_interval_sec,
            )
            payload["final_status"] = final_status

        local_remote_dir = local_run_dir / "remote"
        local_remote_dir.mkdir(parents=True, exist_ok=True)
        if remote_path_exists(args.key_path, pod["ssh_port"], pod["public_ip"], remote_run_dir):
            run_rsync(
                [f"root@{pod['public_ip']}:{remote_run_dir}/"],
                f"{local_remote_dir}/",
                key_path=args.key_path,
                ssh_port=pod["ssh_port"],
            )
        remote_checkpoint_dir = f"{REMOTE_DIR}/unified_hourly_experiment/checkpoints/{args.run_name}"
        if remote_path_exists(args.key_path, pod["ssh_port"], pod["public_ip"], remote_checkpoint_dir):
            run_rsync(
                [f"root@{pod['public_ip']}:{remote_checkpoint_dir}/"],
                f"{local_run_dir}/checkpoints/",
                key_path=args.key_path,
                ssh_port=pod["ssh_port"],
            )
        if args.detach and not args.keep_pod:
            payload["pod_retained"] = True
            payload["pod_retained_reason"] = "detach requested; pod left running for remote training completion"
        payload["elapsed_hours"] = round((time.monotonic() - start) / 3600.0, 4)
        write_manifest(local_run_dir / "completion_manifest.json", payload)
        return 0
    finally:
        if pod is not None and should_delete_pod:
            delete_pod(api_key, pod["pod_id"])


if __name__ == "__main__":
    raise SystemExit(main())
