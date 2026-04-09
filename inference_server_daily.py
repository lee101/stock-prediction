#!/usr/bin/env python3
"""
Ephemeral daily inference server.
Provisions a RunPod A40 pod, runs inference with test-time search, returns decisions, terminates pod.

Usage:
  python inference_server_daily.py \
    --checkpoint pufferlib_market/checkpoints/stocks_deployment_candidate.pt \
    --symbols AAPL,MSFT,NVDA,GOOG,META,TSLA,SPY,QQQ,PLTR,JPM,AMZN,AMD \
    --gpu-type a40 \
    --tts-k 64 \
    --dry-run
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
    GPU_ALIASES,
    Pod,
    PodConfig,
    RunPodClient,
    TRAINING_DOCKER_IMAGE,
    resolve_gpu_type,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REMOTE_DIR = "/workspace/stock-prediction"
POD_PROVISION_TIMEOUT = 300  # 5 min to become RUNNING

GPU_FALLBACKS: dict[str, str] = {
    "a40": "l40",
    "l40": "l40s",
    "l40s": "a100",
    "a100": "a100-sxm",
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

log = logging.getLogger("inference_server_daily")


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_path)),
    ])


# ---------------------------------------------------------------------------
# SSH / rsync helpers
# ---------------------------------------------------------------------------

def _ssh_host(pod: Pod) -> str:
    return f"root@{pod.ssh_host}"


def _ssh_opts() -> list[str]:
    return [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=30",
        "-p", "22",
    ]


def _run_ssh(pod: Pod, cmd: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command on the pod via SSH."""
    ssh_cmd = ["ssh"] + _ssh_opts() + [_ssh_host(pod), cmd]
    log.info("SSH [%s]: %s", pod.ssh_host, cmd[:120])
    return subprocess.run(
        ssh_cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def _run_rsync(src: str, dst: str, *, excludes: Sequence[str] = ()) -> subprocess.CompletedProcess:
    """rsync src to dst with optional excludes."""
    # Build ssh -e string from the same opts list used by _run_ssh / _scp_*.
    ssh_e = "ssh " + " ".join(_ssh_opts())
    cmd = ["rsync", "-az", "--progress", "-e", ssh_e]
    for ex in excludes:
        cmd += ["--exclude", ex]
    cmd += [src, dst]
    log.info("RSYNC %s -> %s", src, dst)
    return subprocess.run(cmd, check=True, text=True)


def _scp_to_pod(local: str, pod: Pod, remote: str) -> None:
    cmd = ["scp"] + _ssh_opts() + [local, f"{_ssh_host(pod)}:{remote}"]
    log.info("SCP %s -> %s:%s", local, pod.ssh_host, remote)
    subprocess.run(cmd, check=True)


def _scp_from_pod(pod: Pod, remote: str, local: str) -> None:
    cmd = ["scp"] + _ssh_opts() + [f"{_ssh_host(pod)}:{remote}", local]
    log.info("SCP %s:%s -> %s", pod.ssh_host, remote, local)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Data export
# ---------------------------------------------------------------------------

def export_inference_data(
    symbols: list[str],
    output_path: Path,
    *,
    dry_run: bool = False,
) -> None:
    """Export latest market data for the given symbols to a .bin file.

    Uses export_data_alpaca_daily.py pattern: reads from trainingdatahourly/stocks/
    and trainingdata/train/ and writes MKTD v2 binary.
    """
    if dry_run:
        log.info("[dry-run] Would export inference data for %d symbols -> %s", len(symbols), output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sym_arg = ",".join(symbols)
    cmd = [
        sys.executable, "-u",
        str(REPO / "export_data_alpaca_daily.py"),
        "--symbols", sym_arg,
        "--output-train", str(output_path),
        "--output-val", str(output_path).replace(".bin", "_val.bin"),
    ]
    log.info("Exporting inference data: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO))
    log.info("Exported inference data to %s (%d bytes)", output_path, output_path.stat().st_size)


# ---------------------------------------------------------------------------
# Pod provisioning
# ---------------------------------------------------------------------------

def provision_pod(
    client: RunPodClient,
    gpu_type: str,
    pod_name: str,
    *,
    dry_run: bool = False,
) -> Optional[Pod]:
    """Provision a RunPod pod, retrying once with a fallback GPU type on failure."""
    if dry_run:
        log.info("[dry-run] Would provision pod name=%s gpu=%s", pod_name, gpu_type)
        return None

    attempts = [(gpu_type, resolve_gpu_type(gpu_type))]
    fallback = GPU_FALLBACKS.get(gpu_type)
    if fallback:
        attempts.append((fallback, resolve_gpu_type(fallback)))

    last_exc: Optional[Exception] = None
    for alias, full_name in attempts:
        try:
            log.info("Provisioning pod with GPU: %s (%s)", alias, full_name)
            config = PodConfig(
                name=pod_name,
                gpu_type=full_name,
                image=TRAINING_DOCKER_IMAGE,
                container_disk=20,
                volume_size=20,
                env_vars={"INFERENCE_ONLY": "1"},
            )
            pod = client.create_pod(config)
            log.info("Pod created: id=%s, waiting for RUNNING...", pod.id)
            pod = client.wait_for_pod(pod.id, timeout=POD_PROVISION_TIMEOUT)
            log.info("Pod RUNNING: %s:%d", pod.ssh_host, pod.ssh_port)
            return pod
        except Exception as exc:
            log.warning("Provision attempt with %s failed: %s", alias, exc)
            last_exc = exc

    raise RuntimeError(f"Pod provisioning failed after all attempts. Last error: {last_exc}")


# ---------------------------------------------------------------------------
# Remote setup
# ---------------------------------------------------------------------------

def rsync_code_to_pod(pod: Pod, checkpoint_path: Path, data_path: Path) -> None:
    """Rsync repo, checkpoint and data to the pod."""
    remote_root = f"{_ssh_host(pod)}:{REMOTE_DIR}/"

    # Rsync the main codebase (exclude heavy data dirs and git)
    _run_rsync(
        str(REPO) + "/",
        remote_root,
        excludes=[
            ".git",
            "pufferlib_market/data/",
            "trainingdata/",
            "trainingdatahourly/",
            "__pycache__",
            "*.pyc",
            ".venv",
            ".venv313",
            ".venv312",
            "chronos-forecasting/",
            "modded-nanogpt/",
        ],
    )

    # Upload checkpoint separately (may be outside repo or large)
    remote_ckpt = f"{REMOTE_DIR}/checkpoint.pt"
    _scp_to_pod(str(checkpoint_path), pod, remote_ckpt)

    # Upload inference data binary
    remote_data = f"{REMOTE_DIR}/inference_data.bin"
    _scp_to_pod(str(data_path), pod, remote_data)

    log.info("Code, checkpoint and data synced to pod")


def setup_pod_env(pod: Pod) -> None:
    """Install uv, create venv, install deps, build C extension on the pod."""
    setup_cmd = " && ".join([
        f"cd {REMOTE_DIR}",
        "pip install uv -q",
        "uv venv .venv -q",
        "source .venv/bin/activate",
        "uv pip install -e . -q 2>&1 | tail -5",
        "cd pufferlib_market && python setup.py build_ext --inplace -q 2>&1 | tail -5 && cd ..",
    ])
    _run_ssh(pod, f"bash -lc {repr(setup_cmd)}")
    log.info("Pod environment ready")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference_on_pod(
    pod: Pod,
    *,
    tts_k: int,
    horizon: int,
    symbols: list[str],
) -> str:
    """Run inference with test-time search on the pod. Returns remote decisions JSON path."""
    remote_decisions = "/workspace/decisions.json"
    sym_arg = ",".join(symbols)

    inference_cmd = " && ".join([
        f"cd {REMOTE_DIR}",
        "source .venv/bin/activate",
        " ".join([
            "python -m pufferlib_market.inference_tts",
            "--checkpoint checkpoint.pt",
            "--data-path inference_data.bin",
            f"--tts-k {tts_k}",
            f"--horizon {horizon}",
            f"--symbols {sym_arg}",
            f"--output-json {remote_decisions}",
        ]),
    ])
    _run_ssh(pod, f"bash -lc {repr(inference_cmd)}")
    log.info("Inference complete on pod")
    return remote_decisions


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(
    manifest_dir: Path,
    *,
    today: str,
    pod_id: str,
    pod_ip: str,
    checkpoint_used: str,
    symbols: list[str],
    tts_k: int,
    gpu_type: str,
    cost_estimate: float,
    decisions_path: str,
) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"inference_{today}.json"
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "today": today,
        "pod_id": pod_id,
        "pod_ip": pod_ip,
        "checkpoint_used": checkpoint_used,
        "symbols": symbols,
        "tts_k": tts_k,
        "gpu_type": gpu_type,
        "cost_estimate_usd": round(cost_estimate, 4),
        "decisions_path": decisions_path,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("Manifest written: %s", manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

DEFAULT_SYMBOLS = "AAPL,MSFT,NVDA,GOOG,META,TSLA,SPY,QQQ,PLTR,JPM,AMZN,AMD"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ephemeral daily inference server: provision RunPod A40, "
            "rsync code+checkpoint+data, run TTS inference, pull decisions, terminate pod."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="pufferlib_market/checkpoints/stocks_deployment_candidate.pt",
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--symbols",
        default=DEFAULT_SYMBOLS,
        help="Comma-separated list of symbols to run inference on",
    )
    parser.add_argument(
        "--gpu-type",
        default="a40",
        choices=list(GPU_ALIASES.keys()),
        help="GPU type alias (default: a40, 48GB VRAM at $0.69/hr)",
    )
    parser.add_argument(
        "--tts-k",
        type=int,
        default=64,
        help="Number of K-rollouts for test-time search (default: 64)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Rollout horizon for TTS (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Local directory for decisions JSON output (default: current dir)",
    )
    parser.add_argument(
        "--manifest-dir",
        default="inference_manifests",
        help="Directory for manifest and log files (default: inference_manifests/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pipeline steps without provisioning a pod",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    today = time.strftime("%Y%m%d")
    manifest_dir = REPO / args.manifest_dir
    log_path = manifest_dir / f"inference_{today}.log"
    _setup_logging(log_path)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    checkpoint_path = REPO / args.checkpoint
    _local_tmp = REPO / "tmp"
    _local_tmp.mkdir(parents=True, exist_ok=True)
    data_path = _local_tmp / f"inference_data_{today}.bin"
    output_dir = REPO / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_local = str(output_dir / f"decisions_{today}.json")
    pod_name = f"inference-{today}"

    resolved_gpu = resolve_gpu_type(args.gpu_type)
    rate = HOURLY_RATES.get(resolved_gpu, 0.0)
    cost_est = rate * 15 / 60

    log.info("=== Ephemeral A40 Daily Inference Server ===")
    log.info("Today:      %s", today)
    log.info("Symbols:    %s", ", ".join(symbols))
    log.info("Checkpoint: %s", checkpoint_path)
    log.info("GPU:        %s (%s)", args.gpu_type, resolved_gpu)
    log.info("TTS-K:      %d, horizon=%d", args.tts_k, args.horizon)
    log.info("Cost est:   $%.3f (15 min @ $%.2f/hr)", cost_est, rate)

    if args.dry_run:
        log.info("")
        log.info("[dry-run] Step 1: export market data for %d symbols -> %s", len(symbols), data_path)
        log.info("[dry-run] Step 2: provision RunPod pod name=%s gpu=%s", pod_name, args.gpu_type)
        log.info("[dry-run] Step 3: rsync code + checkpoint + data to pod")
        log.info("[dry-run] Step 4: setup pod env (uv venv, pip install, build C ext)")
        log.info("[dry-run] Step 5: run inference --tts-k %d --horizon %d", args.tts_k, args.horizon)
        log.info("[dry-run] Step 6: scp decisions.json -> %s", decisions_local)
        log.info("[dry-run] Step 7: terminate pod")
        log.info("[dry-run] Step 8: write manifest to %s/inference_%s.json", manifest_dir, today)
        return 0

    # Validate RUNPOD_API_KEY present before doing any work
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        log.error("RUNPOD_API_KEY environment variable is not set")
        return 1

    if not checkpoint_path.exists():
        log.error("Checkpoint not found: %s", checkpoint_path)
        return 1

    client = RunPodClient(api_key=api_key)
    pod: Optional[Pod] = None

    try:
        # Step 1: Export fresh market data locally
        log.info("Step 1/7: Exporting market data...")
        export_inference_data(symbols, data_path)

        # Step 2: Provision pod
        log.info("Step 2/7: Provisioning pod...")
        pod = provision_pod(client, args.gpu_type, pod_name)
        if pod is None:
            log.error("Pod provisioning returned None unexpectedly")
            return 1

        # Step 3: Rsync code + checkpoint + data
        log.info("Step 3/7: Syncing code, checkpoint and data to pod...")
        try:
            rsync_code_to_pod(pod, checkpoint_path, data_path)
        except subprocess.CalledProcessError as exc:
            log.error("rsync failed: %s", exc)
            return 1

        # Step 4: Setup pod environment
        log.info("Step 4/7: Setting up pod environment...")
        setup_pod_env(pod)

        # Step 5: Run inference
        log.info("Step 5/7: Running inference (tts-k=%d, horizon=%d)...", args.tts_k, args.horizon)
        remote_decisions = run_inference_on_pod(
            pod,
            tts_k=args.tts_k,
            horizon=args.horizon,
            symbols=symbols,
        )

        # Step 6: Pull results
        log.info("Step 6/7: Pulling decisions from pod...")
        try:
            _scp_from_pod(pod, remote_decisions, decisions_local)
        except subprocess.CalledProcessError as exc:
            log.error("Failed to pull decisions: %s", exc)
            # Try to get pod logs for diagnostics
            try:
                _scp_from_pod(pod, "/workspace/inference.log", str(output_dir / f"inference_{today}_pod.log"))
            except Exception:
                pass
            return 1

        # Print decisions
        try:
            with open(decisions_local) as fh:
                decisions = json.load(fh)
            log.info("Decisions (%d symbols):", len(decisions))
            for sym, info in decisions.items():
                log.info("  %-8s  action=%-6s  confidence=%.2f  expected_return=%.4f",
                         sym,
                         info.get("action", "?"),
                         info.get("confidence", 0.0),
                         info.get("expected_return", 0.0))
        except Exception as exc:
            log.warning("Could not parse decisions JSON: %s", exc)

    except Exception as exc:
        log.error("Pipeline failed: %s", exc, exc_info=True)
        return 1

    finally:
        # Step 7: Always terminate pod
        if pod is not None:
            log.info("Step 7/7: Terminating pod %s...", pod.id)
            try:
                client.terminate_pod(pod.id)
            except Exception as exc:
                log.warning("Failed to terminate pod %s: %s", pod.id, exc)

    # Step 8: Write manifest
    write_manifest(
        manifest_dir,
        today=today,
        pod_id=pod.id if pod else "",
        pod_ip=pod.ssh_host if pod else "",
        checkpoint_used=str(checkpoint_path),
        symbols=symbols,
        tts_k=args.tts_k,
        gpu_type=args.gpu_type,
        cost_estimate=cost_est,
        decisions_path=decisions_local,
    )

    log.info("Done. Decisions: %s", decisions_local)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
