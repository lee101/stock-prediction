from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from RLgpt.config import (
    DailyPlanDataConfig,
    PlannerConfig,
    SimulatorConfig,
    TrainingConfig,
    DEFAULT_RLGPT_BATCH_SIZE,
    DEFAULT_RLGPT_DATA_ROOT,
    DEFAULT_RLGPT_DEPTH,
    DEFAULT_RLGPT_DROPOUT,
    DEFAULT_RLGPT_FILL_BUFFER_BPS,
    DEFAULT_RLGPT_FILL_TEMPERATURE_BPS,
    DEFAULT_RLGPT_FORECAST_CACHE_ROOT,
    DEFAULT_RLGPT_HEADS,
    DEFAULT_RLGPT_HIDDEN_DIM,
    DEFAULT_RLGPT_INITIAL_CASH,
    DEFAULT_RLGPT_LEARNING_RATE,
    DEFAULT_RLGPT_MAKER_FEE_BPS,
    DEFAULT_RLGPT_MAX_UNITS_PER_ASSET,
    DEFAULT_RLGPT_SHARED_UNIT_BUDGET,
    DEFAULT_RLGPT_SLIPPAGE_BPS,
    DEFAULT_RLGPT_VALIDATION_DAYS,
    DEFAULT_RLGPT_WEIGHT_DECAY,
    default_forecast_horizons_csv,
    normalize_symbol_list,
    parse_horizon_list,
)
from src.runpod_client import (
    DEFAULT_POD_READY_TIMEOUT_SECONDS,
    Pod,
    PodConfig,
    RunPodClient,
    resolve_gpu_preferences,
)

DEFAULT_REMOTE_REPO_ROOT = "/workspace/stock"
DEFAULT_REMOTE_OUTPUT_SUBDIR = "experiments/RLgpt"
DEFAULT_REMOTE_VENV = ".venv313"
DEFAULT_REMOTE_LOG_DIR = "analysis/remote_logs"


def build_training_command(config: TrainingConfig, *, remote_output_root: str) -> str:
    def _shell_arg(value: object) -> str:
        return shlex.quote(str(value))

    horizons = ",".join(str(value) for value in config.data.forecast_horizons)
    symbols = ",".join(config.data.symbols)
    parts = [
        "python",
        "-m",
        "RLgpt.train",
        "--symbols",
        _shell_arg(symbols),
        "--data-root",
        _shell_arg(config.data.data_root),
        "--forecast-cache-root",
        _shell_arg(config.data.forecast_cache_root),
        "--forecast-horizons",
        _shell_arg(horizons),
        "--validation-days",
        _shell_arg(config.data.validation_days),
        "--epochs",
        _shell_arg(config.epochs),
        "--batch-size",
        _shell_arg(config.batch_size),
        "--learning-rate",
        _shell_arg(config.learning_rate),
        "--weight-decay",
        _shell_arg(config.weight_decay),
        "--hidden-dim",
        _shell_arg(config.planner.hidden_dim),
        "--depth",
        _shell_arg(config.planner.depth),
        "--heads",
        _shell_arg(config.planner.heads),
        "--dropout",
        _shell_arg(config.planner.dropout),
        "--shared-unit-budget",
        _shell_arg(config.simulator.shared_unit_budget),
        "--max-units-per-asset",
        _shell_arg(config.simulator.max_units_per_asset),
        "--initial-cash",
        _shell_arg(config.simulator.initial_cash),
        "--maker-fee-bps",
        _shell_arg(config.simulator.maker_fee_bps),
        "--slippage-bps",
        _shell_arg(config.simulator.slippage_bps),
        "--fill-buffer-bps",
        _shell_arg(config.simulator.fill_buffer_bps),
        "--fill-temperature-bps",
        _shell_arg(config.simulator.fill_temperature_bps),
        "--output-root",
        _shell_arg(remote_output_root),
        "--run-name",
        _shell_arg(config.run_name),
        "--seed",
        _shell_arg(config.seed),
        "--min-history-hours",
        _shell_arg(config.data.min_history_hours),
        "--sequence-length",
        _shell_arg(config.data.sequence_length),
        "--max-feature-lookback-hours",
        _shell_arg(config.data.max_feature_lookback_hours),
        "--min-bars-per-day",
        _shell_arg(config.data.min_bars_per_day),
    ]
    if config.max_train_days is not None:
        parts.extend(["--max-train-days", _shell_arg(config.max_train_days)])
    if config.max_val_days is not None:
        parts.extend(["--max-val-days", _shell_arg(config.max_val_days)])
    if config.data.cache_only:
        parts.append("--cache-only")
    if config.simulator.carry_inventory:
        parts.append("--carry-inventory")
    return " ".join(parts)


def build_launch_manifest(
    *,
    config: TrainingConfig,
    pod_name: str,
    gpu_type: str,
    gpu_count: int,
    volume_size: int,
    container_disk: int,
    repo_root: Path,
    remote_repo_root: str,
    gpu_preferences: tuple[str, ...] | None = None,
    pod: Pod | None = None,
) -> dict[str, Any]:
    remote_output_root = f"{remote_repo_root}/{DEFAULT_REMOTE_OUTPUT_SUBDIR}"
    training_command = build_training_command(config, remote_output_root=remote_output_root)
    quoted_remote_repo_root = shlex.quote(remote_repo_root)
    quoted_run_name = shlex.quote(config.run_name)
    remote_setup = [
        f"cd {quoted_remote_repo_root}",
        "command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)",
        "export PATH=$HOME/.local/bin:$PATH",
        f"python -m venv {DEFAULT_REMOTE_VENV}",
        f"source {DEFAULT_REMOTE_VENV}/bin/activate",
        "uv pip install -e .",
        f"export PYTHONPATH={quoted_remote_repo_root}:$PYTHONPATH",
        f"mkdir -p {DEFAULT_REMOTE_LOG_DIR}",
    ]
    remote_launch = (
        f"cd {quoted_remote_repo_root} && "
        f"source {DEFAULT_REMOTE_VENV}/bin/activate && "
        f"export PYTHONPATH={quoted_remote_repo_root}:$PYTHONPATH && "
        f"nohup {training_command} > {DEFAULT_REMOTE_LOG_DIR}/{quoted_run_name}.log 2>&1 & "
        f"echo $! > {DEFAULT_REMOTE_LOG_DIR}/{quoted_run_name}.pid"
    )

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pod_name": pod_name,
        "gpu_type": gpu_type,
        "gpu_preferences": list(gpu_preferences or (gpu_type,)),
        "gpu_count": gpu_count,
        "volume_size_gb": volume_size,
        "container_disk_gb": container_disk,
        "repo_root": str(repo_root),
        "remote_repo_root": remote_repo_root,
        "training_command": training_command,
        "remote_setup_commands": remote_setup,
        "remote_launch_command": remote_launch,
        "config": _json_ready(asdict(config)),
        "rsync_command_template": (
            "rsync -az --delete --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' "
            f"-e \"ssh -o StrictHostKeyChecking=no -p <port>\" {repo_root}/ root@<host>:{remote_repo_root}/"
        ),
        "ssh_command_template": "ssh -o StrictHostKeyChecking=no -p <port> root@<host>",
    }
    if pod is not None:
        manifest["pod"] = {
            "id": pod.id,
            "status": pod.status,
            "ssh_host": pod.ssh_host,
            "ssh_port": pod.ssh_port,
            "public_ip": pod.public_ip,
        }
        manifest["rsync_command"] = (
            "rsync -az --delete --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' "
            f"-e \"ssh -o StrictHostKeyChecking=no -p {pod.ssh_port}\" {repo_root}/ root@{pod.ssh_host}:{remote_repo_root}/"
        )
        manifest["ssh_command"] = f"ssh -o StrictHostKeyChecking=no -p {pod.ssh_port} root@{pod.ssh_host}"
    return manifest


DEFAULT_RUNPOD_EPOCHS = 12


def _default_manifest_path(*, run_name: str, output_manifest: str) -> Path:
    if output_manifest:
        return Path(output_manifest)
    return Path("analysis") / "remote_runs" / run_name / "launch_manifest.json"


def _format_launch_summary(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    dry_run: bool,
    will_create_pod: bool,
) -> str:
    status = "dry run" if dry_run else "ready"
    gpu_preferences = manifest.get("gpu_preferences") or [manifest.get("gpu_type", "")]
    lines = [
        "RLgpt RunPod Launch Plan",
        f"Status: {status}",
        f"Pod name: {manifest.get('pod_name', '')}",
        f"Manifest path: {manifest_path}",
        f"GPU preferences: {', '.join(str(item) for item in gpu_preferences if item)}",
        f"Remote repo root: {manifest.get('remote_repo_root', '')}",
        f"Training command: {manifest.get('training_command', '')}",
    ]
    pod = manifest.get("pod")
    if isinstance(pod, dict) and pod:
        lines.append(
            "Resolved pod: "
            f"{pod.get('id', '')} @ {pod.get('ssh_host', '')}:{pod.get('ssh_port', '')}"
        )
    next_step = "write the manifest."
    if will_create_pod:
        next_step = "provision the pod and write the manifest."
    lines.append(f"Next step: rerun without --dry-run to {next_step}")
    return "\n".join(lines)


def _cleanup_failed_pod_launch(
    *,
    client: RunPodClient,
    pod: Pod,
    manifest_path: Path,
    exc: BaseException,
) -> None:
    try:
        client.terminate_pod(pod.id)
    except Exception as cleanup_exc:
        exc.add_note(
            "failed to terminate pod "
            f"{pod.id} after local launch failure before writing {manifest_path}: {cleanup_exc}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or launch a RunPod RLgpt training run.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--gpu-type", default="4090")
    parser.add_argument("--gpu-fallbacks")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--volume-size", type=int, default=120)
    parser.add_argument("--container-disk", type=int, default=40)
    parser.add_argument("--data-root", default=str(DEFAULT_RLGPT_DATA_ROOT))
    parser.add_argument("--forecast-cache-root", default=str(DEFAULT_RLGPT_FORECAST_CACHE_ROOT))
    parser.add_argument("--forecast-horizons", default=default_forecast_horizons_csv())
    parser.add_argument("--validation-days", type=int, default=DEFAULT_RLGPT_VALIDATION_DAYS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_RUNPOD_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_RLGPT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_RLGPT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_RLGPT_WEIGHT_DECAY)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_RLGPT_HIDDEN_DIM)
    parser.add_argument("--depth", type=int, default=DEFAULT_RLGPT_DEPTH)
    parser.add_argument("--heads", type=int, default=DEFAULT_RLGPT_HEADS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_RLGPT_DROPOUT)
    parser.add_argument("--shared-unit-budget", type=float, default=DEFAULT_RLGPT_SHARED_UNIT_BUDGET)
    parser.add_argument("--max-units-per-asset", type=float, default=DEFAULT_RLGPT_MAX_UNITS_PER_ASSET)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_RLGPT_INITIAL_CASH)
    parser.add_argument("--maker-fee-bps", type=float, default=DEFAULT_RLGPT_MAKER_FEE_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_RLGPT_SLIPPAGE_BPS)
    parser.add_argument("--fill-buffer-bps", type=float, default=DEFAULT_RLGPT_FILL_BUFFER_BPS)
    parser.add_argument("--fill-temperature-bps", type=float, default=DEFAULT_RLGPT_FILL_TEMPERATURE_BPS)
    parser.add_argument("--output-manifest", default="")
    parser.add_argument("--remote-repo-root", default=DEFAULT_REMOTE_REPO_ROOT)
    parser.add_argument("--create-pod", action="store_true")
    parser.add_argument("--wait-timeout", type=int, default=DEFAULT_POD_READY_TIMEOUT_SECONDS)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--carry-inventory", action="store_true")
    parser.add_argument("--max-train-days", type=int)
    parser.add_argument("--max-val-days", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved launch manifest and exit without creating a pod or writing files.",
    )
    parser.add_argument(
        "--dry-run-text",
        action="store_true",
        help="When used with --dry-run, also print a human-readable summary to stderr.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = _build_training_config(args)
    repo_root = Path(__file__).resolve().parent.parent
    pod_name = f"rlgpt-{args.run_name}"
    pod = None
    client: RunPodClient | None = None
    gpu_preferences = resolve_gpu_preferences(args.gpu_type, args.gpu_fallbacks)
    selected_gpu_type = gpu_preferences[0]
    if args.create_pod and not args.dry_run:
        client = RunPodClient()
        pod = client.create_ready_pod_with_fallback(
            PodConfig(
                name=pod_name,
                gpu_type=gpu_preferences[0],
                gpu_count=args.gpu_count,
                volume_size=args.volume_size,
                container_disk=args.container_disk,
            ),
            gpu_preferences,
            timeout=args.wait_timeout,
        )
        selected_gpu_type = pod.gpu_type

    manifest = build_launch_manifest(
        config=config,
        pod_name=pod_name,
        gpu_type=selected_gpu_type,
        gpu_count=args.gpu_count,
        volume_size=args.volume_size,
        container_disk=args.container_disk,
        repo_root=repo_root,
        remote_repo_root=args.remote_repo_root,
        gpu_preferences=gpu_preferences,
        pod=pod,
    )
    manifest_path = _default_manifest_path(
        run_name=args.run_name,
        output_manifest=args.output_manifest,
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        if args.dry_run_text:
            print(
                _format_launch_summary(
                    manifest=manifest,
                    manifest_path=manifest_path,
                    dry_run=True,
                    will_create_pod=bool(args.create_pod),
                ),
                file=sys.stderr,
            )
        return
    manifest_written = False
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifest_written = True
        print(
            json.dumps(
                {"manifest_path": str(manifest_path), "pod_id": manifest.get("pod", {}).get("id", "")},
                indent=2,
            )
        )
    except Exception as exc:
        if pod is not None and client is not None and not manifest_written:
            _cleanup_failed_pod_launch(
                client=client,
                pod=pod,
                manifest_path=manifest_path,
                exc=exc,
            )
        raise


def _build_training_config(args: argparse.Namespace) -> TrainingConfig:
    symbols = normalize_symbol_list(args.symbols.split(","))
    horizons = parse_horizon_list(args.forecast_horizons)
    data = DailyPlanDataConfig(
        symbols=symbols,
        data_root=Path(args.data_root),
        forecast_cache_root=Path(args.forecast_cache_root),
        forecast_horizons=horizons,
        validation_days=args.validation_days,
        cache_only=bool(args.cache_only),
    )
    planner = PlannerConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
    )
    simulator = SimulatorConfig(
        initial_cash=args.initial_cash,
        shared_unit_budget=args.shared_unit_budget,
        max_units_per_asset=args.max_units_per_asset,
        maker_fee_bps=args.maker_fee_bps,
        slippage_bps=args.slippage_bps,
        fill_buffer_bps=args.fill_buffer_bps,
        fill_temperature_bps=args.fill_temperature_bps,
        carry_inventory=bool(args.carry_inventory),
    )
    return TrainingConfig(
        data=data,
        planner=planner,
        simulator=simulator,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        run_name=args.run_name,
        max_train_days=args.max_train_days,
        max_val_days=args.max_val_days,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    main()
