from __future__ import annotations

import argparse
import json
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
    Pod,
    PodConfig,
    RunPodClient,
    build_gpu_fallback_types,
    is_capacity_error,
    parse_gpu_fallback_types,
)


def build_training_command(config: TrainingConfig, *, remote_output_root: str) -> str:
    horizons = ",".join(str(value) for value in config.data.forecast_horizons)
    symbols = ",".join(config.data.symbols)
    parts = [
        "python -m RLgpt.train",
        f"--symbols {symbols}",
        f"--data-root {config.data.data_root}",
        f"--forecast-cache-root {config.data.forecast_cache_root}",
        f"--forecast-horizons {horizons}",
        f"--validation-days {config.data.validation_days}",
        f"--epochs {config.epochs}",
        f"--batch-size {config.batch_size}",
        f"--learning-rate {config.learning_rate}",
        f"--weight-decay {config.weight_decay}",
        f"--hidden-dim {config.planner.hidden_dim}",
        f"--depth {config.planner.depth}",
        f"--heads {config.planner.heads}",
        f"--dropout {config.planner.dropout}",
        f"--shared-unit-budget {config.simulator.shared_unit_budget}",
        f"--max-units-per-asset {config.simulator.max_units_per_asset}",
        f"--initial-cash {config.simulator.initial_cash}",
        f"--maker-fee-bps {config.simulator.maker_fee_bps}",
        f"--slippage-bps {config.simulator.slippage_bps}",
        f"--fill-buffer-bps {config.simulator.fill_buffer_bps}",
        f"--fill-temperature-bps {config.simulator.fill_temperature_bps}",
        f"--output-root {remote_output_root}",
        f"--run-name {config.run_name}",
        f"--seed {config.seed}",
        f"--min-history-hours {config.data.min_history_hours}",
        f"--sequence-length {config.data.sequence_length}",
        f"--max-feature-lookback-hours {config.data.max_feature_lookback_hours}",
        f"--min-bars-per-day {config.data.min_bars_per_day}",
    ]
    if config.max_train_days is not None:
        parts.append(f"--max-train-days {config.max_train_days}")
    if config.max_val_days is not None:
        parts.append(f"--max-val-days {config.max_val_days}")
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
    pod: Pod | None = None,
) -> dict[str, Any]:
    remote_output_root = f"{remote_repo_root}/experiments/RLgpt"
    training_command = build_training_command(config, remote_output_root=remote_output_root)
    remote_setup = [
        f"cd {remote_repo_root}",
        "command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh)",
        "export PATH=$HOME/.local/bin:$PATH",
        "python -m venv .venv313",
        "source .venv313/bin/activate",
        "uv pip install -e .",
        f"export PYTHONPATH={remote_repo_root}:$PYTHONPATH",
        "mkdir -p analysis/remote_logs",
    ]
    remote_launch = (
        f"cd {remote_repo_root} && "
        "source .venv313/bin/activate && "
        f"export PYTHONPATH={remote_repo_root}:$PYTHONPATH && "
        f"nohup {training_command} > analysis/remote_logs/{config.run_name}.log 2>&1 & "
        f"echo $! > analysis/remote_logs/{config.run_name}.pid"
    )

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pod_name": pod_name,
        "gpu_type": gpu_type,
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


def create_pod_with_fallbacks(
    *,
    client: RunPodClient,
    pod_name: str,
    gpu_type: str,
    gpu_fallbacks: str | None,
    gpu_count: int,
    volume_size: int,
    container_disk: int,
    wait_timeout: int,
) -> tuple[Pod, str]:
    gpu_candidates = build_gpu_fallback_types(
        gpu_type,
        parse_gpu_fallback_types(gpu_fallbacks),
    )
    last_error: BaseException | None = None
    for candidate in gpu_candidates:
        try:
            pod_cfg = PodConfig(
                name=pod_name,
                gpu_type=candidate,
                gpu_count=gpu_count,
                volume_size=volume_size,
                container_disk=container_disk,
            )
            pod = client.create_pod(pod_cfg)
            pod = client.wait_for_pod(pod.id, timeout=wait_timeout)
            return pod, candidate
        except Exception as exc:  # pragma: no cover - exercised in real launch path
            last_error = exc
            if not is_capacity_error(exc):
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to create a RunPod pod.")


DEFAULT_RUNPOD_EPOCHS = 12


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
    parser.add_argument("--remote-repo-root", default="/workspace/stock")
    parser.add_argument("--create-pod", action="store_true")
    parser.add_argument("--wait-timeout", type=int, default=900)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--carry-inventory", action="store_true")
    parser.add_argument("--max-train-days", type=int)
    parser.add_argument("--max-val-days", type=int)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = _build_training_config(args)
    repo_root = Path(__file__).resolve().parent.parent
    pod_name = f"rlgpt-{args.run_name}"
    pod = None
    selected_gpu_type = args.gpu_type
    if args.create_pod:
        client = RunPodClient()
        pod, selected_gpu_type = create_pod_with_fallbacks(
            client=client,
            pod_name=pod_name,
            gpu_type=args.gpu_type,
            gpu_fallbacks=args.gpu_fallbacks,
            gpu_count=args.gpu_count,
            volume_size=args.volume_size,
            container_disk=args.container_disk,
            wait_timeout=args.wait_timeout,
        )

    manifest = build_launch_manifest(
        config=config,
        pod_name=pod_name,
        gpu_type=selected_gpu_type,
        gpu_count=args.gpu_count,
        volume_size=args.volume_size,
        container_disk=args.container_disk,
        repo_root=repo_root,
        remote_repo_root=args.remote_repo_root,
        pod=pod,
    )
    manifest_path = Path(args.output_manifest) if args.output_manifest else Path("analysis") / "remote_runs" / args.run_name / "launch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), "pod_id": manifest.get("pod", {}).get("id", "")}, indent=2))


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
