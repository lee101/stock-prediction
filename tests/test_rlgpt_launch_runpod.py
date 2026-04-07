from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from RLgpt.config import DailyPlanDataConfig, PlannerConfig, SimulatorConfig, TrainingConfig
from RLgpt.config import DEFAULT_RLGPT_DATA_ROOT, DEFAULT_RLGPT_FORECAST_CACHE_ROOT, default_forecast_horizons_csv
from RLgpt.launch_runpod import (
    DEFAULT_RUNPOD_EPOCHS,
    build_launch_manifest,
    build_training_command,
    create_pod_with_fallbacks,
    main,
    parse_args,
)
from src.runpod_client import Pod


def test_build_training_command_includes_core_rlgpt_args():
    config = TrainingConfig(
        data=DailyPlanDataConfig(
            symbols=("BTCUSD", "ETHUSD"),
            validation_days=12,
            cache_only=True,
        ),
        planner=PlannerConfig(hidden_dim=96, depth=2, heads=4, dropout=0.05),
        simulator=SimulatorConfig(carry_inventory=True),
        epochs=7,
        batch_size=8,
        learning_rate=1e-4,
        weight_decay=5e-4,
        run_name="btc_eth_daily",
    )

    command = build_training_command(config, remote_output_root="/workspace/stock/experiments/RLgpt")

    assert command.startswith("python -m RLgpt.train")
    assert "--symbols BTCUSD,ETHUSD" in command
    assert "--validation-days 12" in command
    assert "--cache-only" in command
    assert "--carry-inventory" in command
    assert "--output-root /workspace/stock/experiments/RLgpt" in command


def test_build_launch_manifest_uses_real_pod_coordinates():
    config = TrainingConfig(
        data=DailyPlanDataConfig(symbols=("BTCUSD",), cache_only=True),
        planner=PlannerConfig(),
        simulator=SimulatorConfig(),
        run_name="btc_daily_test",
    )
    pod = Pod(
        id="pod-123",
        name="rlgpt-btc_daily_test",
        status="RUNNING",
        ssh_host="1.2.3.4",
        ssh_port=11022,
        public_ip="1.2.3.4",
    )

    manifest = build_launch_manifest(
        config=config,
        pod_name="rlgpt-btc_daily_test",
        gpu_type="4090",
        gpu_count=1,
        volume_size=120,
        container_disk=40,
        repo_root=Path("/home/lee/code/stock"),
        remote_repo_root="/workspace/stock",
        pod=pod,
    )

    assert manifest["pod"]["id"] == "pod-123"
    assert "root@1.2.3.4" in manifest["ssh_command"]
    assert "-p 11022" in manifest["rsync_command"]
    assert manifest["training_command"].startswith("python -m RLgpt.train")
    assert "analysis/remote_logs/btc_daily_test.log" in manifest["remote_launch_command"]


def test_create_pod_with_fallbacks_retries_after_capacity_error():
    client = MagicMock()
    client.create_pod_with_fallback.return_value = Pod(
        id="pod-2",
        name="rlgpt-run",
        status="CREATED",
        gpu_type="NVIDIA RTX PRO 4500 Ada Generation",
    )
    client.wait_for_pod.return_value = Pod(
        id="pod-2",
        name="rlgpt-run",
        status="RUNNING",
        gpu_type="NVIDIA RTX PRO 4500 Ada Generation",
        ssh_host="2.3.4.5",
        ssh_port=12022,
    )

    pod, gpu_type = create_pod_with_fallbacks(
        client=client,
        pod_name="rlgpt-run",
        gpu_type="4090",
        gpu_fallbacks="4500-ada,l4",
        gpu_count=1,
        volume_size=120,
        container_disk=40,
        wait_timeout=60,
    )

    assert pod.id == "pod-2"
    assert gpu_type == "NVIDIA RTX PRO 4500 Ada Generation"
    client.create_pod_with_fallback.assert_called_once()
    create_config, gpu_preferences = client.create_pod_with_fallback.call_args.args
    assert create_config.gpu_type == "NVIDIA GeForce RTX 4090"
    assert gpu_preferences == [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX PRO 4500 Ada Generation",
        "NVIDIA L4",
    ]
    client.wait_for_pod.assert_called_once_with("pod-2", timeout=60)


def test_parse_args_defaults_match_shared_rlgpt_defaults():
    args = parse_args(["--symbols", "BTCUSD", "--run-name", "demo"])

    assert args.data_root == str(DEFAULT_RLGPT_DATA_ROOT)
    assert args.forecast_cache_root == str(DEFAULT_RLGPT_FORECAST_CACHE_ROOT)
    assert args.forecast_horizons == default_forecast_horizons_csv()
    assert args.epochs == DEFAULT_RUNPOD_EPOCHS


def test_main_dry_run_prints_manifest_without_creating_pod_or_writing_file(
    monkeypatch,
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "launch_manifest.json"

    monkeypatch.setattr(
        "RLgpt.launch_runpod.RunPodClient",
        lambda: (_ for _ in ()).throw(AssertionError("RunPodClient should not be created in --dry-run mode")),
    )

    main(
        [
            "--symbols",
            "BTCUSD,ETHUSD",
            "--run-name",
            "demo",
            "--gpu-type",
            "4090",
            "--gpu-fallbacks",
            "l4",
            "--output-manifest",
            str(manifest_path),
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["pod_name"] == "rlgpt-demo"
    assert payload["gpu_type"] == "NVIDIA GeForce RTX 4090"
    assert payload["gpu_preferences"] == ["NVIDIA GeForce RTX 4090", "NVIDIA L4"]
    assert "pod" not in payload
    assert payload["training_command"].startswith("python -m RLgpt.train")
    assert manifest_path.exists() is False


def test_main_dry_run_text_prints_summary_to_stderr(monkeypatch, tmp_path, capsys):
    manifest_path = tmp_path / "launch_manifest.json"

    monkeypatch.setattr(
        "RLgpt.launch_runpod.RunPodClient",
        lambda: (_ for _ in ()).throw(AssertionError("RunPodClient should not be created in --dry-run mode")),
    )
    monkeypatch.setattr(
        "RLgpt.launch_runpod.build_gpu_fallback_types",
        lambda primary, fallbacks=None: ["NVIDIA GeForce RTX 4090"],
    )

    main(
        [
            "--symbols",
            "BTCUSD",
            "--run-name",
            "demo",
            "--create-pod",
            "--output-manifest",
            str(manifest_path),
            "--dry-run",
            "--dry-run-text",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["gpu_preferences"] == ["NVIDIA GeForce RTX 4090"]
    assert "RLgpt RunPod Launch Plan" in captured.err
    assert "Status: dry run" in captured.err
    assert f"Manifest path: {manifest_path}" in captured.err
    assert "GPU preferences: NVIDIA GeForce RTX 4090" in captured.err
    assert "rerun without --dry-run to provision the pod and write the manifest." in captured.err
    assert manifest_path.exists() is False
