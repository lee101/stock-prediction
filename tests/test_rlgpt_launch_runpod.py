from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from RLgpt.config import DailyPlanDataConfig, PlannerConfig, SimulatorConfig, TrainingConfig
from RLgpt.launch_runpod import build_launch_manifest, build_training_command, create_pod_with_fallbacks
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
    first_error = RuntimeError("RunPod capacity error: does not have the resources to deploy your pod")
    client.create_pod.side_effect = [
        first_error,
        Pod(id="pod-2", name="rlgpt-run", status="CREATED"),
    ]
    client.wait_for_pod.return_value = Pod(
        id="pod-2",
        name="rlgpt-run",
        status="RUNNING",
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
    assert gpu_type != "4090"
    assert client.create_pod.call_count == 2
