from __future__ import annotations

import json
import shlex
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from RLgpt.config import DailyPlanDataConfig, PlannerConfig, SimulatorConfig, TrainingConfig
from RLgpt.config import DEFAULT_RLGPT_DATA_ROOT, DEFAULT_RLGPT_FORECAST_CACHE_ROOT, default_forecast_horizons_csv
from RLgpt.launch_runpod import (
    DEFAULT_REMOTE_REPO_ROOT,
    DEFAULT_REMOTE_LOG_DIR,
    DEFAULT_RUNPOD_EPOCHS,
    build_launch_manifest,
    build_training_command,
    main,
    parse_args,
)
from src.runpod_client import DEFAULT_POD_READY_TIMEOUT_SECONDS, Pod


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


def test_build_training_command_shell_quotes_dynamic_values():
    config = TrainingConfig(
        data=DailyPlanDataConfig(
            symbols=("BTCUSD",),
            data_root=Path("/tmp/data root"),
            forecast_cache_root=Path("/tmp/cache root"),
        ),
        planner=PlannerConfig(),
        simulator=SimulatorConfig(),
        run_name="demo; touch /tmp/pwned",
    )

    command = build_training_command(config, remote_output_root="/tmp/out root")

    assert f"--data-root {shlex.quote('/tmp/data root')}" in command
    assert f"--forecast-cache-root {shlex.quote('/tmp/cache root')}" in command
    assert f"--output-root {shlex.quote('/tmp/out root')}" in command
    assert f"--run-name {shlex.quote('demo; touch /tmp/pwned')}" in command
    assert "--run-name demo; touch /tmp/pwned" not in command


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
    assert f"{DEFAULT_REMOTE_LOG_DIR}/btc_daily_test.log" in manifest["remote_launch_command"]


def test_build_launch_manifest_shell_quotes_remote_launch_paths():
    config = TrainingConfig(
        data=DailyPlanDataConfig(symbols=("BTCUSD",), cache_only=True),
        planner=PlannerConfig(),
        simulator=SimulatorConfig(),
        run_name="demo; touch /tmp/pwned",
    )

    manifest = build_launch_manifest(
        config=config,
        pod_name="rlgpt-demo",
        gpu_type="4090",
        gpu_count=1,
        volume_size=120,
        container_disk=40,
        repo_root=Path("/home/lee/code/stock"),
        remote_repo_root="/workspace/stock dir",
    )

    quoted_root = shlex.quote("/workspace/stock dir")
    quoted_run_name = shlex.quote("demo; touch /tmp/pwned")
    assert f"cd {quoted_root}" in manifest["remote_launch_command"]
    assert f"PYTHONPATH={quoted_root}:$PYTHONPATH" in manifest["remote_launch_command"]
    assert f"{DEFAULT_REMOTE_LOG_DIR}/{quoted_run_name}.log" in manifest["remote_launch_command"]
    assert "demo; touch /tmp/pwned.log" not in manifest["remote_launch_command"]


def test_main_create_pod_uses_shared_ready_helper(monkeypatch, tmp_path, capsys):
    manifest_path = tmp_path / "launch_manifest.json"

    client = MagicMock()
    client.create_ready_pod_with_fallback.return_value = Pod(
        id="pod-2",
        name="rlgpt-run",
        status="RUNNING",
        gpu_type="NVIDIA RTX PRO 4500 Ada Generation",
        ssh_host="2.3.4.5",
        ssh_port=12022,
    )
    monkeypatch.setattr("RLgpt.launch_runpod.RunPodClient", lambda: client)

    main(
        [
            "--symbols",
            "BTCUSD",
            "--run-name",
            "demo",
            "--gpu-type",
            "4090",
            "--gpu-fallbacks",
            "4500-ada,l4",
            "--create-pod",
            "--wait-timeout",
            "60",
            "--output-manifest",
            str(manifest_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["pod_id"] == "pod-2"
    client.create_ready_pod_with_fallback.assert_called_once()
    create_config, gpu_preferences = client.create_ready_pod_with_fallback.call_args.args
    assert create_config.gpu_type == "NVIDIA GeForce RTX 4090"
    assert gpu_preferences == (
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX PRO 4500 Ada Generation",
        "NVIDIA L4",
    )
    assert client.create_ready_pod_with_fallback.call_args.kwargs["timeout"] == 60


def test_main_terminates_pod_when_manifest_write_fails(monkeypatch, tmp_path):
    manifest_path = tmp_path / "launch_manifest.json"

    client = MagicMock()
    client.create_ready_pod_with_fallback.return_value = Pod(
        id="pod-2",
        name="rlgpt-run",
        status="RUNNING",
        gpu_type="NVIDIA RTX PRO 4500 Ada Generation",
        ssh_host="2.3.4.5",
        ssh_port=12022,
    )
    monkeypatch.setattr("RLgpt.launch_runpod.RunPodClient", lambda: client)

    real_write_text = Path.write_text

    def _fake_write_text(self: Path, *args, **kwargs):
        if self == manifest_path:
            raise PermissionError("disk full")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _fake_write_text)

    with pytest.raises(PermissionError, match="disk full"):
        main(
            [
                "--symbols",
                "BTCUSD",
                "--run-name",
                "demo",
                "--create-pod",
                "--output-manifest",
                str(manifest_path),
            ]
        )

    client.terminate_pod.assert_called_once_with("pod-2")


def test_main_adds_note_when_manifest_write_cleanup_fails(monkeypatch, tmp_path):
    manifest_path = tmp_path / "launch_manifest.json"

    client = MagicMock()
    client.create_ready_pod_with_fallback.return_value = Pod(
        id="pod-2",
        name="rlgpt-run",
        status="RUNNING",
        gpu_type="NVIDIA RTX PRO 4500 Ada Generation",
        ssh_host="2.3.4.5",
        ssh_port=12022,
    )
    client.terminate_pod.side_effect = RuntimeError("terminate failed")
    monkeypatch.setattr("RLgpt.launch_runpod.RunPodClient", lambda: client)

    real_write_text = Path.write_text

    def _fake_write_text(self: Path, *args, **kwargs):
        if self == manifest_path:
            raise PermissionError("disk full")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _fake_write_text)

    with pytest.raises(PermissionError, match="disk full") as excinfo:
        main(
            [
                "--symbols",
                "BTCUSD",
                "--run-name",
                "demo",
                "--create-pod",
                "--output-manifest",
                str(manifest_path),
            ]
        )

    assert excinfo.value.__notes__ is not None
    assert any(
        "failed to terminate pod pod-2 after local launch failure before writing" in note
        and "terminate failed" in note
        for note in excinfo.value.__notes__
    )


def test_parse_args_defaults_match_shared_rlgpt_defaults():
    args = parse_args(["--symbols", "BTCUSD", "--run-name", "demo"])

    assert args.data_root == str(DEFAULT_RLGPT_DATA_ROOT)
    assert args.forecast_cache_root == str(DEFAULT_RLGPT_FORECAST_CACHE_ROOT)
    assert args.forecast_horizons == default_forecast_horizons_csv()
    assert args.epochs == DEFAULT_RUNPOD_EPOCHS
    assert args.remote_repo_root == DEFAULT_REMOTE_REPO_ROOT
    assert args.wait_timeout == DEFAULT_POD_READY_TIMEOUT_SECONDS


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
        "RLgpt.launch_runpod.resolve_gpu_preferences",
        lambda primary, fallbacks=None: ("NVIDIA GeForce RTX 4090",),
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
