"""Tests for inference_server_daily.py — all RunPod API calls and subprocess calls are mocked."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import sys
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import inference_server_daily as isd
from inference_server_daily import (
    _ssh_host,
    _ssh_opts,
    export_inference_data,
    parse_args,
    provision_pod,
    write_manifest,
    main,
    GPU_FALLBACKS,
    REMOTE_DIR,
)
from src.runpod_client import Pod, PodConfig, RunPodClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_running_pod() -> Pod:
    return Pod(
        id="pod-test-123",
        name="inference-20260322",
        status="RUNNING",
        gpu_type="NVIDIA A40",
        ssh_host="10.0.0.1",
        ssh_port=22,
        public_ip="10.0.0.1",
    )


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

def test_parse_args_defaults():
    args = parse_args([])
    assert args.gpu_type == "a40"
    assert args.tts_k == 64
    assert args.horizon == 20
    assert args.dry_run is False


def test_parse_args_dry_run():
    args = parse_args(["--dry-run"])
    assert args.dry_run is True


def test_parse_args_symbols():
    args = parse_args(["--symbols", "AAPL,MSFT,NVDA"])
    assert args.symbols == "AAPL,MSFT,NVDA"


def test_parse_args_gpu_type_a40():
    args = parse_args(["--gpu-type", "a40"])
    assert args.gpu_type == "a40"


def test_parse_args_gpu_type_h100():
    args = parse_args(["--gpu-type", "h100"])
    assert args.gpu_type == "h100"


def test_parse_args_custom_tts_k():
    args = parse_args(["--tts-k", "128"])
    assert args.tts_k == 128


def test_parse_args_invalid_gpu_type_raises(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--gpu-type", "invalid-gpu-xyz"])


# ---------------------------------------------------------------------------
# _ssh_host / _ssh_opts
# ---------------------------------------------------------------------------

def test_ssh_host_format():
    pod = _make_running_pod()
    assert _ssh_host(pod) == "root@10.0.0.1"


def test_ssh_opts_contains_stricthostkeychecking():
    opts = _ssh_opts()
    assert "-o" in opts
    assert "StrictHostKeyChecking=no" in opts


def test_ssh_opts_contains_connect_timeout():
    opts = _ssh_opts()
    assert "ConnectTimeout=30" in opts


# ---------------------------------------------------------------------------
# export_inference_data — dry_run mode
# ---------------------------------------------------------------------------

def test_export_inference_data_dry_run_does_not_call_subprocess(tmp_path):
    out = tmp_path / "out.bin"
    # Should log and return without calling subprocess
    with patch("subprocess.run") as mock_run:
        export_inference_data(["AAPL", "MSFT"], out, dry_run=True)
    mock_run.assert_not_called()


def test_export_inference_data_dry_run_does_not_create_file(tmp_path):
    out = tmp_path / "out.bin"
    export_inference_data(["AAPL"], out, dry_run=True)
    assert not out.exists()


# ---------------------------------------------------------------------------
# provision_pod — dry_run mode
# ---------------------------------------------------------------------------

def test_provision_pod_dry_run_returns_none():
    mock_client = MagicMock(spec=RunPodClient)
    pod = provision_pod(mock_client, "a40", "inference-test", dry_run=True)
    assert pod is None
    mock_client.create_pod.assert_not_called()


def test_provision_pod_dry_run_no_api_calls():
    mock_client = MagicMock(spec=RunPodClient)
    provision_pod(mock_client, "h100", "pod-name", dry_run=True)
    mock_client.create_pod.assert_not_called()
    mock_client.wait_for_pod.assert_not_called()


# ---------------------------------------------------------------------------
# provision_pod — real mode with mock client
# ---------------------------------------------------------------------------

def test_provision_pod_creates_and_waits():
    mock_client = MagicMock(spec=RunPodClient)
    created = Pod(id="pod-abc", name="p", status="CREATED")
    running = _make_running_pod()
    mock_client.create_pod.return_value = created
    mock_client.wait_for_pod.return_value = running

    result = provision_pod(mock_client, "a40", "inference-test")

    assert result is running
    mock_client.create_pod.assert_called_once()
    mock_client.wait_for_pod.assert_called_once_with("pod-abc", timeout=isd.POD_PROVISION_TIMEOUT)


def test_provision_pod_falls_back_on_failure():
    """When primary GPU fails, provision_pod retries with fallback GPU type."""
    mock_client = MagicMock(spec=RunPodClient)
    running = _make_running_pod()

    # First create_pod call raises; second succeeds
    created_fallback = Pod(id="pod-fallback", name="p", status="CREATED")
    mock_client.create_pod.side_effect = [RuntimeError("no a40"), created_fallback]
    mock_client.wait_for_pod.return_value = running

    result = provision_pod(mock_client, "a40", "inference-test")
    assert result is running
    # Two create_pod calls: one with a40, one with l40 (fallback)
    assert mock_client.create_pod.call_count == 2


def test_provision_pod_raises_when_all_fail():
    mock_client = MagicMock(spec=RunPodClient)
    mock_client.create_pod.side_effect = RuntimeError("no GPU available")

    with pytest.raises(RuntimeError, match="Pod provisioning failed"):
        provision_pod(mock_client, "a40", "inference-test")


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------

def test_write_manifest_creates_json(tmp_path):
    manifest_path = write_manifest(
        tmp_path,
        today="20260322",
        pod_id="pod-123",
        pod_ip="10.0.0.1",
        checkpoint_used="/path/to/ckpt.pt",
        symbols=["AAPL", "MSFT"],
        tts_k=64,
        gpu_type="a40",
        cost_estimate=0.172,
        decisions_path="/tmp/decisions.json",
    )
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["pod_id"] == "pod-123"
    assert data["pod_ip"] == "10.0.0.1"
    assert data["symbols"] == ["AAPL", "MSFT"]
    assert data["tts_k"] == 64
    assert data["gpu_type"] == "a40"
    assert data["decisions_path"] == "/tmp/decisions.json"


def test_write_manifest_file_name_contains_date(tmp_path):
    manifest_path = write_manifest(
        tmp_path,
        today="20260101",
        pod_id="p",
        pod_ip="",
        checkpoint_used="",
        symbols=[],
        tts_k=1,
        gpu_type="a40",
        cost_estimate=0.0,
        decisions_path="",
    )
    assert "20260101" in manifest_path.name


def test_write_manifest_creates_parent_dir(tmp_path):
    deep_dir = tmp_path / "a" / "b" / "c"
    assert not deep_dir.exists()
    write_manifest(
        deep_dir,
        today="20260322",
        pod_id="p",
        pod_ip="",
        checkpoint_used="",
        symbols=[],
        tts_k=1,
        gpu_type="a40",
        cost_estimate=0.0,
        decisions_path="",
    )
    assert deep_dir.exists()


# ---------------------------------------------------------------------------
# GPU_FALLBACKS
# ---------------------------------------------------------------------------

def test_gpu_fallbacks_a40_falls_back_to_l40():
    assert "a40" in GPU_FALLBACKS
    assert GPU_FALLBACKS["a40"] == "l40"


def test_gpu_fallbacks_chain_eventually_reaches_a100():
    """Fallback chain from a40 should reach a100 within a few hops."""
    visited = set()
    current = "a40"
    while current in GPU_FALLBACKS:
        visited.add(current)
        current = GPU_FALLBACKS[current]
        if current in visited:
            break
    # Should end up at a100 or a100-sxm
    assert "a100" in current


# ---------------------------------------------------------------------------
# main() — dry-run integration
# ---------------------------------------------------------------------------

def test_main_dry_run_returns_zero(tmp_path, monkeypatch):
    """Dry-run must return 0 without RUNPOD_API_KEY set."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    rc = main([
        "--symbols", "AAPL,MSFT",
        "--dry-run",
        "--manifest-dir", str(tmp_path / "manifests"),
    ])
    assert rc == 0


def test_main_dry_run_no_pod_created(tmp_path, monkeypatch):
    """Dry-run must not call RunPodClient."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    with patch("inference_server_daily.RunPodClient") as mock_cls:
        main([
            "--symbols", "AAPL",
            "--dry-run",
            "--manifest-dir", str(tmp_path / "manifests"),
        ])
    mock_cls.assert_not_called()


def test_main_missing_api_key_returns_nonzero(tmp_path, monkeypatch):
    """Without RUNPOD_API_KEY and not dry-run, main must return non-zero."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    rc = main([
        "--symbols", "AAPL",
        "--manifest-dir", str(tmp_path / "manifests"),
    ])
    assert rc != 0


def test_main_missing_checkpoint_returns_nonzero(tmp_path, monkeypatch):
    """Nonexistent checkpoint path should return nonzero."""
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    rc = main([
        "--symbols", "AAPL",
        "--checkpoint", str(tmp_path / "nonexistent.pt"),
        "--manifest-dir", str(tmp_path / "manifests"),
    ])
    assert rc != 0


# ---------------------------------------------------------------------------
# bootstrap_runpod_rl --inference-only mode
# ---------------------------------------------------------------------------

def test_bootstrap_inference_only_skips_r2(monkeypatch, capsys):
    """--inference-only must not call get_r2_client."""
    from pufferlib_market import bootstrap_runpod_rl

    with patch.object(bootstrap_runpod_rl, "get_r2_client") as mock_r2, \
         patch.object(bootstrap_runpod_rl, "detect_h100", return_value=False):
        bootstrap_runpod_rl.main.__wrapped__ if hasattr(bootstrap_runpod_rl.main, "__wrapped__") else None
        # Invoke via parse_args + direct call
        old_argv = sys.argv[:]
        sys.argv = ["bootstrap_runpod_rl", "--run-id", "test-run", "--inference-only"]
        try:
            bootstrap_runpod_rl.main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

    mock_r2.assert_not_called()


def test_bootstrap_inference_only_env_var(monkeypatch, capsys):
    """INFERENCE_ONLY=1 env var triggers --inference-only behaviour."""
    from pufferlib_market import bootstrap_runpod_rl

    monkeypatch.setenv("INFERENCE_ONLY", "1")
    with patch.object(bootstrap_runpod_rl, "get_r2_client") as mock_r2, \
         patch.object(bootstrap_runpod_rl, "detect_h100", return_value=False):
        old_argv = sys.argv[:]
        sys.argv = ["bootstrap_runpod_rl", "--run-id", "test-run"]
        try:
            bootstrap_runpod_rl.main()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

    mock_r2.assert_not_called()
    out = capsys.readouterr().out
    assert "inference-only" in out.lower() or "Inference environment" in out
