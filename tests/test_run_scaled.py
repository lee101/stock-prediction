from __future__ import annotations

from types import SimpleNamespace

import pytest

import sharpnessadjustedproximalpolicy.run_scaled as run_scaled
import src.runpod_client as runpod_client


def test_run_remote_uses_shared_runpod_fallback_helper(monkeypatch) -> None:
    events: dict[str, object] = {}

    class _FakeClient:
        def create_pod_with_fallback(self, config, gpu_preferences):
            events["config"] = config
            events["gpu_preferences"] = list(gpu_preferences)
            return SimpleNamespace(id="pod-123", gpu_type="NVIDIA A40")

        def wait_for_pod(self, pod_id, timeout=0):
            events["wait"] = (pod_id, timeout)
            return SimpleNamespace(id=pod_id, gpu_type="NVIDIA A40", ssh_host="1.2.3.4", ssh_port=22022)

        def terminate_pod(self, pod_id):
            events["terminated"] = pod_id

    monkeypatch.setattr(runpod_client, "RunPodClient", lambda: _FakeClient())
    monkeypatch.setattr(
        runpod_client,
        "build_gpu_fallback_types",
        lambda primary: ["NVIDIA GeForce RTX 4090", "NVIDIA A40"],
    )
    monkeypatch.setattr(run_scaled.time, "strftime", lambda fmt: "04071234")
    monkeypatch.setattr(run_scaled, "_bootstrap_remote", lambda host, port, remote_dir: events.setdefault("bootstrapped", (host, port, remote_dir)))
    monkeypatch.setattr(run_scaled, "_run_remote_sweep", lambda host, port, remote_dir, args: events.setdefault("ran", (host, port, remote_dir, args.epochs)))
    monkeypatch.setattr(run_scaled, "_download_results", lambda host, port, remote_dir: events.setdefault("downloaded", (host, port, remote_dir)))

    args = SimpleNamespace(gpu="4090", epochs=15, compile=False, symbols=None, configs=None, keep_pod=False)

    run_scaled.run_remote(args)

    config = events["config"]
    assert config.name == "sap-sweep-04071234"
    assert config.gpu_type == "NVIDIA GeForce RTX 4090"
    assert events["gpu_preferences"] == ["NVIDIA GeForce RTX 4090", "NVIDIA A40"]
    assert events["wait"] == ("pod-123", run_scaled.RUNPOD_SSH_READY_TIMEOUT_SECONDS)
    assert events["bootstrapped"] == ("1.2.3.4", 22022, "/workspace/sap")
    assert events["ran"] == ("1.2.3.4", 22022, "/workspace/sap", 15)
    assert events["downloaded"] == ("1.2.3.4", 22022, "/workspace/sap")
    assert events["terminated"] == "pod-123"


def test_run_remote_keeps_pod_when_requested(monkeypatch) -> None:
    terminated: list[str] = []

    class _FakeClient:
        def create_pod_with_fallback(self, config, gpu_preferences):
            return SimpleNamespace(id="pod-keep", gpu_type="NVIDIA GeForce RTX 4090")

        def wait_for_pod(self, pod_id, timeout=0):
            return SimpleNamespace(id=pod_id, gpu_type="NVIDIA GeForce RTX 4090", ssh_host="1.2.3.4", ssh_port=22022)

        def terminate_pod(self, pod_id):
            terminated.append(pod_id)

    monkeypatch.setattr(runpod_client, "RunPodClient", lambda: _FakeClient())
    monkeypatch.setattr(runpod_client, "build_gpu_fallback_types", lambda primary: ["NVIDIA GeForce RTX 4090"])
    monkeypatch.setattr(run_scaled, "_bootstrap_remote", lambda host, port, remote_dir: None)
    monkeypatch.setattr(run_scaled, "_run_remote_sweep", lambda host, port, remote_dir, args: None)
    monkeypatch.setattr(run_scaled, "_download_results", lambda host, port, remote_dir: None)

    args = SimpleNamespace(gpu="4090", epochs=15, compile=False, symbols=None, configs=None, keep_pod=True)

    run_scaled.run_remote(args)

    assert terminated == []


def test_bootstrap_remote_raises_on_local_sync_failure(monkeypatch) -> None:
    def _fake_run(cmd, check=False, text=False, capture_output=False):
        assert check is False
        assert text is True
        assert capture_output is True
        return SimpleNamespace(returncode=23, stdout="partial transfer", stderr="ssh: connection refused")

    monkeypatch.setattr(run_scaled.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="Failed to sync repository to remote pod"):
        run_scaled._bootstrap_remote("1.2.3.4", 22022, "/workspace/sap")


def test_run_remote_sweep_raises_on_remote_command_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        run_scaled,
        "_ssh",
        lambda host, port, cmd: SimpleNamespace(returncode=17, stdout="remote stdout", stderr="remote stderr"),
    )

    args = SimpleNamespace(gpu="4090", epochs=15, compile=True, symbols=["BTCUSD"], configs="fast")

    with pytest.raises(RuntimeError, match="Remote training sweep failed"):
        run_scaled._run_remote_sweep("1.2.3.4", 22022, "/workspace/sap", args)


def test_run_remote_terminates_pod_when_bootstrap_fails(monkeypatch) -> None:
    terminated: list[str] = []

    class _FakeClient:
        def create_pod_with_fallback(self, config, gpu_preferences):
            return SimpleNamespace(id="pod-fail", gpu_type="NVIDIA GeForce RTX 4090")

        def wait_for_pod(self, pod_id, timeout=0):
            return SimpleNamespace(id=pod_id, gpu_type="NVIDIA GeForce RTX 4090", ssh_host="1.2.3.4", ssh_port=22022)

        def terminate_pod(self, pod_id):
            terminated.append(pod_id)

    monkeypatch.setattr(runpod_client, "RunPodClient", lambda: _FakeClient())
    monkeypatch.setattr(runpod_client, "build_gpu_fallback_types", lambda primary: ["NVIDIA GeForce RTX 4090"])
    monkeypatch.setattr(
        run_scaled,
        "_bootstrap_remote",
        lambda host, port, remote_dir: (_ for _ in ()).throw(RuntimeError("bootstrap failed")),
    )
    monkeypatch.setattr(run_scaled, "_run_remote_sweep", lambda host, port, remote_dir, args: None)
    monkeypatch.setattr(run_scaled, "_download_results", lambda host, port, remote_dir: None)

    args = SimpleNamespace(gpu="4090", epochs=15, compile=False, symbols=None, configs=None, keep_pod=False)

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        run_scaled.run_remote(args)

    assert terminated == ["pod-fail"]
