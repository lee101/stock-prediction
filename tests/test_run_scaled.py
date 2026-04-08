from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import sharpnessadjustedproximalpolicy.run_scaled as run_scaled
import src.runpod_client as runpod_client
import src.runpod_remote_utils as remote_utils


def test_resolve_requested_configs_uses_top_configs_by_default() -> None:
    configs = run_scaled._resolve_requested_configs(None)

    assert [config["name"] for config in configs] == [
        config["name"] for config in run_scaled.TOP_CONFIGS
    ]


def test_resolve_requested_configs_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown --configs values: not_a_real_config"):
        run_scaled._resolve_requested_configs("baseline_wd01,not_a_real_config")


def test_run_remote_rejects_unknown_configs_before_provisioning(monkeypatch) -> None:
    created: list[bool] = []

    class _FakeClient:
        def __init__(self) -> None:
            created.append(True)

    monkeypatch.setattr(runpod_client, "RunPodClient", _FakeClient)

    args = SimpleNamespace(
        gpu="4090",
        epochs=15,
        compile=False,
        symbols=None,
        configs="not_a_real_config",
        keep_pod=False,
    )

    with pytest.raises(ValueError, match="Unknown --configs values: not_a_real_config"):
        run_scaled.run_remote(args)

    assert created == []


def test_main_dry_run_prints_remote_plan_without_creating_pod(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        runpod_client,
        "RunPodClient",
        lambda: (_ for _ in ()).throw(AssertionError("RunPodClient should not be created in --dry-run mode")),
    )
    monkeypatch.setattr(
        runpod_client,
        "resolve_gpu_preferences",
        lambda primary, fallbacks=None: ("NVIDIA GeForce RTX 4090", "NVIDIA A40"),
    )

    run_scaled.main(
        [
            "--gpu",
            "4090",
            "--epochs",
            "15",
            "--configs",
            "baseline_wd01,periodic_wd01",
            "--symbols",
            "BTCUSD",
            "ETHUSD",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload == {
        "compile": False,
        "config_names": ["baseline_wd01", "periodic_wd01"],
        "epochs": 15,
        "gpu_preferences": ["NVIDIA GeForce RTX 4090", "NVIDIA A40"],
        "keep_pod": False,
        "mode": "runpod_remote",
        "remote_command_preview": (
            "cd /workspace/sap && source /workspace/venv/bin/activate 2>/dev/null || true && "
            "PYTHONUNBUFFERED=1 python -m sharpnessadjustedproximalpolicy.run_scaled --local --epochs 15 "
            "--symbols BTCUSD ETHUSD --configs baseline_wd01,periodic_wd01"
        ),
        "remote_dir": run_scaled.DEFAULT_REMOTE_WORKSPACE,
        "status": "dry_run",
        "symbol_source": "cli",
        "symbols": ["BTCUSD", "ETHUSD"],
        "wait_timeout_seconds": run_scaled.RUNPOD_SSH_READY_TIMEOUT_SECONDS,
    }


def test_main_dry_run_text_prints_summary_to_stderr(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        runpod_client,
        "resolve_gpu_preferences",
        lambda primary, fallbacks=None: ("NVIDIA GeForce RTX 4090",),
    )

    run_scaled.main(
        [
            "--gpu",
            "4090",
            "--configs",
            "baseline_wd01",
            "--dry-run",
            "--dry-run-text",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["gpu_preferences"] == ["NVIDIA GeForce RTX 4090"]
    assert payload["config_names"] == ["baseline_wd01"]
    assert payload["symbol_source"] == "auto"
    assert "SAP RunPod Sweep Plan" in captured.err
    assert "Status: dry run" in captured.err
    assert "GPU preferences: NVIDIA GeForce RTX 4090" in captured.err
    assert "Symbols: auto-detect eligible cached symbols on remote host" in captured.err
    assert "Next step: rerun without --dry-run to provision the pod and execute the sweep." in captured.err


def test_parse_args_rejects_dry_run_without_remote_gpu(capsys) -> None:
    with pytest.raises(SystemExit):
        run_scaled.parse_args(["--local", "--dry-run"])

    assert "--dry-run requires --gpu TYPE" in capsys.readouterr().err


def test_parse_args_rejects_dry_run_text_without_dry_run(capsys) -> None:
    with pytest.raises(SystemExit):
        run_scaled.parse_args(["--gpu", "4090", "--dry-run-text"])

    assert "--dry-run-text requires --dry-run" in capsys.readouterr().err


def test_parse_args_rejects_gpu_fallbacks_without_gpu(capsys) -> None:
    with pytest.raises(SystemExit):
        run_scaled.parse_args(["--gpu-fallbacks", "a40,l4"])

    assert "--gpu-fallbacks requires --gpu TYPE" in capsys.readouterr().err


def test_main_dry_run_allows_custom_gpu_fallbacks(monkeypatch, capsys) -> None:
    observed: dict[str, object] = {}

    def _fake_resolve_gpu_preferences(primary, fallbacks=None):
        observed["primary"] = primary
        observed["fallbacks"] = fallbacks
        return ("NVIDIA GeForce RTX 4090", "NVIDIA RTX 6000 Ada Generation")

    monkeypatch.setattr(runpod_client, "resolve_gpu_preferences", _fake_resolve_gpu_preferences)

    run_scaled.main(
        [
            "--gpu",
            "4090",
            "--gpu-fallbacks",
            "6000-ada",
            "--configs",
            "baseline_wd01",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert observed == {"primary": "4090", "fallbacks": "6000-ada"}
    assert payload["gpu_preferences"] == [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA RTX 6000 Ada Generation",
    ]


def test_main_dry_run_allows_disabling_gpu_fallbacks(monkeypatch, capsys) -> None:
    observed: dict[str, object] = {}

    def _fake_resolve_gpu_preferences(primary, fallbacks=None):
        observed["primary"] = primary
        observed["fallbacks"] = fallbacks
        return ("NVIDIA GeForce RTX 4090",)

    monkeypatch.setattr(runpod_client, "resolve_gpu_preferences", _fake_resolve_gpu_preferences)

    run_scaled.main(
        [
            "--gpu",
            "4090",
            "--gpu-fallbacks",
            "none",
            "--configs",
            "baseline_wd01",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert observed == {"primary": "4090", "fallbacks": "none"}
    assert payload["gpu_preferences"] == ["NVIDIA GeForce RTX 4090"]


def test_run_remote_uses_shared_runpod_fallback_helper(monkeypatch) -> None:
    events: dict[str, object] = {}

    class _FakeClient:
        def create_ready_pod_with_fallback(self, config, gpu_preferences, *, timeout=0, poll_interval=0):
            events["config"] = config
            events["gpu_preferences"] = list(gpu_preferences)
            events["wait"] = ("pod-123", timeout, poll_interval)
            return SimpleNamespace(id="pod-123", gpu_type="NVIDIA A40", ssh_host="1.2.3.4", ssh_port=22022)

        def terminate_pod(self, pod_id):
            events["terminated"] = pod_id

    monkeypatch.setattr(runpod_client, "RunPodClient", lambda: _FakeClient())
    monkeypatch.setattr(
        runpod_client,
        "resolve_gpu_preferences",
        lambda primary, fallbacks=None: ("NVIDIA GeForce RTX 4090", "NVIDIA A40"),
    )
    monkeypatch.setattr(run_scaled.time, "strftime", lambda fmt: "04071234")
    monkeypatch.setattr(run_scaled, "_bootstrap_remote", lambda host, port, remote_dir: events.setdefault("bootstrapped", (host, port, remote_dir)))
    monkeypatch.setattr(run_scaled, "_run_remote_sweep", lambda host, port, remote_dir, args: events.setdefault("ran", (host, port, remote_dir, args.epochs)))
    monkeypatch.setattr(run_scaled, "_download_results", lambda host, port, remote_dir: events.setdefault("downloaded", (host, port, remote_dir)))

    args = SimpleNamespace(
        gpu="4090",
        gpu_fallbacks=None,
        epochs=15,
        compile=False,
        symbols=None,
        configs=None,
        keep_pod=False,
    )

    run_scaled.run_remote(args)

    config = events["config"]
    assert config.name == "sap-sweep-04071234"
    assert config.gpu_type == "NVIDIA GeForce RTX 4090"
    assert events["gpu_preferences"] == ["NVIDIA GeForce RTX 4090", "NVIDIA A40"]
    assert events["wait"] == ("pod-123", run_scaled.RUNPOD_SSH_READY_TIMEOUT_SECONDS, 0)
    assert events["bootstrapped"] == ("1.2.3.4", 22022, "/workspace/sap")
    assert events["ran"] == ("1.2.3.4", 22022, "/workspace/sap", 15)
    assert events["downloaded"] == ("1.2.3.4", 22022, "/workspace/sap")
    assert events["terminated"] == "pod-123"


def test_run_remote_keeps_pod_when_requested(monkeypatch) -> None:
    terminated: list[str] = []

    class _FakeClient:
        def create_ready_pod_with_fallback(self, config, gpu_preferences, *, timeout=0, poll_interval=0):
            return SimpleNamespace(id="pod-keep", gpu_type="NVIDIA GeForce RTX 4090", ssh_host="1.2.3.4", ssh_port=22022)

        def terminate_pod(self, pod_id):
            terminated.append(pod_id)

    monkeypatch.setattr(runpod_client, "RunPodClient", lambda: _FakeClient())
    monkeypatch.setattr(
        runpod_client,
        "resolve_gpu_preferences",
        lambda primary, fallbacks=None: ("NVIDIA GeForce RTX 4090",),
    )
    monkeypatch.setattr(run_scaled, "_bootstrap_remote", lambda host, port, remote_dir: None)
    monkeypatch.setattr(run_scaled, "_run_remote_sweep", lambda host, port, remote_dir, args: None)
    monkeypatch.setattr(run_scaled, "_download_results", lambda host, port, remote_dir: None)

    args = SimpleNamespace(
        gpu="4090",
        gpu_fallbacks=None,
        epochs=15,
        compile=False,
        symbols=None,
        configs=None,
        keep_pod=True,
    )

    run_scaled.run_remote(args)

    assert terminated == []


def test_bootstrap_remote_raises_on_local_sync_failure(monkeypatch) -> None:
    def _fake_run(cmd, check=False, text=False, capture_output=False):
        assert check is False
        assert text is True
        assert capture_output is True
        return SimpleNamespace(returncode=23, stdout="partial transfer", stderr="ssh: connection refused")

    monkeypatch.setattr(remote_utils.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="Failed to sync repository to remote pod"):
        run_scaled._bootstrap_remote("1.2.3.4", 22022, "/workspace/sap")


def test_bootstrap_remote_shell_quotes_install_directory(monkeypatch) -> None:
    captured: dict[str, str] = {}

    monkeypatch.setattr(run_scaled, "_run_checked_local", lambda *args, **kwargs: None)

    def _fake_ssh(host, port, cmd):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_scaled, "_ssh", _fake_ssh)

    run_scaled._bootstrap_remote("1.2.3.4", 22022, "/workspace/sap dir; touch /tmp/pwned")

    cmd = captured["cmd"]
    assert f"cd {run_scaled.shlex.quote('/workspace/sap dir; touch /tmp/pwned')}" in cmd
    assert "cd /workspace/sap dir; touch /tmp/pwned &&" not in cmd


def test_bootstrap_remote_reports_install_failure_with_context(monkeypatch) -> None:
    monkeypatch.setattr(run_scaled, "_run_checked_local", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_scaled,
        "_ssh",
        lambda host, port, cmd: SimpleNamespace(returncode=17, stdout="pip output", stderr="pip failed"),
    )

    with pytest.raises(RuntimeError) as excinfo:
        run_scaled._bootstrap_remote("1.2.3.4", 22022, "/workspace/sap")

    message = str(excinfo.value)
    assert "RunPod SAP remote bootstrap/install failed" in message
    assert "ssh_host: 1.2.3.4" in message
    assert "ssh_port: 22022" in message
    assert "remote_dir: /workspace/sap" in message
    assert "Remote dependency installation failed (exit 17)" in message
    assert "stdout excerpt:\npip output" in message
    assert "stderr excerpt:\npip failed" in message


def test_run_remote_sweep_raises_on_remote_command_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        run_scaled,
        "_ssh",
        lambda host, port, cmd: SimpleNamespace(returncode=17, stdout="remote stdout", stderr="remote stderr"),
    )

    args = SimpleNamespace(gpu="4090", epochs=15, compile=True, symbols=["BTCUSD"], configs="fast")

    with pytest.raises(RuntimeError, match="Remote training sweep failed"):
        run_scaled._run_remote_sweep("1.2.3.4", 22022, "/workspace/sap", args)


def test_run_remote_sweep_shell_quotes_remote_args(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_ssh(host, port, cmd):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_scaled, "_ssh", _fake_ssh)

    args = SimpleNamespace(
        gpu="4090",
        epochs=15,
        compile=False,
        symbols=["BTCUSD", "BAD; touch /tmp/pwned"],
        configs="baseline; rm -rf /",
    )

    run_scaled._run_remote_sweep("1.2.3.4", 22022, "/workspace/sap dir", args)

    cmd = captured["cmd"]
    assert f"cd {run_scaled.shlex.quote('/workspace/sap dir')}" in cmd
    assert run_scaled.shlex.quote("BAD; touch /tmp/pwned") in cmd
    assert run_scaled.shlex.quote("baseline; rm -rf /") in cmd
    assert "--configs baseline; rm -rf /" not in cmd


def test_download_results_raises_on_rsync_failure(monkeypatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd, check=False, text=False, capture_output=False):
        calls.append(cmd)
        return SimpleNamespace(returncode=24, stdout="partial files", stderr="connection reset")

    monkeypatch.setattr(remote_utils.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="Failed to download remote result summaries"):
        run_scaled._download_results("1.2.3.4", 22022, "/workspace/sap")

    assert calls


def test_run_remote_terminates_pod_when_bootstrap_fails(monkeypatch) -> None:
    terminated: list[str] = []

    class _FakeClient:
        def create_ready_pod_with_fallback(self, config, gpu_preferences, *, timeout=0, poll_interval=0):
            return SimpleNamespace(id="pod-fail", gpu_type="NVIDIA GeForce RTX 4090", ssh_host="1.2.3.4", ssh_port=22022)

        def terminate_pod(self, pod_id):
            terminated.append(pod_id)

    monkeypatch.setattr(runpod_client, "RunPodClient", lambda: _FakeClient())
    monkeypatch.setattr(
        runpod_client,
        "resolve_gpu_preferences",
        lambda primary, fallbacks=None: ("NVIDIA GeForce RTX 4090",),
    )
    monkeypatch.setattr(
        run_scaled,
        "_bootstrap_remote",
        lambda host, port, remote_dir: (_ for _ in ()).throw(RuntimeError("bootstrap failed")),
    )
    monkeypatch.setattr(run_scaled, "_run_remote_sweep", lambda host, port, remote_dir, args: None)
    monkeypatch.setattr(run_scaled, "_download_results", lambda host, port, remote_dir: None)

    args = SimpleNamespace(
        gpu="4090",
        gpu_fallbacks=None,
        epochs=15,
        compile=False,
        symbols=None,
        configs=None,
        keep_pod=False,
    )

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        run_scaled.run_remote(args)

    assert terminated == ["pod-fail"]


def test_run_remote_wraps_stage_error_with_pod_context(monkeypatch) -> None:
    terminated: list[str] = []

    class _FakeClient:
        def create_ready_pod_with_fallback(self, config, gpu_preferences, *, timeout=0, poll_interval=0):
            return SimpleNamespace(
                id="pod-fail",
                gpu_type="NVIDIA GeForce RTX 4090",
                ssh_host="1.2.3.4",
                ssh_port=22022,
            )

        def terminate_pod(self, pod_id):
            terminated.append(pod_id)

    monkeypatch.setattr(runpod_client, "RunPodClient", lambda: _FakeClient())
    monkeypatch.setattr(
        runpod_client,
        "resolve_gpu_preferences",
        lambda primary, fallbacks=None: ("NVIDIA GeForce RTX 4090",),
    )
    monkeypatch.setattr(
        run_scaled,
        "_bootstrap_remote",
        lambda host, port, remote_dir: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )
    monkeypatch.setattr(run_scaled, "_run_remote_sweep", lambda host, port, remote_dir, args: None)
    monkeypatch.setattr(run_scaled, "_download_results", lambda host, port, remote_dir: None)

    args = SimpleNamespace(
        gpu="4090",
        gpu_fallbacks=None,
        epochs=15,
        compile=False,
        symbols=None,
        configs=None,
        keep_pod=False,
    )

    with pytest.raises(RuntimeError) as excinfo:
        run_scaled.run_remote(args)

    message = str(excinfo.value)
    assert "RunPod SAP remote bootstrap/install failed" in message
    assert "pod_id: pod-fail" in message
    assert "gpu_type: NVIDIA GeForce RTX 4090" in message
    assert "ssh_host: 1.2.3.4" in message
    assert "ssh_port: 22022" in message
    assert "remote_dir: /workspace/sap" in message
    assert "sync failed" in message
    assert terminated == ["pod-fail"]


def test_run_remote_preserves_stage_error_when_terminate_fails(monkeypatch) -> None:
    class _FakeClient:
        def create_ready_pod_with_fallback(self, config, gpu_preferences, *, timeout=0, poll_interval=0):
            return SimpleNamespace(
                id="pod-fail",
                gpu_type="NVIDIA GeForce RTX 4090",
                ssh_host="1.2.3.4",
                ssh_port=22022,
            )

        def terminate_pod(self, pod_id):
            raise RuntimeError("terminate failed")

    monkeypatch.setattr(runpod_client, "RunPodClient", lambda: _FakeClient())
    monkeypatch.setattr(
        runpod_client,
        "resolve_gpu_preferences",
        lambda primary, fallbacks=None: ("NVIDIA GeForce RTX 4090",),
    )
    monkeypatch.setattr(
        run_scaled,
        "_bootstrap_remote",
        lambda host, port, remote_dir: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )
    monkeypatch.setattr(run_scaled, "_run_remote_sweep", lambda host, port, remote_dir, args: None)
    monkeypatch.setattr(run_scaled, "_download_results", lambda host, port, remote_dir: None)

    args = SimpleNamespace(
        gpu="4090",
        gpu_fallbacks=None,
        epochs=15,
        compile=False,
        symbols=None,
        configs=None,
        keep_pod=False,
    )

    with pytest.raises(RuntimeError) as excinfo:
        run_scaled.run_remote(args)

    message = str(excinfo.value)
    assert "RunPod SAP remote bootstrap/install failed" in message
    assert "sync failed" in message
    assert excinfo.value.__notes__ is not None
    assert any(
        "failed to terminate pod pod-fail after remote failure: terminate failed" in note
        for note in excinfo.value.__notes__
    )


def test_run_remote_wraps_install_failure_with_pod_context(monkeypatch) -> None:
    terminated: list[str] = []

    class _FakeClient:
        def create_ready_pod_with_fallback(self, config, gpu_preferences, *, timeout=0, poll_interval=0):
            return SimpleNamespace(
                id="pod-fail",
                gpu_type="NVIDIA GeForce RTX 4090",
                ssh_host="1.2.3.4",
                ssh_port=22022,
            )

        def terminate_pod(self, pod_id):
            terminated.append(pod_id)

    monkeypatch.setattr(runpod_client, "RunPodClient", lambda: _FakeClient())
    monkeypatch.setattr(
        runpod_client,
        "resolve_gpu_preferences",
        lambda primary, fallbacks=None: ("NVIDIA GeForce RTX 4090",),
    )
    monkeypatch.setattr(
        run_scaled,
        "_bootstrap_remote",
        lambda host, port, remote_dir: (_ for _ in ()).throw(
            RuntimeError(
                "RunPod SAP remote bootstrap/install failed\n"
                "ssh_host: 1.2.3.4\n"
                "ssh_port: 22022\n"
                "remote_dir: /workspace/sap\n"
                "Remote dependency installation failed (exit 17)"
            )
        ),
    )
    monkeypatch.setattr(run_scaled, "_run_remote_sweep", lambda host, port, remote_dir, args: None)
    monkeypatch.setattr(run_scaled, "_download_results", lambda host, port, remote_dir: None)

    args = SimpleNamespace(
        gpu="4090",
        gpu_fallbacks=None,
        epochs=15,
        compile=False,
        symbols=None,
        configs=None,
        keep_pod=False,
    )

    with pytest.raises(RuntimeError) as excinfo:
        run_scaled.run_remote(args)

    message = str(excinfo.value)
    assert "RunPod SAP remote bootstrap/install failed" in message
    assert "pod_id: pod-fail" in message
    assert "gpu_type: NVIDIA GeForce RTX 4090" in message
    assert "ssh_host: 1.2.3.4" in message
    assert "ssh_port: 22022" in message
    assert "remote_dir: /workspace/sap" in message
    assert terminated == ["pod-fail"]
