from __future__ import annotations

import builtins
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.runpod_remote_utils as remote_utils
from src import runpod_client
from src.daily_stock_defaults import DEFAULT_CHECKPOINT, DEFAULT_EXTRA_CHECKPOINTS, DEFAULT_SYMBOLS


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "runpod_daily_stock_planner.py"


def _load_module():
    spec = spec_from_file_location("runpod_daily_stock_planner", SCRIPT_PATH)
    assert spec is not None
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_json_and_remote_path_helpers(tmp_path) -> None:
    mod = _load_module()

    payload = mod._extract_json('boot logs\n{"plan": {"direction": "hold"}}\n')
    assert payload == {"plan": {"direction": "hold"}}

    in_repo = mod.REPO / "trade_daily_stock_prod.py"
    assert mod._relative_remote_path(in_repo, "/workspace/stock") == "/workspace/stock/trade_daily_stock_prod.py"

    external = tmp_path / "checkpoint.pt"
    external.write_text("stub", encoding="utf-8")
    remote_external = mod._relative_remote_path(external, "/workspace/stock")
    assert remote_external.startswith("/workspace/stock/.external/checkpoint-")
    assert remote_external.endswith(".pt")


def test_load_module_falls_back_when_runpod_client_lacks_resolve_gpu_preferences(monkeypatch) -> None:
    monkeypatch.delattr(runpod_client, "resolve_gpu_preferences", raising=False)
    monkeypatch.setattr(runpod_client, "parse_gpu_fallback_types", lambda value: ["l4"] if value else None)
    monkeypatch.setattr(
        runpod_client,
        "build_gpu_fallback_types",
        lambda primary, fallbacks=None: [primary, *(fallbacks or [])],
    )

    mod = _load_module()

    assert mod.resolve_gpu_preferences("4090", "l4") == ("4090", "l4")


def test_load_module_does_not_import_daily_stock_entrypoint(monkeypatch) -> None:
    original_import = builtins.__import__

    def _guarded_import(name, _globals=None, _locals=None, fromlist=(), level=0):
        if name == "trade_daily_stock_prod":
            raise AssertionError("planner should not import trade_daily_stock_prod for defaults")
        return original_import(name, _globals, _locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    mod = _load_module()

    assert mod.DEFAULT_CHECKPOINT == DEFAULT_CHECKPOINT
    assert mod.DEFAULT_SYMBOLS == DEFAULT_SYMBOLS


def test_daily_stock_defaults_are_immutable_sequences() -> None:
    assert isinstance(DEFAULT_SYMBOLS, tuple)
    assert isinstance(DEFAULT_EXTRA_CHECKPOINTS, tuple)


def test_relative_remote_path_disambiguates_external_paths_with_same_name(tmp_path) -> None:
    mod = _load_module()
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "checkpoint.pt"
    second = second_dir / "checkpoint.pt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")

    first_remote = mod._relative_remote_path(first, "/workspace/stock")
    second_remote = mod._relative_remote_path(second, "/workspace/stock")

    assert first_remote != second_remote
    assert first_remote.startswith("/workspace/stock/.external/checkpoint-")
    assert second_remote.startswith("/workspace/stock/.external/checkpoint-")
    assert first_remote.endswith(".pt")
    assert second_remote.endswith(".pt")


def test_extract_json_reports_invalid_remote_output() -> None:
    mod = _load_module()

    with pytest.raises(RuntimeError, match="did not emit valid JSON payload"):
        mod._extract_json("boot logs\nnot-json\n")


def test_extract_json_rejects_non_object_payload() -> None:
    mod = _load_module()

    with pytest.raises(RuntimeError, match="JSON payload must be an object"):
        mod._extract_json('["not", "an", "object"]')


def test_normalize_forward_env_names_filters_invalid_and_duplicate_names() -> None:
    mod = _load_module()

    normalized, invalid = mod._normalize_forward_env_names(
        [" ALPACA_API_KEY ", "BAD-NAME", "ALPACA_API_KEY", "ALSO=BAD", ""]
    )

    assert normalized == ("ALPACA_API_KEY",)
    assert invalid == ("BAD-NAME", "ALSO=BAD")


def test_rsync_repo_wraps_subprocess_failures(monkeypatch) -> None:
    mod = _load_module()

    def _fake_run(cmd, check=False, text=False, capture_output=False):
        assert check is False
        assert text is True
        assert capture_output is True
        return SimpleNamespace(returncode=23, stdout="partial transfer", stderr="ssh: connection refused")

    monkeypatch.setattr(remote_utils.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="Failed to rsync repository"):
        mod._rsync_repo(
            ssh_host="1.2.3.4",
            ssh_port=22022,
            remote_workspace="/workspace/stock-prediction",
        )


def test_main_syncs_inputs_and_prints_remote_payload(monkeypatch, tmp_path, capsys) -> None:
    mod = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    extra_one = tmp_path / "extra_one.pt"
    extra_two = tmp_path / "extra_two.pt"
    extra_one.write_text("one", encoding="utf-8")
    extra_two.write_text("two", encoding="utf-8")

    class _FakeClient:
        def __init__(self) -> None:
            self.created = []
            self.requested_gpu_preferences: tuple[str, ...] | None = None
            self.wait_timeout: int | None = None
            self.wait_poll_interval: int | None = None
            self.terminated: list[str] = []

        def create_ready_pod_with_fallback(self, config, gpu_preferences, *, timeout=0, poll_interval=0):
            self.created.append(config)
            self.requested_gpu_preferences = tuple(gpu_preferences)
            self.wait_timeout = timeout
            self.wait_poll_interval = poll_interval
            return SimpleNamespace(
                id="pod-123",
                gpu_type="NVIDIA RTX 4090",
                ssh_host="1.2.3.4",
                ssh_port=22022,
            )

        def terminate_pod(self, pod_id):
            self.terminated.append(pod_id)

    fake_client = _FakeClient()
    rsync_repo_calls: list[dict[str, object]] = []
    rsync_path_calls: list[dict[str, object]] = []
    ssh_cmds: list[str] = []

    def _fake_ssh_run(*, ssh_host: str, ssh_port: int, remote_cmd: str, capture_output: bool = False):
        ssh_cmds.append(remote_cmd)
        if "python -m pip install -e ." in remote_cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "python trade_daily_stock_prod.py" in remote_cmd:
            return SimpleNamespace(
                returncode=0,
                stdout='planner logs\n{"plan": {"symbol": "AAPL", "direction": "long"}}\n',
                stderr="",
            )
        raise AssertionError(f"unexpected ssh command: {remote_cmd}")

    monkeypatch.setattr(mod, "RunPodClient", lambda: fake_client)
    monkeypatch.setattr(mod, "resolve_gpu_preferences", lambda primary, fallbacks=None: (primary,))
    monkeypatch.setattr(mod, "_rsync_repo", lambda **kwargs: rsync_repo_calls.append(kwargs))
    monkeypatch.setattr(mod, "_rsync_path", lambda **kwargs: rsync_path_calls.append(kwargs))
    monkeypatch.setattr(mod, "_ssh_run", _fake_ssh_run)
    monkeypatch.setattr(mod.time, "time", lambda: 1_700_000_000)
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")

    mod.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--extra-checkpoints",
            str(extra_one),
            str(extra_two),
            "--symbols",
            "AAPL",
            "MSFT",
            "--forward-env",
            "ALPACA_API_KEY",
            "MISSING_ENV",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["plan"] == {"symbol": "AAPL", "direction": "long"}
    assert payload["runpod"] == {
        "pod_id": "pod-123",
        "gpu_type": "NVIDIA RTX 4090",
        "ssh_host": "1.2.3.4",
    }
    assert fake_client.created[0].name == "daily-stock-plan-1700000000"
    assert fake_client.created[0].gpu_type == "4090"
    assert fake_client.requested_gpu_preferences == ("4090",)
    assert fake_client.wait_timeout == mod.DEFAULT_POD_READY_TIMEOUT_SECONDS
    assert fake_client.wait_poll_interval == mod.DEFAULT_POD_READY_POLL_INTERVAL_SECONDS
    assert fake_client.terminated == ["pod-123"]

    assert rsync_repo_calls == [
        {
            "ssh_host": "1.2.3.4",
            "ssh_port": 22022,
            "remote_workspace": mod.DEFAULT_REMOTE_WORKSPACE,
        }
    ]
    assert len(rsync_path_calls) == 4
    assert {Path(call["local_path"]).name for call in rsync_path_calls} == {
        "checkpoint.pt",
        "data",
        "extra_one.pt",
        "extra_two.pt",
    }

    planner_cmd = next(cmd for cmd in ssh_cmds if "python trade_daily_stock_prod.py" in cmd)
    checkpoint_remote = mod._relative_remote_path(checkpoint, mod.DEFAULT_REMOTE_WORKSPACE)
    data_dir_remote = mod._relative_remote_path(data_dir, mod.DEFAULT_REMOTE_WORKSPACE)
    extra_one_remote = mod._relative_remote_path(extra_one, mod.DEFAULT_REMOTE_WORKSPACE)
    extra_two_remote = mod._relative_remote_path(extra_two, mod.DEFAULT_REMOTE_WORKSPACE)
    assert "ALPACA_API_KEY=test-key python trade_daily_stock_prod.py" in planner_cmd
    assert "--dry-run" in planner_cmd
    assert "--print-payload" in planner_cmd
    assert f"--checkpoint {checkpoint_remote}" in planner_cmd
    assert f"--data-dir {data_dir_remote}" in planner_cmd
    assert "--extra-checkpoints" in planner_cmd
    assert extra_one_remote in planner_cmd
    assert extra_two_remote in planner_cmd
    assert "--symbols AAPL MSFT" in planner_cmd


def test_main_terminates_pod_when_remote_output_is_invalid(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    class _FakeClient:
        def __init__(self) -> None:
            self.terminated: list[str] = []

        def create_ready_pod_with_fallback(self, config, gpu_preferences, *, timeout=0, poll_interval=0):
            return SimpleNamespace(
                id="pod-123",
                gpu_type="NVIDIA RTX 4090",
                ssh_host="1.2.3.4",
                ssh_port=22022,
            )

        def terminate_pod(self, pod_id):
            self.terminated.append(pod_id)

    fake_client = _FakeClient()

    def _fake_ssh_run(*, ssh_host: str, ssh_port: int, remote_cmd: str, capture_output: bool = False):
        if "python -m pip install -e ." in remote_cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "python trade_daily_stock_prod.py" in remote_cmd:
            return SimpleNamespace(returncode=0, stdout="planner logs\nnot-json\n", stderr="")
        raise AssertionError(f"unexpected ssh command: {remote_cmd}")

    monkeypatch.setattr(mod, "RunPodClient", lambda: fake_client)
    monkeypatch.setattr(mod, "resolve_gpu_preferences", lambda primary, fallbacks=None: (primary,))
    monkeypatch.setattr(mod, "_rsync_repo", lambda **kwargs: None)
    monkeypatch.setattr(mod, "_rsync_path", lambda **kwargs: None)
    monkeypatch.setattr(mod, "_ssh_run", _fake_ssh_run)

    with pytest.raises(RuntimeError) as excinfo:
        mod.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                str(data_dir),
                "--no-ensemble",
                "--symbols",
                "AAPL",
            ]
        )

    message = str(excinfo.value)
    assert "RunPod planner planner output parsing failed" in message
    assert "pod_id: pod-123" in message
    assert "gpu_type: NVIDIA RTX 4090" in message
    assert "ssh_host: 1.2.3.4" in message
    assert "did not emit valid JSON payload" in message
    assert "python trade_daily_stock_prod.py" in message
    assert fake_client.terminated == ["pod-123"]


def test_main_reports_remote_bootstrap_failure_with_stage_context(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    class _FakeClient:
        def __init__(self) -> None:
            self.terminated: list[str] = []

        def create_ready_pod_with_fallback(self, config, gpu_preferences, *, timeout=0, poll_interval=0):
            return SimpleNamespace(
                id="pod-123",
                gpu_type="NVIDIA RTX 4090",
                ssh_host="1.2.3.4",
                ssh_port=22022,
            )

        def terminate_pod(self, pod_id):
            self.terminated.append(pod_id)

    fake_client = _FakeClient()

    def _fake_ssh_run(*, ssh_host: str, ssh_port: int, remote_cmd: str, capture_output: bool = False):
        if "python -m pip install -e ." in remote_cmd:
            return SimpleNamespace(returncode=19, stdout="bootstrap stdout", stderr="bootstrap stderr")
        raise AssertionError(f"unexpected ssh command: {remote_cmd}")

    monkeypatch.setattr(mod, "RunPodClient", lambda: fake_client)
    monkeypatch.setattr(mod, "resolve_gpu_preferences", lambda primary, fallbacks=None: (primary,))
    monkeypatch.setattr(mod, "_rsync_repo", lambda **kwargs: None)
    monkeypatch.setattr(mod, "_rsync_path", lambda **kwargs: None)
    monkeypatch.setattr(mod, "_ssh_run", _fake_ssh_run)

    with pytest.raises(RuntimeError) as excinfo:
        mod.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                str(data_dir),
                "--no-ensemble",
                "--symbols",
                "AAPL",
            ]
        )

    message = str(excinfo.value)
    assert "RunPod planner bootstrap failed" in message
    assert "pod_id: pod-123" in message
    assert "gpu_type: NVIDIA RTX 4090" in message
    assert "ssh_host: 1.2.3.4" in message
    assert "remote bootstrap command returned exit 19" in message
    assert "/tmp/runpod_daily_pip.log" in message
    assert "stdout excerpt:\nbootstrap stdout" in message
    assert "stderr excerpt:\nbootstrap stderr" in message
    assert fake_client.terminated == ["pod-123"]


def test_main_reports_remote_planner_exit_with_stage_context(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    class _FakeClient:
        def __init__(self) -> None:
            self.terminated: list[str] = []

        def create_ready_pod_with_fallback(self, config, gpu_preferences, *, timeout=0, poll_interval=0):
            return SimpleNamespace(
                id="pod-123",
                gpu_type="NVIDIA RTX 4090",
                ssh_host="1.2.3.4",
                ssh_port=22022,
            )

        def terminate_pod(self, pod_id):
            self.terminated.append(pod_id)

    fake_client = _FakeClient()

    def _fake_ssh_run(*, ssh_host: str, ssh_port: int, remote_cmd: str, capture_output: bool = False):
        if "python -m pip install -e ." in remote_cmd:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "python trade_daily_stock_prod.py" in remote_cmd:
            return SimpleNamespace(returncode=17, stdout="planner stdout", stderr="planner stderr")
        raise AssertionError(f"unexpected ssh command: {remote_cmd}")

    monkeypatch.setattr(mod, "RunPodClient", lambda: fake_client)
    monkeypatch.setattr(mod, "resolve_gpu_preferences", lambda primary, fallbacks=None: (primary,))
    monkeypatch.setattr(mod, "_rsync_repo", lambda **kwargs: None)
    monkeypatch.setattr(mod, "_rsync_path", lambda **kwargs: None)
    monkeypatch.setattr(mod, "_ssh_run", _fake_ssh_run)

    with pytest.raises(RuntimeError) as excinfo:
        mod.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                str(data_dir),
                "--no-ensemble",
                "--symbols",
                "AAPL",
            ]
        )

    message = str(excinfo.value)
    assert "RunPod planner planner command failed" in message
    assert "pod_id: pod-123" in message
    assert "gpu_type: NVIDIA RTX 4090" in message
    assert "ssh_host: 1.2.3.4" in message
    assert "remote planner command returned exit 17" in message
    assert "planner_command_preview:" in message
    assert "stdout excerpt:\nplanner stdout" in message
    assert "stderr excerpt:\nplanner stderr" in message
    assert fake_client.terminated == ["pod-123"]


def test_main_dry_run_prints_resolved_plan_without_creating_pod(monkeypatch, tmp_path, capsys) -> None:
    mod = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    def _unexpected_runpod_client():
        raise AssertionError("RunPodClient should not be created in --dry-run mode")

    monkeypatch.setattr(mod, "RunPodClient", _unexpected_runpod_client)
    monkeypatch.setattr(mod, "resolve_gpu_preferences", lambda primary, fallbacks=None: ("4090", "A40", "L4"))
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")

    mod.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--no-ensemble",
            "--symbols",
            "AAPL",
            "MSFT",
            "--gpu-fallbacks",
            "A40,L4",
            "--forward-env",
            "ALPACA_API_KEY",
            "MISSING_ENV",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["ready"] is True
    assert payload["gpu_preferences"] == ["4090", "A40", "L4"]
    assert payload["checkpoint"]["local"] == str(checkpoint)
    assert payload["data_dir"]["local"] == str(data_dir)
    assert payload["forward_env_present"] == ["ALPACA_API_KEY"]
    assert payload["forward_env_missing"] == ["MISSING_ENV"]
    assert "Forwarded environment variables not set: MISSING_ENV" in payload["warnings"]
    assert "ALPACA_API_KEY=$ALPACA_API_KEY python trade_daily_stock_prod.py" in payload["planner_command_preview"]
    assert "--no-ensemble" in payload["planner_command_preview"]
    assert "test-key" not in payload["planner_command_preview"]


def test_main_dry_run_auto_forwards_alpaca_paper_credentials(monkeypatch, tmp_path, capsys) -> None:
    mod = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")

    monkeypatch.setattr(mod, "RunPodClient", lambda: (_ for _ in ()).throw(AssertionError("should not create pod")))
    monkeypatch.setenv("ALP_KEY_ID_PAPER", "paper-key")
    monkeypatch.setenv("ALP_SECRET_KEY_PAPER", "paper-secret")

    mod.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--data-source",
            "alpaca",
            "--no-ensemble",
            "--symbols",
            "AAPL",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["auto_forward_env"] == ["ALP_KEY_ID_PAPER", "ALP_SECRET_KEY_PAPER"]
    assert payload["forward_env_present"] == ["ALP_KEY_ID_PAPER", "ALP_SECRET_KEY_PAPER"]
    assert "Automatically forwarding Alpaca paper credential env vars" in "\n".join(payload["warnings"])
    assert "ALP_KEY_ID_PAPER=$ALP_KEY_ID_PAPER" in payload["planner_command_preview"]
    assert "ALP_SECRET_KEY_PAPER=$ALP_SECRET_KEY_PAPER" in payload["planner_command_preview"]
    assert "paper-key" not in payload["planner_command_preview"]
    assert "paper-secret" not in payload["planner_command_preview"]


def test_main_dry_run_warns_when_alpaca_credentials_are_not_forwarded(monkeypatch, tmp_path, capsys) -> None:
    mod = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")

    monkeypatch.setattr(mod, "RunPodClient", lambda: (_ for _ in ()).throw(AssertionError("should not create pod")))
    monkeypatch.delenv("ALP_KEY_ID_PAPER", raising=False)
    monkeypatch.delenv("ALP_SECRET_KEY_PAPER", raising=False)
    monkeypatch.delenv("ALP_KEY_ID", raising=False)
    monkeypatch.delenv("ALP_SECRET_KEY", raising=False)

    mod.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--data-source",
            "alpaca",
            "--no-ensemble",
            "--symbols",
            "AAPL",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["auto_forward_env"] == []
    assert payload["forward_env_present"] == []
    assert payload["forward_env_missing"] == []
    assert any("Alpaca paper credentials are not being forwarded." in warning for warning in payload["warnings"])


def test_main_dry_run_does_not_auto_forward_live_alpaca_credentials(monkeypatch, tmp_path, capsys) -> None:
    mod = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")

    monkeypatch.setattr(mod, "RunPodClient", lambda: (_ for _ in ()).throw(AssertionError("should not create pod")))
    monkeypatch.delenv("ALP_KEY_ID_PAPER", raising=False)
    monkeypatch.delenv("ALP_SECRET_KEY_PAPER", raising=False)
    monkeypatch.setenv("ALP_KEY_ID", "live-key")
    monkeypatch.setenv("ALP_SECRET_KEY", "live-secret")

    mod.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--data-source",
            "alpaca",
            "--no-ensemble",
            "--symbols",
            "AAPL",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["auto_forward_env"] == []
    assert payload["forward_env_present"] == []
    assert payload["forward_env_missing"] == []
    warnings = "\n".join(payload["warnings"])
    assert "Live Alpaca credentials detected but not auto-forwarded." in warnings
    assert "ALP_KEY_ID=$ALP_KEY_ID" not in payload["planner_command_preview"]
    assert "ALP_SECRET_KEY=$ALP_SECRET_KEY" not in payload["planner_command_preview"]
    assert "live-key" not in payload["planner_command_preview"]
    assert "live-secret" not in payload["planner_command_preview"]


def test_main_dry_run_text_prints_human_summary_to_stderr(monkeypatch, tmp_path, capsys) -> None:
    mod = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    monkeypatch.setattr(mod, "RunPodClient", lambda: (_ for _ in ()).throw(AssertionError("should not create pod")))
    monkeypatch.setattr(mod, "resolve_gpu_preferences", lambda primary, fallbacks=None: (primary,))
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")

    mod.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--no-ensemble",
            "--symbols",
            "AAPL",
            "MSFT",
            "--forward-env",
            "ALPACA_API_KEY",
            "MISSING_ENV",
            "--dry-run",
            "--dry-run-text",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["ready"] is True
    assert "RunPod Daily Planner" in captured.err
    assert "Status: ready" in captured.err
    assert "Resolved symbols: AAPL, MSFT" in captured.err
    assert "Forward env: 1 present, 1 missing" in captured.err
    assert "rerun without --dry-run" in captured.err
    assert "test-key" not in captured.err


def test_main_dry_run_text_reports_auto_forwarded_env(monkeypatch, tmp_path, capsys) -> None:
    mod = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")

    monkeypatch.setattr(mod, "RunPodClient", lambda: (_ for _ in ()).throw(AssertionError("should not create pod")))
    monkeypatch.setenv("ALP_KEY_ID_PAPER", "paper-key")
    monkeypatch.setenv("ALP_SECRET_KEY_PAPER", "paper-secret")

    mod.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--data-source",
            "alpaca",
            "--no-ensemble",
            "--symbols",
            "AAPL",
            "--dry-run",
            "--dry-run-text",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ready"] is True
    assert "Auto-forward env: ALP_KEY_ID_PAPER, ALP_SECRET_KEY_PAPER" in captured.err
    assert "paper-key" not in captured.err
    assert "paper-secret" not in captured.err


def test_main_dry_run_rejects_invalid_forward_env_names(monkeypatch, tmp_path, capsys) -> None:
    mod = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    monkeypatch.setattr(mod, "RunPodClient", lambda: (_ for _ in ()).throw(AssertionError("should not create pod")))
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")

    with pytest.raises(SystemExit, match="1"):
        mod.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                str(data_dir),
                "--no-ensemble",
                "--symbols",
                "AAPL",
                "--forward-env",
                "ALPACA_API_KEY",
                "BAD-NAME",
                "ALSO=BAD",
                "--dry-run",
            ]
        )

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["forward_env_present"] == ["ALPACA_API_KEY"]
    assert payload["forward_env_missing"] == []
    assert payload["errors"] == ["Invalid --forward-env names: BAD-NAME, ALSO=BAD"]
    assert "ALPACA_API_KEY=$ALPACA_API_KEY" in payload["planner_command_preview"]
    assert "BAD-NAME" not in payload["planner_command_preview"]
    assert "ALSO=BAD" not in payload["planner_command_preview"]


def test_main_preflight_fails_before_provisioning_when_checkpoint_missing(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    def _unexpected_runpod_client():
        raise AssertionError("RunPodClient should not be created when preflight fails")

    monkeypatch.setattr(mod, "RunPodClient", _unexpected_runpod_client)

    with pytest.raises(RuntimeError, match="Planner preflight failed"):
        mod.main(
            [
                "--checkpoint",
                str(tmp_path / "missing.pt"),
                "--data-dir",
                str(data_dir),
                "--no-ensemble",
                "--symbols",
                "AAPL",
            ]
        )


def test_main_dry_run_supports_symbols_file(monkeypatch, tmp_path, capsys) -> None:
    mod = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\nmsft\nAAPL\n# comment\n", encoding="utf-8")

    monkeypatch.setattr(mod, "RunPodClient", lambda: (_ for _ in ()).throw(AssertionError("should not create pod")))
    monkeypatch.setattr(mod, "resolve_gpu_preferences", lambda primary, fallbacks=None: (primary,))

    mod.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--data-dir",
            str(data_dir),
            "--symbols-file",
            str(symbols_file),
            "--no-ensemble",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["symbol_source"] == "symbols_file"
    assert payload["symbols"] == ["AAPL", "MSFT"]
    assert payload["symbol_count"] == 2
    assert payload["removed_duplicate_symbols"] == ["AAPL"]
    assert payload["symbols_file"]["local"] == str(symbols_file)
    assert "Removed duplicate symbols: AAPL" in payload["warnings"]
    assert "--symbols-file" in payload["planner_command_preview"]
    assert str(symbols_file) not in payload["planner_command_preview"]


def test_main_dry_run_rejects_symbols_file_without_valid_symbols(monkeypatch, tmp_path, capsys) -> None:
    mod = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("# comment only\n\n", encoding="utf-8")

    monkeypatch.setattr(mod, "RunPodClient", lambda: (_ for _ in ()).throw(AssertionError("should not create pod")))

    with pytest.raises(SystemExit, match="1"):
        mod.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                str(data_dir),
                "--symbols-file",
                str(symbols_file),
                "--no-ensemble",
                "--dry-run",
            ]
        )

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["symbols"] == []
    assert payload["symbol_count"] == 0
    assert payload["errors"] == [f"Invalid symbols file {symbols_file}: No valid symbols found in {symbols_file}"]


def test_main_dry_run_text_includes_errors_for_invalid_plan(monkeypatch, tmp_path, capsys) -> None:
    mod = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("# comment only\n\n", encoding="utf-8")

    monkeypatch.setattr(mod, "RunPodClient", lambda: (_ for _ in ()).throw(AssertionError("should not create pod")))

    with pytest.raises(SystemExit, match="1"):
        mod.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data-dir",
                str(data_dir),
                "--symbols-file",
                str(symbols_file),
                "--no-ensemble",
                "--dry-run",
                "--dry-run-text",
            ]
        )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ready"] is False
    assert "Status: not ready" in captured.err
    assert "Errors:" in captured.err
    assert str(symbols_file) in captured.err
