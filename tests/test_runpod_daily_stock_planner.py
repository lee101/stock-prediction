from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "runpod_daily_stock_planner.py"


def _load_module():
    spec = spec_from_file_location("runpod_daily_stock_planner", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_json_and_remote_path_helpers(tmp_path) -> None:
    mod = _load_module()

    payload = mod._extract_json("boot logs\n{\"plan\": {\"direction\": \"hold\"}}\n")
    assert payload == {"plan": {"direction": "hold"}}

    in_repo = mod.REPO / "trade_daily_stock_prod.py"
    assert mod._relative_remote_path(in_repo, "/workspace/stock") == "/workspace/stock/trade_daily_stock_prod.py"

    external = tmp_path / "checkpoint.pt"
    external.write_text("stub", encoding="utf-8")
    assert mod._relative_remote_path(external, "/workspace/stock") == "/workspace/stock/.external/checkpoint.pt"


def test_extract_json_reports_invalid_remote_output() -> None:
    mod = _load_module()

    with pytest.raises(RuntimeError, match="did not emit valid JSON payload"):
        mod._extract_json("boot logs\nnot-json\n")


def test_extract_json_rejects_non_object_payload() -> None:
    mod = _load_module()

    with pytest.raises(RuntimeError, match="JSON payload must be an object"):
        mod._extract_json('["not", "an", "object"]')


def test_rsync_repo_wraps_subprocess_failures(monkeypatch) -> None:
    mod = _load_module()

    def _fake_run(cmd, check=False, text=False, capture_output=False):
        assert check is False
        assert text is True
        assert capture_output is True
        return SimpleNamespace(returncode=23, stdout="partial transfer", stderr="ssh: connection refused")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

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
            self.terminated: list[str] = []

        def create_pod(self, config):
            self.created.append(config)
            return SimpleNamespace(id="pod-123")

        def wait_for_pod(self, pod_id, timeout=0, poll_interval=0):
            return SimpleNamespace(
                id=pod_id,
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
    monkeypatch.setattr(mod, "build_gpu_fallback_types", lambda primary, fallbacks=None: [primary, *(fallbacks or [])])
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
    assert "ALPACA_API_KEY=test-key python trade_daily_stock_prod.py" in planner_cmd
    assert "--dry-run" in planner_cmd
    assert "--print-payload" in planner_cmd
    assert "--checkpoint /workspace/stock-prediction/.external/checkpoint.pt" in planner_cmd
    assert "--data-dir /workspace/stock-prediction/.external/data" in planner_cmd
    assert "--extra-checkpoints" in planner_cmd
    assert "/workspace/stock-prediction/.external/extra_one.pt" in planner_cmd
    assert "/workspace/stock-prediction/.external/extra_two.pt" in planner_cmd
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

        def create_pod(self, config):
            return SimpleNamespace(id="pod-123")

        def wait_for_pod(self, pod_id, timeout=0, poll_interval=0):
            return SimpleNamespace(
                id=pod_id,
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
    monkeypatch.setattr(mod, "build_gpu_fallback_types", lambda primary, fallbacks=None: [primary, *(fallbacks or [])])
    monkeypatch.setattr(mod, "_rsync_repo", lambda **kwargs: None)
    monkeypatch.setattr(mod, "_rsync_path", lambda **kwargs: None)
    monkeypatch.setattr(mod, "_ssh_run", _fake_ssh_run)

    with pytest.raises(RuntimeError, match="did not emit valid JSON payload"):
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
    monkeypatch.setattr(mod, "build_gpu_fallback_types", lambda primary, fallbacks=None: [primary, *(fallbacks or [])])
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
    monkeypatch.setattr(mod, "build_gpu_fallback_types", lambda primary, fallbacks=None: [primary, *(fallbacks or [])])

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
