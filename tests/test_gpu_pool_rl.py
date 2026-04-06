"""Tests for pufferlib_market.gpu_pool_rl."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Patch RunPodClient import before loading the module so we don't need network/API key.
# We insert the mock only temporarily: after importing gpu_pool_rl we restore the original
# (or remove the mock) so that test_runpod_client.py can import the real src.runpod_client
# without getting the mock.
import sys
_mock_runpod = MagicMock()
_mock_pod_cls = MagicMock()
_mock_pod_cfg_cls = MagicMock()
_real_runpod_module = sys.modules.get("src.runpod_client")  # None if not yet imported
sys.modules["src.runpod_client"] = _mock_runpod

from pufferlib_market.gpu_pool_rl import (  # noqa: E402
    DEFAULT_POOL_LIMITS,
    HOURLY_RATES,
    SETUP_OVERHEAD_SECS,
    PoolPod,
    PoolState,
    _count_gpu_type,
    _detect_remote_h100,
    _find_available_pod,
    _resolve_gpu,
    bootstrap_pod,
    check_running_cost,
    cmd_run,
    estimate_cost,
    load_pool_state,
    refresh_pod_status,
    save_pool_state,
    cmd_status,
)

# Restore the real src.runpod_client module so other test files see the real module.
if _real_runpod_module is None:
    sys.modules.pop("src.runpod_client", None)
else:
    sys.modules["src.runpod_client"] = _real_runpod_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pod(
    pod_id: str = "pod1",
    name: str = "rl-a100-1",
    gpu_type: str = "NVIDIA A100 80GB PCIe",
    status: str = "ready",
    ssh_host: str = "1.2.3.4",
    ssh_port: int = 22222,
    current_experiment: str = "",
) -> PoolPod:
    return PoolPod(
        pod_id=pod_id,
        name=name,
        gpu_type=gpu_type,
        gpu_count=1,
        status=status,
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        created_at="2026-01-01T00:00:00+00:00",
        current_experiment=current_experiment,
    )


# ---------------------------------------------------------------------------
# load_pool_state
# ---------------------------------------------------------------------------

def test_load_pool_state_returns_empty_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent" / "pool.json"
    with patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", missing):
        state = load_pool_state()
    assert isinstance(state, PoolState)
    assert state.pods == {}
    assert state.total_experiments_run == 0
    # limits should match defaults
    assert state.limits == dict(DEFAULT_POOL_LIMITS)


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.json"
    pod = _make_pod()
    state = PoolState(
        pods={"rl-a100-1": pod},
        limits={"NVIDIA A100 80GB PCIe": 3},
        total_experiments_run=7,
    )

    with patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", pool_file):
        save_pool_state(state)
        loaded = load_pool_state()

    assert loaded.total_experiments_run == 7
    assert loaded.limits == {"NVIDIA A100 80GB PCIe": 3}
    assert "rl-a100-1" in loaded.pods
    loaded_pod = loaded.pods["rl-a100-1"]
    assert loaded_pod.pod_id == pod.pod_id
    assert loaded_pod.ssh_host == pod.ssh_host
    assert loaded_pod.ssh_port == pod.ssh_port
    assert loaded_pod.status == pod.status


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    pool_file = tmp_path / "deep" / "nested" / "pool.json"
    state = PoolState()
    with patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", pool_file):
        save_pool_state(state)
    assert pool_file.exists()
    data = json.loads(pool_file.read_text())
    assert "pods" in data


def test_save_pool_state_writes_via_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pufferlib_market import gpu_pool_rl

    pool_file = tmp_path / "pool.json"
    state = PoolState()
    replace_calls: list[tuple[str, str]] = []
    original_replace = os.replace

    def _record_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        replace_calls.append((os.fspath(src), os.fspath(dst)))
        original_replace(src, dst)

    monkeypatch.setattr(gpu_pool_rl.os, "replace", _record_replace)

    with patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", pool_file):
        save_pool_state(state)

    assert replace_calls == [
        (str(tmp_path / "pool.json.tmp"), str(pool_file))
    ]
    assert pool_file.exists()


def test_pool_state_guard_uses_shared_and_exclusive_file_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pufferlib_market import gpu_pool_rl

    pool_file = tmp_path / "pool.json"
    pool_file.write_text(
        json.dumps(
            {
                "pods": {},
                "limits": dict(DEFAULT_POOL_LIMITS),
                "total_experiments_run": 0,
            }
        )
        + "\n"
    )
    operations: list[object] = []

    class _FakeFcntl:
        LOCK_SH = "LOCK_SH"
        LOCK_EX = "LOCK_EX"
        LOCK_UN = "LOCK_UN"

        @staticmethod
        def flock(_fileno: int, operation: object) -> None:
            operations.append(operation)

    monkeypatch.setattr(gpu_pool_rl, "_fcntl", _FakeFcntl)

    with patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", pool_file):
        load_pool_state()
        save_pool_state(PoolState())

    assert operations == [
        _FakeFcntl.LOCK_SH,
        _FakeFcntl.LOCK_UN,
        _FakeFcntl.LOCK_EX,
        _FakeFcntl.LOCK_UN,
    ]


# ---------------------------------------------------------------------------
# _count_gpu_type
# ---------------------------------------------------------------------------

def test_count_gpu_type_excludes_dead_and_stopped() -> None:
    gpu = "NVIDIA A100 80GB PCIe"
    state = PoolState(pods={
        "pod-ready": _make_pod(pod_id="p1", name="pod-ready", status="ready"),
        "pod-busy": _make_pod(pod_id="p2", name="pod-busy", status="busy"),
        "pod-dead": _make_pod(pod_id="p3", name="pod-dead", status="dead"),
        "pod-stopped": _make_pod(pod_id="p4", name="pod-stopped", status="stopped"),
        "pod-prov": _make_pod(pod_id="p5", name="pod-prov", status="provisioning"),
    })
    assert _count_gpu_type(state, gpu) == 3  # ready + busy + provisioning


def test_count_gpu_type_only_counts_matching_type() -> None:
    state = PoolState(pods={
        "a100": _make_pod(pod_id="p1", name="a100", gpu_type="NVIDIA A100 80GB PCIe", status="ready"),
        "h100": _make_pod(pod_id="p2", name="h100", gpu_type="NVIDIA H100 80GB HBM3", status="ready"),
    })
    assert _count_gpu_type(state, "NVIDIA A100 80GB PCIe") == 1
    assert _count_gpu_type(state, "NVIDIA H100 80GB HBM3") == 1
    assert _count_gpu_type(state, "NVIDIA GeForce RTX 4090") == 0


def test_count_gpu_type_empty_pool() -> None:
    state = PoolState()
    assert _count_gpu_type(state, "NVIDIA A100 80GB PCIe") == 0


# ---------------------------------------------------------------------------
# _find_available_pod
# ---------------------------------------------------------------------------

def test_find_available_pod_returns_ready_pod() -> None:
    gpu = "NVIDIA A100 80GB PCIe"
    ready_pod = _make_pod(pod_id="p1", name="pod-ready", status="ready")
    state = PoolState(pods={"pod-ready": ready_pod})
    result = _find_available_pod(state, gpu)
    assert result is ready_pod


def test_find_available_pod_returns_none_when_all_busy() -> None:
    gpu = "NVIDIA A100 80GB PCIe"
    state = PoolState(pods={
        "pod-busy": _make_pod(pod_id="p1", name="pod-busy", status="busy"),
    })
    assert _find_available_pod(state, gpu) is None


def test_find_available_pod_returns_none_when_empty() -> None:
    assert _find_available_pod(PoolState(), "NVIDIA A100 80GB PCIe") is None


def test_find_available_pod_ignores_wrong_gpu_type() -> None:
    state = PoolState(pods={
        "h100-pod": _make_pod(
            pod_id="p1", name="h100-pod",
            gpu_type="NVIDIA H100 80GB HBM3",
            status="ready",
        )
    })
    assert _find_available_pod(state, "NVIDIA A100 80GB PCIe") is None


# ---------------------------------------------------------------------------
# _resolve_gpu
# ---------------------------------------------------------------------------

def test_resolve_gpu_known_alias() -> None:
    assert _resolve_gpu("a100") == "NVIDIA A100 80GB PCIe"
    assert _resolve_gpu("h100") == "NVIDIA H100 80GB HBM3"
    assert _resolve_gpu("4090") == "NVIDIA GeForce RTX 4090"


def test_resolve_gpu_passthrough_unknown() -> None:
    assert _resolve_gpu("NVIDIA Some Unknown GPU") == "NVIDIA Some Unknown GPU"


# ---------------------------------------------------------------------------
# refresh_pod_status
# ---------------------------------------------------------------------------

def test_refresh_pod_status_marks_missing_pods_dead_and_removes_them() -> None:
    client = MagicMock()
    client.list_pods.return_value = []  # no live pods

    state = PoolState(pods={"rl-a100-1": _make_pod()})
    refresh_pod_status(state, client)

    # Dead pod should be removed
    assert "rl-a100-1" not in state.pods


def test_refresh_pod_status_marks_non_running_as_stopped() -> None:
    live_pod = MagicMock()
    live_pod.id = "pod1"
    live_pod.status = "EXITED"

    client = MagicMock()
    client.list_pods.return_value = [live_pod]

    pool_pod = _make_pod(pod_id="pod1", status="ready")
    state = PoolState(pods={"rl-a100-1": pool_pod})
    refresh_pod_status(state, client)

    assert state.pods["rl-a100-1"].status == "stopped"


def test_refresh_pod_status_updates_ssh_and_marks_ready() -> None:
    live_pod = MagicMock()
    live_pod.id = "pod1"
    live_pod.status = "RUNNING"

    full_pod = MagicMock()
    full_pod.ssh_host = "5.6.7.8"
    full_pod.ssh_port = 33333

    client = MagicMock()
    client.list_pods.return_value = [live_pod]
    client.get_pod.return_value = full_pod

    pool_pod = _make_pod(pod_id="pod1", status="provisioning", ssh_host="", ssh_port=0)
    state = PoolState(pods={"rl-a100-1": pool_pod})
    refresh_pod_status(state, client)

    assert state.pods["rl-a100-1"].status == "ready"
    assert state.pods["rl-a100-1"].ssh_host == "5.6.7.8"
    assert state.pods["rl-a100-1"].ssh_port == 33333


def test_refresh_pod_status_warns_on_api_error(capsys: pytest.CaptureFixture) -> None:
    client = MagicMock()
    client.list_pods.side_effect = RuntimeError("network error")

    state = PoolState(pods={"rl-a100-1": _make_pod()})
    refresh_pod_status(state, client)  # must not raise

    captured = capsys.readouterr()
    assert "Warning" in captured.out


# ---------------------------------------------------------------------------
# cmd_status — empty pool
# ---------------------------------------------------------------------------

def test_cmd_status_empty_pool(capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    """When a state file exists but has no pods, show 'Pool is empty'."""
    pool_file = tmp_path / "pool.json"

    # Write an empty state so the file EXISTS but has no pods.
    empty_state = PoolState()
    with patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", pool_file):
        save_pool_state(empty_state)

    mock_client = MagicMock()
    mock_client.list_pods.return_value = []

    args = SimpleNamespace()

    with (
        patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", pool_file),
        patch("pufferlib_market.gpu_pool_rl._require_client", return_value=mock_client),
    ):
        cmd_status(args)

    captured = capsys.readouterr()
    assert "Pool is empty" in captured.out
    assert "Limits" in captured.out


def test_cmd_status_with_pods(capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.json"

    pod = _make_pod(status="busy", current_experiment="test_run")
    state = PoolState(pods={"rl-a100-1": pod}, total_experiments_run=3)

    mock_client = MagicMock()
    # Return live pod so it stays in state
    live = MagicMock()
    live.id = "pod1"
    live.status = "RUNNING"
    full = MagicMock()
    full.ssh_host = pod.ssh_host
    full.ssh_port = pod.ssh_port
    mock_client.list_pods.return_value = [live]
    mock_client.get_pod.return_value = full

    args = SimpleNamespace()

    with (
        patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", pool_file),
        patch("pufferlib_market.gpu_pool_rl._require_client", return_value=mock_client),
    ):
        save_pool_state(state)
        cmd_status(args)

    captured = capsys.readouterr()
    assert "rl-a100-1" in captured.out
    assert "test_run" in captured.out


# ---------------------------------------------------------------------------
# bootstrap_pod — subprocess mocking
# ---------------------------------------------------------------------------

def test_bootstrap_pod_success(tmp_path: Path) -> None:
    pod = _make_pod()

    rsync_result = MagicMock()
    rsync_result.returncode = 0

    ssh_result = MagicMock()
    ssh_result.stdout = "some output\nBOOTSTRAP_OK\n"
    ssh_result.returncode = 0

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [rsync_result, ssh_result]
        bootstrap_pod(pod, repo_root=tmp_path, remote_dir="/workspace/stock-prediction")

    assert mock_run.call_count == 2
    # First call should be rsync
    first_call_args = mock_run.call_args_list[0][0][0]
    assert first_call_args[0] == "rsync"
    # Excludes data and checkpoints directories
    assert "pufferlib_market/data/" in first_call_args
    assert "pufferlib_market/checkpoints/" in first_call_args


def test_bootstrap_pod_rsync_failure(tmp_path: Path) -> None:
    pod = _make_pod()

    rsync_result = MagicMock()
    rsync_result.returncode = 1
    rsync_result.stderr = "rsync: error"

    with patch("subprocess.run", return_value=rsync_result):
        with pytest.raises(RuntimeError, match="rsync failed"):
            bootstrap_pod(pod, repo_root=tmp_path, remote_dir="/workspace/stock-prediction")


def test_bootstrap_pod_rsync_exit_23_is_ok(tmp_path: Path) -> None:
    """Exit code 23 (partial transfer) is treated as success for rsync."""
    pod = _make_pod()

    rsync_result = MagicMock()
    rsync_result.returncode = 23

    ssh_result = MagicMock()
    ssh_result.stdout = "BOOTSTRAP_OK\n"
    ssh_result.returncode = 0

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [rsync_result, ssh_result]
        # Should not raise
        bootstrap_pod(pod, repo_root=tmp_path, remote_dir="/workspace/stock-prediction")


def test_bootstrap_pod_setup_failure(tmp_path: Path) -> None:
    pod = _make_pod()

    rsync_result = MagicMock()
    rsync_result.returncode = 0

    ssh_result = MagicMock()
    ssh_result.stdout = "something went wrong\n"
    ssh_result.stderr = "error detail"
    ssh_result.returncode = 1

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [rsync_result, ssh_result]
        with pytest.raises(RuntimeError, match="Bootstrap failed"):
            bootstrap_pod(pod, repo_root=tmp_path, remote_dir="/workspace/stock-prediction")


# ---------------------------------------------------------------------------
# ssh_exec — no timeout
# ---------------------------------------------------------------------------

def test_ssh_exec_does_not_pass_timeout() -> None:
    """ssh_exec must not pass a timeout= kwarg to subprocess.run."""
    from pufferlib_market.gpu_pool_rl import ssh_exec

    pod = _make_pod()
    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        ssh_exec(pod, "echo hello")

    call_kwargs = mock_run.call_args[1]
    assert "timeout" not in call_kwargs, "ssh_exec must not set a timeout — training may be long"


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

def test_estimate_cost_a100_known_rate() -> None:
    """estimate_cost uses HOURLY_RATES and includes setup overhead."""
    gpu = "NVIDIA A100 80GB PCIe"
    rate = HOURLY_RATES.get(gpu, 0)
    cost = estimate_cost(gpu, num_seeds=1, time_budget_secs=300)
    expected = rate * ((SETUP_OVERHEAD_SECS + 300) / 3600)
    assert abs(cost - expected) < 1e-9


def test_estimate_cost_alias_resolves() -> None:
    """estimate_cost resolves short aliases like 'a100'."""
    cost_alias = estimate_cost("a100", num_seeds=1, time_budget_secs=300)
    cost_full = estimate_cost("NVIDIA A100 80GB PCIe", num_seeds=1, time_budget_secs=300)
    assert abs(cost_alias - cost_full) < 1e-9


def test_estimate_cost_scales_with_seeds() -> None:
    """More seeds → higher cost."""
    cost1 = estimate_cost("NVIDIA A100 80GB PCIe", num_seeds=1, time_budget_secs=300)
    cost2 = estimate_cost("NVIDIA A100 80GB PCIe", num_seeds=2, time_budget_secs=300)
    assert cost2 > cost1


def test_estimate_cost_zero_for_unknown_gpu() -> None:
    """Unknown GPU type → 0 cost (rate = 0)."""
    cost = estimate_cost("NVIDIA RTX 9999 Unknown", num_seeds=1, time_budget_secs=300)
    assert cost == 0.0


# ---------------------------------------------------------------------------
# check_running_cost
# ---------------------------------------------------------------------------

def test_check_running_cost_no_warning_for_ready_pod(capsys: pytest.CaptureFixture) -> None:
    """Ready pods are not checked — only busy pods trigger warnings."""
    pod = _make_pod(status="ready")
    state = PoolState(pods={"pod1": pod})
    check_running_cost(state, budget_limit=1.0)
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


def test_check_running_cost_no_warning_within_budget(capsys: pytest.CaptureFixture) -> None:
    """Busy pod within budget should not trigger a warning."""
    pod = _make_pod(status="busy")
    # created_at is far in the future relative to now — effectively 0h elapsed
    pod.created_at = "9999-01-01T00:00:00+00:00"
    state = PoolState(pods={"pod1": pod})
    check_running_cost(state, budget_limit=100.0)
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


def test_check_running_cost_warns_over_budget(capsys: pytest.CaptureFixture) -> None:
    """Busy pod that has been running for a very long time should trigger a warning."""
    pod = _make_pod(status="busy")
    # Set created_at far in the past so elapsed time is huge.
    pod.created_at = "2000-01-01T00:00:00+00:00"
    state = PoolState(pods={"pod1": pod})
    check_running_cost(state, budget_limit=1.0)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


def test_check_running_cost_skips_when_limit_zero(capsys: pytest.CaptureFixture) -> None:
    """budget_limit=0 disables the check."""
    pod = _make_pod(status="busy")
    pod.created_at = "2000-01-01T00:00:00+00:00"
    state = PoolState(pods={"pod1": pod})
    check_running_cost(state, budget_limit=0)
    captured = capsys.readouterr()
    assert "WARNING" not in captured.out


# ---------------------------------------------------------------------------
# cmd_run budget enforcement
# ---------------------------------------------------------------------------

def test_cmd_run_refuses_when_over_budget(tmp_path: Path) -> None:
    """cmd_run raises SystemExit when estimated cost exceeds budget_limit."""
    from pufferlib_market.gpu_pool_rl import cmd_run

    pool_file = tmp_path / "pool.json"
    mock_client = MagicMock()
    mock_client.list_pods.return_value = []

    # Use a100 @ $1.64/hr with tiny budget: should exceed $0.01
    args = SimpleNamespace(
        gpu="a100",
        gpu_count=1,
        time_budget=300,
        max_trials=50,
        experiment="test_exp",
        wandb_project="stock",
        remote_dir="/workspace/stock-prediction",
        stop_after=False,
        budget_limit=0.01,  # tiny limit
    )

    with (
        patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", pool_file),
        patch("pufferlib_market.gpu_pool_rl._require_client", return_value=mock_client),
        pytest.raises(SystemExit),
    ):
        cmd_run(args)


def test_cmd_run_allows_when_within_budget(tmp_path: Path) -> None:
    """cmd_run proceeds when estimated cost is within budget_limit."""
    pool_file = tmp_path / "pool.json"
    mock_client = MagicMock()
    mock_client.list_pods.return_value = []

    ready_pod = _make_pod(status="ready")

    args = SimpleNamespace(
        gpu="a100",
        gpu_count=1,
        time_budget=300,
        max_trials=50,
        experiment="test_exp",
        wandb_project="stock",
        remote_dir="/workspace/stock-prediction",
        stop_after=False,
        budget_limit=100.0,  # generous limit
        train_data="pufferlib_market/data/train.bin",
        val_data="pufferlib_market/data/val.bin",
        dry_run=False,
        descriptions="",
    )

    bootstrap_called = []

    def mock_bootstrap(pod, *, repo_root, remote_dir):
        bootstrap_called.append(True)

    def mock_get_or_create(state, client, gpu_type, gpu_count=1):
        return ready_pod

    with (
        patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", pool_file),
        patch("pufferlib_market.gpu_pool_rl._require_client", return_value=mock_client),
        patch("pufferlib_market.gpu_pool_rl.get_or_create_pod", side_effect=mock_get_or_create),
        patch("pufferlib_market.gpu_pool_rl.bootstrap_pod", side_effect=mock_bootstrap),
        patch("pufferlib_market.gpu_pool_rl.sync_data_files"),
        patch("pufferlib_market.gpu_pool_rl.run_rl_experiment_on_pod", return_value={"status": "completed"}),
    ):
        cmd_run(args)

    assert bootstrap_called, "bootstrap_pod should have been called"


def test_cmd_run_no_limit_when_budget_zero(tmp_path: Path) -> None:
    """budget_limit=0 disables the budget check and proceeds to provision."""
    pool_file = tmp_path / "pool.json"
    mock_client = MagicMock()
    mock_client.list_pods.return_value = []

    ready_pod = _make_pod(status="ready")

    args = SimpleNamespace(
        gpu="h100",
        gpu_count=1,
        time_budget=3600,
        max_trials=50,
        experiment="test_exp",
        wandb_project="stock",
        remote_dir="/workspace/stock-prediction",
        stop_after=False,
        budget_limit=0,  # disable limit
        train_data="pufferlib_market/data/train.bin",
        val_data="pufferlib_market/data/val.bin",
        dry_run=False,
        descriptions="",
    )

    with (
        patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", pool_file),
        patch("pufferlib_market.gpu_pool_rl._require_client", return_value=mock_client),
        patch("pufferlib_market.gpu_pool_rl.get_or_create_pod", return_value=ready_pod),
        patch("pufferlib_market.gpu_pool_rl.bootstrap_pod"),
        patch("pufferlib_market.gpu_pool_rl.sync_data_files"),
        patch("pufferlib_market.gpu_pool_rl.run_rl_experiment_on_pod", return_value={"status": "completed"}),
    ):
        # Should not raise SystemExit even though h100 @ $3.89/hr with 1h budget is expensive
        cmd_run(args)


def test_cmd_run_forwards_replay_eval_arguments(tmp_path: Path) -> None:
    pool_file = tmp_path / "pool.json"
    mock_client = MagicMock()
    mock_client.list_pods.return_value = []
    ready_pod = _make_pod(status="ready")
    captured: dict[str, object] = {}

    args = SimpleNamespace(
        gpu="a100",
        gpu_count=1,
        time_budget=300,
        max_trials=10,
        experiment="test_exp",
        wandb_project="stock",
        remote_dir="/workspace/stock-prediction",
        stop_after=False,
        budget_limit=100.0,
        train_data="pufferlib_market/data/train.bin",
        val_data="pufferlib_market/data/val.bin",
        dry_run=False,
        descriptions="baseline_anneal_lr",
        replay_eval_data="pufferlib_market/data/replay.bin",
        replay_eval_hourly_root="trainingdatahourly",
        replay_eval_start_date="2025-06-01",
        replay_eval_end_date="2026-02-05",
        replay_eval_run_hourly_policy=True,
        replay_eval_robust_start_states="flat,long:BTCUSD:0.25",
        replay_eval_fill_buffer_bps=5.0,
        replay_eval_hourly_periods_per_year=8760.0,
    )

    def _fake_run_rl_experiment_on_pod(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "completed"}

    with (
        patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", pool_file),
        patch("pufferlib_market.gpu_pool_rl._require_client", return_value=mock_client),
        patch("pufferlib_market.gpu_pool_rl.get_or_create_pod", return_value=ready_pod),
        patch("pufferlib_market.gpu_pool_rl.bootstrap_pod"),
        patch("pufferlib_market.gpu_pool_rl.sync_data_files"),
        patch("pufferlib_market.gpu_pool_rl.run_rl_experiment_on_pod", side_effect=_fake_run_rl_experiment_on_pod),
    ):
        cmd_run(args)

    assert captured["replay_eval_data"] == "pufferlib_market/data/replay.bin"
    assert captured["replay_eval_hourly_root"] == "trainingdatahourly"
    assert captured["replay_eval_start_date"] == "2025-06-01"
    assert captured["replay_eval_end_date"] == "2026-02-05"
    assert captured["replay_eval_run_hourly_policy"] is True
    assert captured["replay_eval_robust_start_states"] == "flat,long:BTCUSD:0.25"
    assert captured["replay_eval_fill_buffer_bps"] == 5.0
    assert captured["replay_eval_hourly_periods_per_year"] == 8760.0


# ---------------------------------------------------------------------------
# cmd_status — no network when no state file
# ---------------------------------------------------------------------------

def test_cmd_status_no_state_file_prints_no_pool_state(
    capsys: pytest.CaptureFixture, tmp_path: Path
) -> None:
    """When no state file exists, status prints 'No pool state found' without hitting network."""
    pool_file = tmp_path / "does_not_exist.json"
    args = SimpleNamespace()

    with patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", pool_file):
        cmd_status(args)

    captured = capsys.readouterr()
    assert "No pool state found" in captured.out
    # _require_client must NOT have been called (no API key needed)


# ---------------------------------------------------------------------------
# cmd_run --dry-run
# ---------------------------------------------------------------------------

def test_cmd_run_dry_run_prints_plan(capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    """--dry-run prints what would happen without calling _require_client or provisioning."""
    pool_file = tmp_path / "pool.json"

    args = SimpleNamespace(
        gpu="a100",
        gpu_count=1,
        time_budget=600,
        max_trials=50,
        experiment="dry_exp",
        wandb_project="stock",
        remote_dir="/workspace/stock-prediction",
        stop_after=False,
        budget_limit=10.0,
        train_data="pufferlib_market/data/train.bin",
        val_data="pufferlib_market/data/val.bin",
        dry_run=True,
        descriptions="",
    )

    require_client_called = []

    def _fake_require_client():
        require_client_called.append(True)
        return MagicMock()

    with (
        patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", pool_file),
        patch("pufferlib_market.gpu_pool_rl._require_client", side_effect=_fake_require_client),
    ):
        cmd_run(args)

    assert not require_client_called, "_require_client must NOT be called during dry-run"
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.out
    assert "dry_exp" in captured.out
    assert "NVIDIA A100 80GB PCIe" in captured.out
    assert "No pod provisioned" in captured.out


def test_cmd_run_dry_run_warns_over_budget(capsys: pytest.CaptureFixture, tmp_path: Path) -> None:
    """--dry-run still warns when estimated cost exceeds budget, but does not exit."""
    pool_file = tmp_path / "pool.json"

    args = SimpleNamespace(
        gpu="h100",
        gpu_count=1,
        time_budget=7200,
        max_trials=50,
        experiment="big_exp",
        wandb_project="stock",
        remote_dir="/workspace/stock-prediction",
        stop_after=False,
        budget_limit=0.01,  # tiny budget → should warn
        train_data="pufferlib_market/data/train.bin",
        val_data="pufferlib_market/data/val.bin",
        dry_run=True,
        descriptions="",
    )

    with patch("pufferlib_market.gpu_pool_rl.POOL_STATE_FILE", pool_file):
        # Must not raise SystemExit
        cmd_run(args)

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "No pod provisioned" in captured.out


# ---------------------------------------------------------------------------
# _detect_remote_h100
# ---------------------------------------------------------------------------

def test_detect_remote_h100_true_when_h100_in_name() -> None:
    pod = _make_pod()
    result = MagicMock()
    result.returncode = 0
    result.stdout = "NVIDIA H100 80GB HBM3\n"

    with patch("pufferlib_market.gpu_pool_rl.ssh_exec", return_value=result):
        assert _detect_remote_h100(pod) is True


def test_detect_remote_h100_false_when_a100() -> None:
    pod = _make_pod()
    result = MagicMock()
    result.returncode = 0
    result.stdout = "NVIDIA A100 80GB PCIe\n"

    with patch("pufferlib_market.gpu_pool_rl.ssh_exec", return_value=result):
        assert _detect_remote_h100(pod) is False


def test_detect_remote_h100_false_on_error() -> None:
    pod = _make_pod()
    result = MagicMock()
    result.returncode = 1
    result.stdout = ""

    with patch("pufferlib_market.gpu_pool_rl.ssh_exec", return_value=result):
        assert _detect_remote_h100(pod) is False
