"""Tests for stocks-mode dispatch in scripts/dispatch_rl_training.py.

Covers:
- --stocks flag parsing and data-path defaults
- --stocks propagation to remote and local autoresearch commands
- bootstrap script includes PufferLib install and C ext compile
- dry-run output shows stocks-mode parameters
- H100/H100-SXM GPU aliases work end-to-end
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Load dispatch_rl_training as a module (it is a script, not a package).
_DISPATCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dispatch_rl_training.py"
_spec = importlib.util.spec_from_file_location("dispatch_rl_training", _DISPATCH_PATH)
dispatch = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(dispatch)  # type: ignore[union-attr]

# Real rate/alias data so tests are isolated from monkeypatching side-effects.
_REAL_HOURLY_RATES = {
    "NVIDIA A100 80GB PCIe": 1.64,
    "NVIDIA A100-SXM4-80GB": 1.94,
    "NVIDIA H100 80GB HBM3": 3.89,
    "NVIDIA H100 SXM": 4.49,
    "NVIDIA GeForce RTX 4090": 0.69,
    "NVIDIA GeForce RTX 5090": 1.25,
}
_REAL_GPU_ALIASES = {
    "a100": "NVIDIA A100 80GB PCIe",
    "a100-sxm": "NVIDIA A100-SXM4-80GB",
    "h100": "NVIDIA H100 80GB HBM3",
    "h100-sxm": "NVIDIA H100 SXM",
    "4090": "NVIDIA GeForce RTX 4090",
    "5090": "NVIDIA GeForce RTX 5090",
}


# ---------------------------------------------------------------------------
# parse_args — --stocks flag
# ---------------------------------------------------------------------------


def test_parse_args_stocks_flag_present() -> None:
    args = dispatch.parse_args([
        "--data-train", "train.bin",
        "--data-val", "val.bin",
        "--stocks",
    ])
    assert args.stocks is True


def test_parse_args_stocks_flag_absent() -> None:
    args = dispatch.parse_args([
        "--data-train", "train.bin",
        "--data-val", "val.bin",
    ])
    assert args.stocks is False


def test_parse_args_stocks_makes_data_paths_optional() -> None:
    """--stocks should allow omitting --data-train and --data-val."""
    args = dispatch.parse_args(["--stocks"])
    assert args.stocks is True
    assert args.data_train is None
    assert args.data_val is None


def test_parse_args_stocks_explicit_paths_override_defaults() -> None:
    args = dispatch.parse_args([
        "--stocks",
        "--data-train", "custom_train.bin",
        "--data-val", "custom_val.bin",
    ])
    assert args.data_train == "custom_train.bin"
    assert args.data_val == "custom_val.bin"


def test_parse_args_holdout_n_windows_passthrough() -> None:
    args = dispatch.parse_args([
        "--stocks",
        "--holdout-n-windows", "0",
    ])
    assert args.holdout_n_windows == 0


def test_parse_args_eval_num_episodes_passthrough() -> None:
    args = dispatch.parse_args([
        "--stocks",
        "--eval-num-episodes", "12",
    ])
    assert args.eval_num_episodes == 12


# ---------------------------------------------------------------------------
# main() — stocks-mode applies default data paths
# ---------------------------------------------------------------------------


def test_main_stocks_fills_default_data_paths(monkeypatch) -> None:
    """When --stocks is passed without data paths, main() fills in the defaults."""
    recorded_args: list[argparse.Namespace] = []

    def _fake_dry_run(args, *, remote, seeds=None):
        recorded_args.append(args)

    monkeypatch.setattr(dispatch, "print_dry_run_plan", _fake_dry_run)
    monkeypatch.setattr(dispatch, "should_run_remote", lambda _: False)

    dispatch.main(["--stocks", "--dry-run"])

    assert recorded_args, "print_dry_run_plan should have been called"
    args = recorded_args[0]
    assert args.data_train == dispatch._STOCKS_DEFAULT_TRAIN
    assert args.data_val == dispatch._STOCKS_DEFAULT_VAL


# ---------------------------------------------------------------------------
# _build_remote_autoresearch_cmd — --stocks forwarded
# ---------------------------------------------------------------------------


def test_build_remote_cmd_includes_stocks_flag() -> None:
    args = argparse.Namespace(
        data_train="pufferlib_market/data/stocks12_daily_train.bin",
        data_val="pufferlib_market/data/stocks12_daily_val.bin",
        time_budget=300,
        max_trials=50,
        wandb_project="stock",
        descriptions="",
        holdout_n_windows=None,
        stocks=True,
    )
    cmd = dispatch._build_remote_autoresearch_cmd(
        args, "/workspace/stock-prediction", "lb.csv", "checkpoints"
    )
    assert "--stocks" in cmd


def test_build_remote_cmd_no_stocks_flag_when_absent() -> None:
    args = argparse.Namespace(
        data_train="train.bin",
        data_val="val.bin",
        time_budget=300,
        max_trials=50,
        wandb_project="stock",
        descriptions="",
        holdout_n_windows=None,
        stocks=False,
    )
    cmd = dispatch._build_remote_autoresearch_cmd(
        args, "/workspace/stock-prediction", "lb.csv", "checkpoints"
    )
    assert "--stocks" not in cmd


def test_build_remote_cmd_passes_holdout_n_windows() -> None:
    args = argparse.Namespace(
        data_train="train.bin",
        data_val="val.bin",
        time_budget=300,
        max_trials=50,
        wandb_project="stock",
        descriptions="stock_probe_cpu_fast",
        holdout_n_windows=0,
        eval_num_episodes=None,
        stocks=True,
    )
    cmd = dispatch._build_remote_autoresearch_cmd(
        args, "/workspace/stock-prediction", "lb.csv", "checkpoints"
    )
    assert "--holdout-n-windows" in cmd
    assert " 0" in cmd or "'0'" in cmd


def test_build_remote_cmd_passes_eval_num_episodes() -> None:
    args = argparse.Namespace(
        data_train="train.bin",
        data_val="val.bin",
        time_budget=300,
        max_trials=50,
        wandb_project="stock",
        descriptions="stock_trade_pen_05",
        holdout_n_windows=None,
        eval_num_episodes=20,
        stocks=True,
    )
    cmd = dispatch._build_remote_autoresearch_cmd(
        args, "/workspace/stock-prediction", "lb.csv", "checkpoints"
    )
    assert "--eval-num-episodes-override" in cmd
    assert " 20" in cmd or "'20'" in cmd


# ---------------------------------------------------------------------------
# run_local — --stocks forwarded to subprocess
# ---------------------------------------------------------------------------


def test_run_local_passes_stocks_to_subprocess(monkeypatch) -> None:
    captured_cmds: list[list[str]] = []

    class _FakeResult:
        returncode = 0

    def _fake_run(cmd, *, cwd=None, **kwargs):
        captured_cmds.append(list(cmd))
        return _FakeResult()

    monkeypatch.setattr(dispatch.subprocess, "run", _fake_run)

    args = argparse.Namespace(
        run_id="test_stocks",
        data_train=dispatch._STOCKS_DEFAULT_TRAIN,
        data_val=dispatch._STOCKS_DEFAULT_VAL,
        time_budget=300,
        max_trials=50,
        leaderboard="",
        checkpoint_dir="",
        wandb_project="stock",
        descriptions="",
        holdout_n_windows=None,
        stocks=True,
    )
    dispatch.run_local(args)
    assert captured_cmds, "subprocess.run should have been called"
    cmd = captured_cmds[0]
    assert "--stocks" in cmd


def test_run_local_no_stocks_when_not_set(monkeypatch) -> None:
    captured_cmds: list[list[str]] = []

    class _FakeResult:
        returncode = 0

    def _fake_run(cmd, *, cwd=None, **kwargs):
        captured_cmds.append(list(cmd))
        return _FakeResult()

    monkeypatch.setattr(dispatch.subprocess, "run", _fake_run)

    args = argparse.Namespace(
        run_id="test_no_stocks",
        data_train="train.bin",
        data_val="val.bin",
        time_budget=300,
        max_trials=50,
        leaderboard="",
        checkpoint_dir="",
        wandb_project="stock",
        descriptions="",
        holdout_n_windows=None,
        stocks=False,
    )
    dispatch.run_local(args)
    cmd = captured_cmds[0]
    assert "--stocks" not in cmd


def test_run_local_passes_holdout_n_windows(monkeypatch) -> None:
    captured_cmds: list[list[str]] = []

    class _FakeResult:
        returncode = 0

    def _fake_run(cmd, *, cwd=None, **kwargs):
        captured_cmds.append(list(cmd))
        return _FakeResult()

    monkeypatch.setattr(dispatch.subprocess, "run", _fake_run)

    args = argparse.Namespace(
        run_id="test_probe",
        data_train=dispatch._STOCKS_DEFAULT_TRAIN,
        data_val=dispatch._STOCKS_DEFAULT_VAL,
        time_budget=300,
        max_trials=50,
        leaderboard="",
        checkpoint_dir="",
        wandb_project="stock",
        descriptions="stock_probe_cpu_fast",
        holdout_n_windows=0,
        eval_num_episodes=None,
        stocks=True,
    )

    dispatch.run_local(args)
    cmd = captured_cmds[0]
    idx = cmd.index("--holdout-n-windows")
    assert cmd[idx + 1] == "0"


def test_run_local_passes_eval_num_episodes(monkeypatch) -> None:
    captured_cmds: list[list[str]] = []

    class _FakeResult:
        returncode = 0

    def _fake_run(cmd, *, cwd=None, **kwargs):
        captured_cmds.append(list(cmd))
        return _FakeResult()

    monkeypatch.setattr(dispatch.subprocess, "run", _fake_run)

    args = argparse.Namespace(
        run_id="test_probe_eval",
        data_train=dispatch._STOCKS_DEFAULT_TRAIN,
        data_val=dispatch._STOCKS_DEFAULT_VAL,
        time_budget=300,
        max_trials=50,
        leaderboard="",
        checkpoint_dir="",
        wandb_project="stock",
        descriptions="stock_trade_pen_05",
        holdout_n_windows=None,
        eval_num_episodes=20,
        stocks=True,
    )

    dispatch.run_local(args)
    cmd = captured_cmds[0]
    idx = cmd.index("--eval-num-episodes-override")
    assert cmd[idx + 1] == "20"


def test_run_remote_retries_with_fallback_gpu_on_capacity_error(monkeypatch, tmp_path: Path) -> None:
    created_gpu_types: list[str] = []
    terminated: list[str] = []

    class _FakeClient:
        def create_pod(self, config):
            created_gpu_types.append(config.gpu_type)
            if len(created_gpu_types) == 1:
                raise RuntimeError("This machine does not have the resources to deploy your pod")
            return SimpleNamespace(id="pod-ok")

        def wait_for_pod(self, pod_id, timeout=0):
            return SimpleNamespace(id=pod_id, ssh_host="1.2.3.4", ssh_port=22)

        def terminate_pod(self, pod_id):
            terminated.append(pod_id)

    def _fake_ssh_run(*args, **kwargs):
        return SimpleNamespace(returncode=0)

    def _fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dispatch, "RunPodClient", lambda: _FakeClient())
    monkeypatch.setattr(
        dispatch,
        "build_gpu_fallback_types",
        lambda primary: ["NVIDIA GeForce RTX 4090", "NVIDIA A40"],
    )
    monkeypatch.setattr(dispatch, "_rsync_to_pod", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "_rsync_data_file", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "_ssh_run", _fake_ssh_run)
    monkeypatch.setattr(dispatch, "_scp_from_pod", lambda *a, **k: True)
    monkeypatch.setattr(dispatch, "_upload_to_r2", lambda *a, **k: None)
    monkeypatch.setattr(dispatch.subprocess, "run", _fake_run)

    leaderboard = tmp_path / "leaderboard.csv"
    checkpoint_dir = tmp_path / "checkpoints"
    args = argparse.Namespace(
        run_id="fallback_test",
        data_train="train.bin",
        data_val="val.bin",
        time_budget=60,
        max_trials=2,
        gpu_type="4090",
        wandb_project="stock",
        checkpoint_dir=str(checkpoint_dir),
        leaderboard=str(leaderboard),
        descriptions="",
        budget_limit=5.0,
        stocks=True,
    )

    result = dispatch.run_remote(args)

    assert result == 0
    assert created_gpu_types == ["NVIDIA GeForce RTX 4090", "NVIDIA A40"]
    assert terminated == ["pod-ok"]


def test_run_remote_terminates_unready_pod_before_retry(monkeypatch, tmp_path: Path) -> None:
    created_gpu_types: list[str] = []
    terminated: list[str] = []

    class _FakeClient:
        def create_pod(self, config):
            created_gpu_types.append(config.gpu_type)
            pod_id = "pod-timeout" if len(created_gpu_types) == 1 else "pod-ready"
            return SimpleNamespace(id=pod_id)

        def wait_for_pod(self, pod_id, timeout=0):
            if pod_id == "pod-timeout":
                raise TimeoutError("ssh never appeared")
            return SimpleNamespace(id=pod_id, ssh_host="1.2.3.4", ssh_port=22)

        def terminate_pod(self, pod_id):
            terminated.append(pod_id)

    def _fake_ssh_run(*args, **kwargs):
        return SimpleNamespace(returncode=0)

    def _fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dispatch, "RunPodClient", lambda: _FakeClient())
    monkeypatch.setattr(
        dispatch,
        "build_gpu_fallback_types",
        lambda primary: ["NVIDIA RTX 6000 Ada Generation", "NVIDIA A40"],
    )
    monkeypatch.setattr(dispatch, "_rsync_to_pod", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "_rsync_data_file", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "_ssh_run", _fake_ssh_run)
    monkeypatch.setattr(dispatch, "_scp_from_pod", lambda *a, **k: True)
    monkeypatch.setattr(dispatch, "_upload_to_r2", lambda *a, **k: None)
    monkeypatch.setattr(dispatch.subprocess, "run", _fake_run)

    leaderboard = tmp_path / "leaderboard.csv"
    checkpoint_dir = tmp_path / "checkpoints"
    args = argparse.Namespace(
        run_id="timeout_retry_test",
        data_train="train.bin",
        data_val="val.bin",
        time_budget=60,
        max_trials=2,
        gpu_type="6000-ada",
        wandb_project="stock",
        checkpoint_dir=str(checkpoint_dir),
        leaderboard=str(leaderboard),
        descriptions="",
        budget_limit=5.0,
        stocks=True,
    )

    result = dispatch.run_remote(args)

    assert result == 0
    assert created_gpu_types == ["NVIDIA RTX 6000 Ada Generation", "NVIDIA A40"]
    assert terminated == ["pod-timeout", "pod-ready"]


# ---------------------------------------------------------------------------
# print_dry_run_plan — stocks mode shows parameters
# ---------------------------------------------------------------------------


def test_dry_run_stocks_shows_mode_and_params(monkeypatch, capsys) -> None:
    monkeypatch.setattr(dispatch, "HOURLY_RATES", _REAL_HOURLY_RATES)
    monkeypatch.setattr(dispatch, "GPU_ALIASES", _REAL_GPU_ALIASES)
    monkeypatch.setattr(dispatch, "resolve_gpu_type", lambda alias: _REAL_GPU_ALIASES.get(alias, alias))

    args = argparse.Namespace(
        run_id="stocks_dry",
        data_train=dispatch._STOCKS_DEFAULT_TRAIN,
        data_val=dispatch._STOCKS_DEFAULT_VAL,
        time_budget=300,
        max_trials=50,
        gpu_type="a100",
        wandb_project="stock",
        checkpoint_dir="",
        leaderboard="",
        descriptions="",
        force_remote=False,
        vram_threshold_gb=16.0,
        budget_limit=5.0,
        stocks=True,
        seeds=[42],
    )
    dispatch.print_dry_run_plan(args, remote=True)
    captured = capsys.readouterr()

    assert "Mode:" in captured.out
    assert "stocks" in captured.out
    assert "fee_rate" in captured.out
    assert str(dispatch._STOCKS_FEE_RATE) in captured.out
    assert "max_steps" in captured.out
    assert str(dispatch._STOCKS_MAX_STEPS) in captured.out
    assert "periods/year" in captured.out
    assert str(int(dispatch._STOCKS_PERIODS_PER_YEAR)) in captured.out
    assert "holdout_eval_steps" in captured.out
    assert str(dispatch._STOCKS_HOLDOUT_EVAL_STEPS) in captured.out


def test_dry_run_non_stocks_shows_crypto_mode(monkeypatch, capsys) -> None:
    monkeypatch.setattr(dispatch, "HOURLY_RATES", _REAL_HOURLY_RATES)
    monkeypatch.setattr(dispatch, "GPU_ALIASES", _REAL_GPU_ALIASES)
    monkeypatch.setattr(dispatch, "resolve_gpu_type", lambda alias: _REAL_GPU_ALIASES.get(alias, alias))

    args = argparse.Namespace(
        run_id="crypto_dry",
        data_train="train.bin",
        data_val="val.bin",
        time_budget=300,
        max_trials=50,
        gpu_type="a100",
        wandb_project="stock",
        checkpoint_dir="",
        leaderboard="",
        descriptions="",
        force_remote=False,
        vram_threshold_gb=16.0,
        budget_limit=5.0,
        stocks=False,
        seeds=[42],
    )
    dispatch.print_dry_run_plan(args, remote=True)
    captured = capsys.readouterr()

    assert "Mode:" in captured.out
    assert "crypto" in captured.out
    # Stocks-specific params should NOT appear
    assert "fee_rate" not in captured.out


def test_dry_run_stocks_shows_bootstrap_pufferlib(monkeypatch, capsys) -> None:
    """Dry-run output should mention PufferLib install in the bootstrap step."""
    monkeypatch.setattr(dispatch, "HOURLY_RATES", _REAL_HOURLY_RATES)
    monkeypatch.setattr(dispatch, "GPU_ALIASES", _REAL_GPU_ALIASES)
    monkeypatch.setattr(dispatch, "resolve_gpu_type", lambda alias: _REAL_GPU_ALIASES.get(alias, alias))

    args = argparse.Namespace(
        run_id="stocks_bootstrap",
        data_train=dispatch._STOCKS_DEFAULT_TRAIN,
        data_val=dispatch._STOCKS_DEFAULT_VAL,
        time_budget=300,
        max_trials=50,
        gpu_type="a100",
        wandb_project="stock",
        checkpoint_dir="",
        leaderboard="",
        descriptions="",
        force_remote=False,
        vram_threshold_gb=16.0,
        budget_limit=5.0,
        stocks=True,
        seeds=[42],
    )
    dispatch.print_dry_run_plan(args, remote=True)
    captured = capsys.readouterr()

    assert "PufferLib" in captured.out
    assert "build_ext" in captured.out


# ---------------------------------------------------------------------------
# H100 / H100-SXM GPU aliases
# ---------------------------------------------------------------------------


def test_h100_alias_in_gpu_aliases() -> None:
    from src.runpod_client import GPU_ALIASES
    assert "h100" in GPU_ALIASES
    assert GPU_ALIASES["h100"] == "NVIDIA H100 80GB HBM3"


def test_h100_sxm_alias_in_gpu_aliases() -> None:
    from src.runpod_client import GPU_ALIASES
    assert "h100-sxm" in GPU_ALIASES
    assert GPU_ALIASES["h100-sxm"] == "NVIDIA H100 SXM"


def test_h100_in_hourly_rates() -> None:
    from src.runpod_client import HOURLY_RATES
    assert "NVIDIA H100 80GB HBM3" in HOURLY_RATES
    assert HOURLY_RATES["NVIDIA H100 80GB HBM3"] > 0


def test_h100_sxm_in_hourly_rates() -> None:
    from src.runpod_client import HOURLY_RATES
    assert "NVIDIA H100 SXM" in HOURLY_RATES
    assert HOURLY_RATES["NVIDIA H100 SXM"] > 0


def test_resolve_h100_alias() -> None:
    from src.runpod_client import resolve_gpu_type
    assert resolve_gpu_type("h100") == "NVIDIA H100 80GB HBM3"


def test_resolve_h100_sxm_alias() -> None:
    from src.runpod_client import resolve_gpu_type
    assert resolve_gpu_type("h100-sxm") == "NVIDIA H100 SXM"


# ---------------------------------------------------------------------------
# Bootstrap script content (gpu_pool_rl.bootstrap_pod)
# ---------------------------------------------------------------------------


def test_bootstrap_pod_script_includes_pufferlib_install(monkeypatch) -> None:
    """bootstrap_pod should install PufferLib/ if present."""
    from pufferlib_market import gpu_pool_rl

    captured_scripts: list[str] = []

    def _fake_ssh_exec(pod, script: str) -> MagicMock:
        captured_scripts.append(script)
        result = MagicMock()
        result.stdout = "BOOTSTRAP_OK"
        result.stderr = ""
        result.returncode = 0
        return result

    # Fake rsync so it doesn't actually run
    def _fake_run(cmd, *, capture_output=False, text=False, timeout=None, **kwargs) -> MagicMock:
        r = MagicMock()
        r.returncode = 0
        r.stdout = ""
        r.stderr = ""
        return r

    monkeypatch.setattr(gpu_pool_rl, "ssh_exec", _fake_ssh_exec)
    monkeypatch.setattr(gpu_pool_rl.subprocess, "run", _fake_run)

    pod = gpu_pool_rl.PoolPod(
        pod_id="test-pod",
        name="rl-a100-1",
        gpu_type="NVIDIA A100 80GB PCIe",
        gpu_count=1,
        status="ready",
        ssh_host="1.2.3.4",
        ssh_port=22222,
    )
    gpu_pool_rl.bootstrap_pod(pod, repo_root=Path("/tmp/fake-repo"))

    assert captured_scripts, "ssh_exec should have been called with the setup script"
    script = captured_scripts[0]
    assert "PufferLib" in script, "bootstrap should install PufferLib/"
    assert "uv pip install -e PufferLib/" in script
    assert "build_ext --inplace" in script


def test_bootstrap_pod_script_includes_c_ext_build(monkeypatch) -> None:
    """bootstrap_pod script must compile the pufferlib_market C extension."""
    from pufferlib_market import gpu_pool_rl

    captured_scripts: list[str] = []

    def _fake_ssh_exec(pod, script: str) -> MagicMock:
        captured_scripts.append(script)
        result = MagicMock()
        result.stdout = "BOOTSTRAP_OK"
        result.stderr = ""
        result.returncode = 0
        return result

    def _fake_run(cmd, **kwargs) -> MagicMock:
        r = MagicMock()
        r.returncode = 0
        r.stdout = ""
        r.stderr = ""
        return r

    monkeypatch.setattr(gpu_pool_rl, "ssh_exec", _fake_ssh_exec)
    monkeypatch.setattr(gpu_pool_rl.subprocess, "run", _fake_run)

    pod = gpu_pool_rl.PoolPod(
        pod_id="test-pod2",
        name="rl-h100-1",
        gpu_type="NVIDIA H100 80GB HBM3",
        gpu_count=1,
        status="ready",
        ssh_host="5.6.7.8",
        ssh_port=12345,
    )
    gpu_pool_rl.bootstrap_pod(pod, repo_root=Path("/tmp/fake-repo"))

    script = captured_scripts[0]
    assert "setup.py build_ext --inplace" in script


def test_bootstrap_pod_raises_on_failure(monkeypatch) -> None:
    """bootstrap_pod must raise RuntimeError when BOOTSTRAP_OK is missing from stdout."""
    from pufferlib_market import gpu_pool_rl

    def _fake_ssh_exec(pod, script: str) -> MagicMock:
        result = MagicMock()
        result.stdout = "some error output, no sentinel"
        result.stderr = "ImportError: something"
        result.returncode = 1
        return result

    def _fake_run(cmd, **kwargs) -> MagicMock:
        r = MagicMock()
        r.returncode = 0
        return r

    monkeypatch.setattr(gpu_pool_rl, "ssh_exec", _fake_ssh_exec)
    monkeypatch.setattr(gpu_pool_rl.subprocess, "run", _fake_run)

    pod = gpu_pool_rl.PoolPod(
        pod_id="fail-pod",
        name="rl-a100-fail",
        gpu_type="NVIDIA A100 80GB PCIe",
        gpu_count=1,
        status="ready",
        ssh_host="1.1.1.1",
        ssh_port=22,
    )
    with pytest.raises(RuntimeError, match="Bootstrap failed"):
        gpu_pool_rl.bootstrap_pod(pod, repo_root=Path("/tmp/fake-repo"))


# ---------------------------------------------------------------------------
# Stocks constants have the right values
# ---------------------------------------------------------------------------


def test_stocks_constants_values() -> None:
    assert dispatch._STOCKS_FEE_RATE == 0.001
    assert dispatch._STOCKS_MAX_STEPS == 252
    assert dispatch._STOCKS_PERIODS_PER_YEAR == 252.0
    assert dispatch._STOCKS_HOLDOUT_EVAL_STEPS == 90
    assert "stocks12_daily_train" in dispatch._STOCKS_DEFAULT_TRAIN
    assert "stocks12_daily_val" in dispatch._STOCKS_DEFAULT_VAL
