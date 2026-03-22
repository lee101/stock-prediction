"""Tests for launch_stocks_autoresearch_remote.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from launch_stocks_autoresearch_remote import (
    CHECKPOINT_ROOT,
    FEE_OVERRIDE,
    HOLDOUT_EVAL_STEPS,
    HOLDOUT_N_WINDOWS,
    LEADERBOARD_NAME,
    MAX_STEPS_OVERRIDE,
    PERIODS_PER_YEAR,
    TRAIN_DATA,
    VAL_DATA,
    _build_rsync_cmd,
    _write_local_manifest,
    parse_args,
)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.time_budget == 300
    assert args.max_trials == 200
    assert args.gpu_type == "h100"
    assert args.train_data == TRAIN_DATA
    assert args.val_data == VAL_DATA
    assert args.holdout_eval_steps == HOLDOUT_EVAL_STEPS
    assert args.holdout_n_windows == HOLDOUT_N_WINDOWS
    assert args.max_steps_override == MAX_STEPS_OVERRIDE
    assert args.periods_per_year == PERIODS_PER_YEAR
    assert args.fee_rate_override == FEE_OVERRIDE
    assert args.dry_run is False


def test_parse_args_dry_run_flag() -> None:
    args = parse_args(["--dry-run"])
    assert args.dry_run is True


def test_parse_args_gpu_type_a100() -> None:
    args = parse_args(["--gpu-type", "a100"])
    assert args.gpu_type == "a100"


def test_parse_args_max_trials_override() -> None:
    args = parse_args(["--max-trials", "5"])
    assert args.max_trials == 5


def test_parse_args_run_id_override() -> None:
    args = parse_args(["--run-id", "mytest_run"])
    assert args.run_id == "mytest_run"


# ---------------------------------------------------------------------------
# _build_rsync_cmd
# ---------------------------------------------------------------------------


def test_build_rsync_cmd_structure() -> None:
    cmd = _build_rsync_cmd("user@host", "/remote/repo")
    assert cmd[0] == "rsync"
    assert "-azR" in cmd
    assert "-e" in cmd
    assert "ssh -o StrictHostKeyChecking=no" in cmd
    assert cmd[-1] == "user@host:/remote/repo/"


def test_build_rsync_cmd_includes_launcher_and_src() -> None:
    cmd = _build_rsync_cmd("user@host", "/remote/repo")
    joined = " ".join(cmd)
    assert "launch_stocks_autoresearch_remote.py" in joined
    assert "src/remote_training_pipeline.py" in joined
    assert "pufferlib_market/" in joined


def test_build_rsync_cmd_extra_paths() -> None:
    cmd = _build_rsync_cmd(
        "user@host",
        "/remote/repo",
        extra_paths=["pufferlib_market/data/stocks12_daily_train.bin"],
    )
    assert "pufferlib_market/data/stocks12_daily_train.bin" in cmd


def test_build_rsync_cmd_no_duplicate_extra_paths() -> None:
    cmd = _build_rsync_cmd(
        "user@host",
        "/remote/repo",
        extra_paths=["pufferlib_market/", "pufferlib_market/"],
    )
    # pufferlib_market/ already in default paths — should not appear twice
    assert cmd.count("pufferlib_market/") == 1


# ---------------------------------------------------------------------------
# _write_local_manifest
# ---------------------------------------------------------------------------


def test_write_local_manifest_creates_file(tmp_path: Path) -> None:
    args = parse_args(["--run-id", "testrun001"])
    manifest_path = tmp_path / "manifest_stocks_testrun001.json"
    result = _write_local_manifest(
        manifest_path=manifest_path,
        args=args,
        plan_payload={
            "run_id": "testrun001",
            "remote_log_path": "analysis/remote_runs/testrun001/pipeline.log",
            "remote_script_path": "analysis/remote_runs/testrun001/pipeline.sh",
            "remote_run_dir": "analysis/remote_runs/testrun001",
            "leaderboard_path": LEADERBOARD_NAME,
            "post_eval_output_path": None,
        },
        pipeline_script="#!/usr/bin/env bash\necho hello\n",
        rsync_cmd=["rsync", "-azR", "dummy"],
    )
    assert result == manifest_path
    assert manifest_path.exists()


def test_write_local_manifest_has_required_keys(tmp_path: Path) -> None:
    args = parse_args(["--run-id", "testrun002"])
    manifest_path = tmp_path / "manifest.json"
    _write_local_manifest(
        manifest_path=manifest_path,
        args=args,
        plan_payload={
            "run_id": "testrun002",
            "remote_log_path": "analysis/remote_runs/testrun002/pipeline.log",
            "remote_script_path": "analysis/remote_runs/testrun002/pipeline.sh",
            "remote_run_dir": "analysis/remote_runs/testrun002",
            "leaderboard_path": LEADERBOARD_NAME,
            "post_eval_output_path": None,
        },
        pipeline_script="#!/usr/bin/env bash\necho hello\n",
        rsync_cmd=["rsync", "dummy"],
    )
    payload = json.loads(manifest_path.read_text())
    assert "generated_at" in payload
    assert "args" in payload
    assert "plan" in payload
    assert "commands" in payload
    cmds = payload["commands"]
    assert "rsync_push" in cmds
    assert "run_script" in cmds
    assert "tail_log" in cmds
    assert "pull_checkpoints" in cmds
    assert "pull_leaderboard" in cmds


def test_write_local_manifest_pull_checkpoints_uses_correct_root(tmp_path: Path) -> None:
    args = parse_args(["--run-id", "testrun003"])
    manifest_path = tmp_path / "manifest.json"
    _write_local_manifest(
        manifest_path=manifest_path,
        args=args,
        plan_payload={
            "run_id": "testrun003",
            "remote_log_path": "analysis/remote_runs/testrun003/pipeline.log",
            "remote_script_path": "analysis/remote_runs/testrun003/pipeline.sh",
            "remote_run_dir": "analysis/remote_runs/testrun003",
            "leaderboard_path": LEADERBOARD_NAME,
            "post_eval_output_path": None,
        },
        pipeline_script="#!/usr/bin/env bash\necho hello\n",
        rsync_cmd=["rsync", "dummy"],
    )
    payload = json.loads(manifest_path.read_text())
    pull_cmd = payload["commands"]["pull_checkpoints"]
    assert any(CHECKPOINT_ROOT in token for token in pull_cmd)


def test_write_local_manifest_pipeline_sh_written(tmp_path: Path) -> None:
    args = parse_args(["--run-id", "testrun004"])
    manifest_path = tmp_path / "manifest.json"
    _write_local_manifest(
        manifest_path=manifest_path,
        args=args,
        plan_payload={
            "run_id": "testrun004",
            "remote_log_path": "analysis/remote_runs/testrun004/pipeline.log",
            "remote_script_path": "analysis/remote_runs/testrun004/pipeline.sh",
            "remote_run_dir": "analysis/remote_runs/testrun004",
            "leaderboard_path": LEADERBOARD_NAME,
            "post_eval_output_path": None,
        },
        pipeline_script="#!/usr/bin/env bash\necho hello\n",
        rsync_cmd=["rsync", "dummy"],
    )
    pipeline_sh = tmp_path / "pipeline.sh"
    assert pipeline_sh.exists()
    assert "echo hello" in pipeline_sh.read_text()


# ---------------------------------------------------------------------------
# End-to-end dry-run via subprocess
# ---------------------------------------------------------------------------


def _run_dry(extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(REPO / "launch_stocks_autoresearch_remote.py"),
        "--dry-run",
        "--run-id",
        "drytest_001",
        "--gpu-type",
        "a100",
        "--max-trials",
        "5",
    ] + (extra_args or [])
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))


def test_dry_run_exits_zero() -> None:
    result = _run_dry()
    assert result.returncode == 0, f"stderr: {result.stderr}"


def test_dry_run_prints_manifest_path() -> None:
    result = _run_dry()
    combined = result.stdout + result.stderr
    assert "Manifest:" in combined
    assert "manifest_stocks_" in combined


def test_dry_run_prints_pipeline_script() -> None:
    result = _run_dry()
    combined = result.stdout + result.stderr
    assert "#!/usr/bin/env bash" in combined
    assert "autoresearch_rl" in combined


def test_dry_run_manifest_file_created() -> None:
    result = _run_dry()
    assert result.returncode == 0, f"stderr: {result.stderr}"
    manifest = REPO / "manifest_stocks_drytest_001.json"
    assert manifest.exists(), f"Manifest not found at {manifest}"
    payload = json.loads(manifest.read_text())
    # All required command keys present
    assert "rsync_push" in payload["commands"]
    assert "run_script" in payload["commands"]
    assert "tail_log" in payload["commands"]
    assert "pull_checkpoints" in payload["commands"]
    assert "pull_leaderboard" in payload["commands"]


def test_dry_run_manifest_stocks_parameters() -> None:
    result = _run_dry()
    assert result.returncode == 0
    manifest = REPO / "manifest_stocks_drytest_001.json"
    payload = json.loads(manifest.read_text())
    plan = payload["plan"]
    assert plan["train_data_path"] == TRAIN_DATA
    assert plan["val_data_path"] == VAL_DATA
    assert plan["leaderboard_path"] == LEADERBOARD_NAME
    assert plan["checkpoint_root"] == CHECKPOINT_ROOT


def test_dry_run_help() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "launch_stocks_autoresearch_remote.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0
    assert "--dry-run" in result.stdout
    assert "--gpu-type" in result.stdout
    assert "--max-trials" in result.stdout
