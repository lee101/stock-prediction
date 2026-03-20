from __future__ import annotations

import json
from pathlib import Path

from scripts.launch_mixed23_retrain import (
    PRESET_DESCRIPTIONS,
    TRAIN_DATA,
    VAL_DATA,
    _build_rsync_cmd,
    _write_local_manifest,
    parse_args,
    resolve_descriptions,
)


def test_resolve_descriptions_uses_champions_preset_by_default() -> None:
    assert resolve_descriptions(preset="champions") == list(PRESET_DESCRIPTIONS["champions"])


def test_resolve_descriptions_honors_override_order() -> None:
    assert resolve_descriptions(preset="champions", descriptions="foo, bar,baz") == ["foo", "bar", "baz"]


def test_parse_args_defaults_to_longer_budget_and_replay_rank() -> None:
    args = parse_args([])

    assert args.time_budget == 1800
    assert args.rank_metric == "replay_hourly_return_pct"
    assert args.max_steps_override == 90
    assert args.train_data == TRAIN_DATA
    assert args.val_data == VAL_DATA


def test_build_rsync_cmd_targets_remote_repo() -> None:
    cmd = _build_rsync_cmd(
        "user@example.com",
        "/remote/repo",
        extra_paths=["pufferlib_market/data/mixed23_fresh_train.bin"],
    )

    assert cmd[:4] == ["rsync", "-azR", "-e", "ssh -o StrictHostKeyChecking=no"]
    assert "scripts/launch_mixed23_retrain.py" in cmd
    assert "src/remote_training_pipeline.py" in cmd
    assert "pufferlib_market/fast_marketsim_eval.py" in cmd
    assert "pufferlib_market/data/mixed23_fresh_train.bin" in cmd
    assert cmd[-1] == "user@example.com:/remote/repo/"


def test_write_local_manifest_records_marketsim_pull(tmp_path: Path) -> None:
    args = parse_args(["--run-id", "demo123"])
    manifest_path = _write_local_manifest(
        manifest_dir=tmp_path / "demo123",
        args=args,
        plan_payload={
            "remote_log_path": "analysis/remote_runs/demo123/pipeline.log",
            "remote_run_dir": "analysis/remote_runs/demo123",
            "leaderboard_path": "pufferlib_market/demo123_leaderboard.csv",
            "post_eval_output_path": "analysis/remote_runs/demo123/marketsim_30_60_90_120.csv",
        },
        pipeline_script="#!/usr/bin/env bash\n",
        rsync_cmd=["rsync", "dummy"],
    )

    payload = json.loads(manifest_path.read_text())
    assert payload["commands"]["rsync_push"] == ["rsync", "dummy"]
    assert payload["commands"]["pull_marketsim_csv"][-2] == (
        "administrator@93.127.141.100:/nvme0n1-disk/code/stock-prediction/"
        "analysis/remote_runs/demo123/marketsim_30_60_90_120.csv"
    )
