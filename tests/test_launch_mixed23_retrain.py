from __future__ import annotations

import json
from pathlib import Path

import launch_mixed23_retrain as compat_launcher
from scripts.launch_mixed23_retrain import (
    DEFAULT_PROD_LAUNCH_SCRIPT,
    PRESET_DESCRIPTIONS,
    TRAIN_DATA,
    VAL_DATA,
    _build_rsync_cmd,
    _write_local_manifest,
    main,
    parse_args,
    resolve_descriptions,
    resolve_eval_constraints,
)


def test_resolve_descriptions_uses_champions_preset_by_default() -> None:
    assert resolve_descriptions(preset="champions") == list(PRESET_DESCRIPTIONS["champions"])


def test_resolve_descriptions_honors_override_order() -> None:
    assert resolve_descriptions(preset="champions", descriptions="foo, bar,baz") == ["foo", "bar", "baz"]


def test_root_launcher_delegates_to_scripts_entrypoint() -> None:
    assert compat_launcher.main is main


def test_parse_args_defaults_to_longer_budget_and_replay_rank() -> None:
    args = parse_args([])

    assert args.time_budget == 1800
    assert args.rank_metric == "replay_hourly_return_pct"
    assert args.max_steps_override == 90
    assert args.train_data == TRAIN_DATA
    assert args.val_data == VAL_DATA
    assert args.replay_eval_run_hourly_policy is False
    assert args.replay_eval_robust_start_states == ""
    assert args.replay_eval_hourly_periods_per_year == 8760.0
    assert args.prod_launch_script == DEFAULT_PROD_LAUNCH_SCRIPT
    assert args.eval_tradable_symbols == ""
    assert args.eval_max_leverage is None
    assert args.eval_disable_shorts is None


def test_parse_args_accepts_robust_replay_rank_metrics() -> None:
    args = parse_args([
        "--rank-metric",
        "replay_hourly_policy_robust_worst_return_pct",
        "--replay-eval-run-hourly-policy",
        "--replay-eval-robust-start-states",
        "flat,long:BTCUSD:0.25",
        "--replay-eval-hourly-periods-per-year",
        "365",
    ])

    assert args.rank_metric == "replay_hourly_policy_robust_worst_return_pct"
    assert args.replay_eval_run_hourly_policy is True
    assert args.replay_eval_robust_start_states == "flat,long:BTCUSD:0.25"
    assert args.replay_eval_hourly_periods_per_year == 365.0


def test_resolve_eval_constraints_defaults_to_prod_launch_config() -> None:
    max_leverage, tradable_symbols, disable_shorts = resolve_eval_constraints(
        prod_launch_script=DEFAULT_PROD_LAUNCH_SCRIPT,
        eval_max_leverage=None,
        eval_tradable_symbols="",
        eval_disable_shorts=None,
    )

    assert max_leverage == 0.5
    assert tradable_symbols == "BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD"
    assert disable_shorts is True


def test_resolve_eval_constraints_honors_explicit_overrides() -> None:
    max_leverage, tradable_symbols, disable_shorts = resolve_eval_constraints(
        prod_launch_script=DEFAULT_PROD_LAUNCH_SCRIPT,
        eval_max_leverage=1.25,
        eval_tradable_symbols="BTCUSD,ETHUSD",
        eval_disable_shorts=False,
    )

    assert max_leverage == 1.25
    assert tradable_symbols == "BTCUSD,ETHUSD"
    assert disable_shorts is False


def test_resolve_eval_constraints_without_prod_launch_keeps_shorts_enabled() -> None:
    max_leverage, tradable_symbols, disable_shorts = resolve_eval_constraints(
        prod_launch_script="",
        eval_max_leverage=None,
        eval_tradable_symbols="",
        eval_disable_shorts=None,
    )

    assert max_leverage == 1.0
    assert tradable_symbols == ""
    assert disable_shorts is False


def test_resolve_eval_constraints_handles_gemini_only_launch(tmp_path: Path) -> None:
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "exec /tmp/.venv/bin/python -u \\",
                "  rl_trading_agent_binance/trade_binance_live.py \\",
                "  --live \\",
                "  --model gemini-3.1-flash-lite-preview \\",
                "  --symbols BTCUSD ETHUSD SOLUSD \\",
                "  --execution-mode margin \\",
                "  --leverage 0.5 \\",
                "  --interval 3600 \\",
                "  --fallback-mode chronos2 \\",
                '  "$@"',
            ]
        )
    )

    max_leverage, tradable_symbols, disable_shorts = resolve_eval_constraints(
        prod_launch_script=str(launch),
        eval_max_leverage=None,
        eval_tradable_symbols="",
        eval_disable_shorts=None,
    )

    assert max_leverage == 0.5
    assert tradable_symbols == "BTCUSD,ETHUSD,SOLUSD"
    assert disable_shorts is False


def test_build_rsync_cmd_targets_remote_repo() -> None:
    cmd = _build_rsync_cmd(
        "user@example.com",
        "/remote/repo",
        extra_paths=["pufferlib_market/data/mixed23_fresh_train.bin"],
    )

    assert cmd[:10] == [
        "rsync",
        "-azR",
        "--exclude",
        "pufferlib_market/checkpoints/",
        "--exclude",
        "pufferlib_market/data/",
        "--exclude",
        "**/__pycache__/",
        "-e",
        "ssh -o StrictHostKeyChecking=no",
    ]
    assert "scripts/launch_mixed23_retrain.py" in cmd
    assert "src/" in cmd
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
