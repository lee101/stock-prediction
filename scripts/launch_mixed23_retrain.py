#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.remote_training_pipeline import (
    DEFAULT_REMOTE_DIR,
    DEFAULT_REMOTE_ENV,
    DEFAULT_REMOTE_HOST,
    build_remote_autoresearch_plan,
    parse_csv_tokens,
    render_remote_pipeline_script,
)

TRAIN_DATA = "pufferlib_market/data/mixed23_fresh_train.bin"
VAL_DATA = "pufferlib_market/data/mixed23_fresh_val.bin"
REPLAY_EVAL_HOURLY_ROOT = "trainingdatahourly"
REPLAY_EVAL_START_DATE = "2025-06-01"
REPLAY_EVAL_END_DATE = "2026-02-05"
DEFAULT_POST_EVAL_PERIODS = (30, 60, 90, 120)

PRESET_DESCRIPTIONS: dict[str, tuple[str, ...]] = {
    "champions": (
        "reg_combo_2",
        "ent_anneal",
        "robust_reg_tp01",
        "robust_reg_tp005_sds02",
        "gspo_like_mix15",
        "gspo_like_drawdown_mix15",
        "per_env_adv_smooth",
    ),
    "robust": (
        "reg_combo_2",
        "robust_reg_tp01",
        "robust_reg_tp005_sds02",
        "robust_reg_tp005_dd002",
        "robust_reg_tp005_ent",
        "robust_reg_h512_tp005",
    ),
    "gspo": (
        "per_env_adv_smooth",
        "gspo_like",
        "gspo_like_mix15",
        "gspo_like_drawdown_mix15",
        "gspo_like_smooth_mix15",
    ),
    "replay": (
        "ent_anneal",
        "reg_combo_2",
        "wd_01",
        "clip_vloss",
        "robust_reg_tp01",
        "robust_reg_tp005_sds02",
    ),
}


def _default_run_id() -> str:
    return time.strftime("mixed23_champions_%Y%m%d_%H%M%S")


def resolve_descriptions(*, preset: str, descriptions: str = "") -> list[str]:
    if descriptions.strip():
        return [str(item).strip() for item in parse_csv_tokens(descriptions) if str(item).strip()]
    return list(PRESET_DESCRIPTIONS[preset])


def _build_rsync_cmd(remote_host: str, remote_dir: str, *, extra_paths: Sequence[str] = ()) -> list[str]:
    paths = [
        "scripts/launch_mixed23_retrain.py",
        "src/remote_training_pipeline.py",
        "pufferlib_market/",
        "AGENTS.md",
        "docs/REMOTE_TRAINING_RUNBOOK.md",
    ]
    for path in extra_paths:
        clean = str(path).strip()
        if clean and clean not in paths:
            paths.append(clean)
    return [
        "rsync",
        "-azR",
        "-e",
        "ssh -o StrictHostKeyChecking=no",
        *paths,
        f"{remote_host}:{remote_dir}/",
    ]


def _build_remote_bootstrap_script(
    *,
    remote_dir: str,
    script_path: str,
    log_path: str,
    pid_path: str,
    pipeline_script: str,
) -> str:
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {json.dumps(str(remote_dir))}",
            f"mkdir -p {json.dumps(str(Path(script_path).parent))}",
            f"mkdir -p {json.dumps(str(Path(log_path).parent))}",
            f"cat > {json.dumps(str(script_path))} <<'__CODEX_REMOTE_PIPELINE__'",
            pipeline_script.rstrip("\n"),
            "__CODEX_REMOTE_PIPELINE__",
            f"chmod +x {json.dumps(str(script_path))}",
            f"nohup bash {json.dumps(str(script_path))} > {json.dumps(str(log_path))} 2>&1 &",
            f"echo $! > {json.dumps(str(pid_path))}",
            f"cat {json.dumps(str(pid_path))}",
        ]
    ) + "\n"


def _write_local_manifest(
    *,
    manifest_dir: Path,
    args: argparse.Namespace,
    plan_payload: dict[str, object],
    pipeline_script: str,
    rsync_cmd: list[str],
) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "pipeline.sh").write_text(pipeline_script)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {key: str(value) for key, value in vars(args).items()},
        "plan": plan_payload,
        "commands": {
            "rsync_push": rsync_cmd,
            "tail_log": [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                str(args.remote_host),
                f"cd {args.remote_dir} && tail -n 80 {plan_payload['remote_log_path']}",
            ],
            "pull_run_dir": [
                "rsync",
                "-az",
                "-e",
                "ssh -o StrictHostKeyChecking=no",
                f"{args.remote_host}:{args.remote_dir}/{plan_payload['remote_run_dir']}/",
                str(manifest_dir) + "/remote_run/",
            ],
            "pull_checkpoints": [
                "rsync",
                "-az",
                "-e",
                "ssh -o StrictHostKeyChecking=no",
                f"{args.remote_host}:{args.remote_dir}/pufferlib_market/checkpoints/{plan_payload['run_id']}/",
                str(manifest_dir) + "/checkpoints/",
            ],
            "pull_leaderboard": [
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                f"{args.remote_host}:{args.remote_dir}/{plan_payload['leaderboard_path']}",
                str(manifest_dir / "leaderboard.csv"),
            ],
        },
    }
    if plan_payload.get("post_eval_output_path"):
        manifest["commands"]["pull_marketsim_csv"] = [
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            f"{args.remote_host}:{args.remote_dir}/{plan_payload['post_eval_output_path']}",
            str(manifest_dir / "marketsim.csv"),
        ]
    manifest_path = manifest_dir / "launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a longer mixed23 daily RL autoresearch run on the remote GPU box.")
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--preset", choices=sorted(PRESET_DESCRIPTIONS), default="champions")
    parser.add_argument("--descriptions", default="", help="Optional comma-separated descriptions override.")
    parser.add_argument("--train-data", default=TRAIN_DATA)
    parser.add_argument("--val-data", default=VAL_DATA)
    parser.add_argument("--holdout-data", default=VAL_DATA)
    parser.add_argument("--time-budget", type=int, default=1800)
    parser.add_argument("--max-trials", type=int, default=0, help="Defaults to the number of selected descriptions.")
    parser.add_argument(
        "--rank-metric",
        choices=[
            "auto",
            "val_return",
            "holdout_robust_score",
            "market_goodness_score",
            "replay_hourly_return_pct",
            "replay_hourly_policy_return_pct",
        ],
        default="replay_hourly_return_pct",
    )
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    parser.add_argument("--max-steps-override", type=int, default=90)
    parser.add_argument("--holdout-eval-steps", type=int, default=90)
    parser.add_argument("--holdout-n-windows", type=int, default=20)
    parser.add_argument("--holdout-fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--holdout-fee-rate", type=float, default=0.001)
    parser.add_argument("--replay-eval-hourly-root", default=REPLAY_EVAL_HOURLY_ROOT)
    parser.add_argument("--replay-eval-start-date", default=REPLAY_EVAL_START_DATE)
    parser.add_argument("--replay-eval-end-date", default=REPLAY_EVAL_END_DATE)
    parser.add_argument("--replay-eval-fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--post-eval-periods", default="30,60,90,120")
    parser.add_argument("--post-eval-sort-period", type=int, default=120)
    parser.add_argument("--post-eval-max-workers", type=int, default=2)
    parser.add_argument("--post-eval-parallel", action="store_true")
    parser.add_argument("--post-eval-use-compile", action="store_true")
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--remote-env", default=DEFAULT_REMOTE_ENV)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    descriptions = resolve_descriptions(preset=str(args.preset), descriptions=str(args.descriptions))
    max_trials = int(args.max_trials) if int(args.max_trials) > 0 else len(descriptions)
    post_eval_periods = [int(value) for value in parse_csv_tokens(args.post_eval_periods, cast=int)] or list(DEFAULT_POST_EVAL_PERIODS)

    plan = build_remote_autoresearch_plan(
        run_id=str(args.run_id),
        train_data_path=str(args.train_data),
        val_data_path=str(args.val_data),
        time_budget=int(args.time_budget),
        max_trials=max_trials,
        descriptions=descriptions,
        rank_metric=str(args.rank_metric),
        periods_per_year=float(args.periods_per_year),
        max_steps_override=int(args.max_steps_override),
        holdout_data=str(args.holdout_data),
        holdout_eval_steps=int(args.holdout_eval_steps),
        holdout_n_windows=int(args.holdout_n_windows),
        holdout_fee_rate=float(args.holdout_fee_rate),
        holdout_fill_buffer_bps=float(args.holdout_fill_buffer_bps),
        replay_eval_data=str(args.holdout_data),
        replay_eval_hourly_root=str(args.replay_eval_hourly_root),
        replay_eval_start_date=str(args.replay_eval_start_date),
        replay_eval_end_date=str(args.replay_eval_end_date),
        replay_eval_fill_buffer_bps=float(args.replay_eval_fill_buffer_bps),
        post_eval_periods=post_eval_periods,
        post_eval_sort_period=int(args.post_eval_sort_period),
        post_eval_max_workers=int(args.post_eval_max_workers),
        post_eval_use_compile=bool(args.post_eval_use_compile),
        post_eval_parallel=bool(args.post_eval_parallel),
    )
    pipeline_script = render_remote_pipeline_script(
        remote_dir=str(args.remote_dir),
        remote_env=str(args.remote_env),
        plan=plan,
    )
    rsync_cmd = _build_rsync_cmd(
        str(args.remote_host),
        str(args.remote_dir),
        extra_paths=[str(args.train_data), str(args.val_data), str(args.holdout_data)],
    )
    manifest_dir = REPO / "analysis" / "remote_runs" / str(args.run_id)
    manifest_path = _write_local_manifest(
        manifest_dir=manifest_dir,
        args=args,
        plan_payload=plan.as_dict(),
        pipeline_script=pipeline_script,
        rsync_cmd=rsync_cmd,
    )

    if args.dry_run:
        print(f"Manifest: {manifest_path}")
        print(pipeline_script)
        return 0

    if not args.no_sync:
        subprocess.run(rsync_cmd, cwd=REPO, check=True)

    bootstrap = _build_remote_bootstrap_script(
        remote_dir=str(args.remote_dir),
        script_path=plan.remote_script_path,
        log_path=plan.remote_log_path,
        pid_path=plan.remote_pid_path,
        pipeline_script=pipeline_script,
    )
    launch = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", str(args.remote_host)],
        input=bootstrap,
        text=True,
        capture_output=True,
        check=True,
    )
    remote_pid = launch.stdout.strip().splitlines()[-1] if launch.stdout.strip() else ""
    print(f"Manifest: {manifest_path}")
    print(f"Remote PID: {remote_pid}")
    print(
        "Tail log:\n"
        f"ssh -o StrictHostKeyChecking=no {args.remote_host} "
        f"'cd {args.remote_dir} && tail -n 80 {plan.remote_log_path}'"
    )
    print(
        "Pull run dir:\n"
        f'rsync -az -e "ssh -o StrictHostKeyChecking=no" '
        f"{args.remote_host}:{args.remote_dir}/{plan.remote_run_dir}/ {manifest_dir}/remote_run/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
