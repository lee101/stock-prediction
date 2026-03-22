#!/usr/bin/env python3
"""Launch stocks daily RL autoresearch on a remote H100/A100 RunPod pod.

Usage:
    # Dry run (just generate manifest, no actual SSH/rsync)
    python launch_stocks_autoresearch_remote.py --dry-run --gpu-type a100

    # Real launch to H100
    python launch_stocks_autoresearch_remote.py --gpu-type h100 --max-trials 200

    # Pull results back after training
    cat manifest_stocks_TIMESTAMP.json | jq -r '.commands.pull_checkpoints[]' | bash
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.remote_training_pipeline import (
    DEFAULT_REMOTE_DIR,
    DEFAULT_REMOTE_ENV,
    DEFAULT_REMOTE_HOST,
    build_remote_autoresearch_plan,
    render_remote_pipeline_script,
)

TRAIN_DATA = "pufferlib_market/data/stocks11_daily_train_2012.bin"  # 11 syms × 4840 days (2012-06-01 to 2025-08-31)
VAL_DATA = "pufferlib_market/data/stocks11_daily_val_2012.bin"  # 11 syms × 201 days (2025-09-01 to 2026-03-20)
# Cross-feature bins (20 feat/sym: adds rolling_corr, rolling_beta, relative_return, breadth_rank)
TRAIN_DATA_CROSS = "pufferlib_market/data/stocks11_daily_train_2012_cross.bin"
VAL_DATA_CROSS = "pufferlib_market/data/stocks11_daily_val_2012_cross.bin"
FEE_OVERRIDE = 0.001  # Alpaca 10bps
HOLDOUT_EVAL_STEPS = 90  # fits in 201-day val set
HOLDOUT_N_WINDOWS = 20
MAX_STEPS_OVERRIDE = 252  # daily steps per year
PERIODS_PER_YEAR = 252.0
LEADERBOARD_NAME = "autoresearch_stock_daily_leaderboard.csv"
CHECKPOINT_ROOT = "pufferlib_market/checkpoints/autoresearch_stock"


def _default_run_id() -> str:
    return time.strftime("stocks_autoresearch_%Y%m%d_%H%M%S")


def _build_rsync_cmd(remote_host: str, remote_dir: str, *, extra_paths: Sequence[str] = ()) -> list[str]:
    paths = [
        "launch_stocks_autoresearch_remote.py",
        "src/remote_training_pipeline.py",
        "pufferlib_market/",
        "AGENTS.md",
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
    manifest_path: Path,
    args: argparse.Namespace,
    plan_payload: dict[str, object],
    pipeline_script: str,
    rsync_cmd: list[str],
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    (manifest_path.parent / "pipeline.sh").write_text(pipeline_script)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {key: str(value) for key, value in vars(args).items()},
        "plan": plan_payload,
        "commands": {
            "rsync_push": rsync_cmd,
            "run_script": [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                str(args.remote_host),
                f"cd {args.remote_dir} && bash {plan_payload['remote_script_path']}",
            ],
            "tail_log": [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                str(args.remote_host),
                f"tail -f {args.remote_dir}/{plan_payload['remote_log_path']}",
            ],
            "pull_checkpoints": [
                "rsync",
                "-az",
                "-e",
                "ssh -o StrictHostKeyChecking=no",
                f"{args.remote_host}:{args.remote_dir}/{CHECKPOINT_ROOT}/",
                str(manifest_path.parent / "checkpoints") + "/",
            ],
            "pull_leaderboard": [
                "rsync",
                "-az",
                "-e",
                "ssh -o StrictHostKeyChecking=no",
                f"{args.remote_host}:{args.remote_dir}/{LEADERBOARD_NAME}",
                str(manifest_path.parent / "leaderboard.csv"),
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch stocks daily RL autoresearch on a remote H100/A100 RunPod pod."
    )
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--gpu-type", default="h100", help="GPU type alias (h100, a100, etc.)")
    parser.add_argument("--train-data", default=TRAIN_DATA)
    parser.add_argument("--val-data", default=VAL_DATA)
    parser.add_argument("--holdout-data", default=VAL_DATA)
    parser.add_argument("--cross-features", action="store_true",
                        help="Use cross-feature bins (20 feat/sym). Overrides --train-data/--val-data "
                             "to use stocks11_daily_{train,val}_2012_cross.bin. "
                             "Add these bins to rsync before launching.")
    parser.add_argument("--time-budget", type=int, default=120,
                        help="Seconds per trial safety timeout. At H100 ~400k sps: 120s >> 37M step cap (71-96s).")
    parser.add_argument("--max-trials", type=int, default=1000,
                        help="1000 trials: ~8% escape rate → ~80 good models. "
                             "Early rejection (25%% checkpoint) cuts degenerate trials to ~28s each, "
                             "so 1000 trials ≈ 80 × 93s + 920 × 28s = 33,000s ≈ 9 hours on H100.")
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
        default="holdout_robust_score",
    )
    parser.add_argument("--periods-per-year", type=float, default=PERIODS_PER_YEAR)
    parser.add_argument("--max-steps-override", type=int, default=MAX_STEPS_OVERRIDE)
    parser.add_argument("--holdout-eval-steps", type=int, default=HOLDOUT_EVAL_STEPS)
    parser.add_argument("--holdout-n-windows", type=int, default=HOLDOUT_N_WINDOWS)
    parser.add_argument("--holdout-fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--holdout-fee-rate", type=float, default=FEE_OVERRIDE)
    parser.add_argument("--fee-rate-override", type=float, default=FEE_OVERRIDE)
    parser.add_argument("--max-timesteps-per-sample", type=int, default=700,
                        help="Step cap per sample. 700 × 53,240 samples = 37.27M steps for stocks11_2012 (optimal). "
                             "H100 at 400k sps hits this in ~93s; RTX 5090 at 92k sps hits it in ~405s. "
                             "Use 700 for stocks11_2012 (confirmed optimal step count).")
    parser.add_argument("--start-from", type=int, default=187,
                        help="Start index in STOCK_EXPERIMENTS pool. Default 187 = N-block first, then random seeds. "
                             "Indices 187-194: N-block h256 formula (h=256, lr=3e-4, slip=12, dp=0.01). "
                             "Index 195+: random_1..450 → mutate_config(best_config, seed_only). "
                             "If N-block h256 wins best_config, subsequent seeds use h256+lr=1e-4+slip12+dp01. "
                             "Use 195 to skip N-block and go straight to seed sweep.")
    parser.add_argument("--seed-only", action="store_true", default=True,
                        help="In random-mutation mode, only change the seed (keep all other params). "
                             "Default: True. With --start-from 195, all 500 H100 trials use s1137's exact "
                             "config (lr=1e-4, h=1024, ent=0.05, sdp=0.0, anneal=True) with different seeds. "
                             "Evidence: sdp=0.2 hurts good seeds (seed=1464: -37.25 → -56.18). "
                             "Use --start-from 187 to also test N-block h256 configs before random seeds.")
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--remote-env", default=DEFAULT_REMOTE_ENV)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Cross-feature override
    if args.cross_features:
        args.train_data = TRAIN_DATA_CROSS
        args.val_data = VAL_DATA_CROSS
        args.holdout_data = VAL_DATA_CROSS

    plan = build_remote_autoresearch_plan(
        run_id=str(args.run_id),
        train_data_path=str(args.train_data),
        val_data_path=str(args.val_data),
        time_budget=int(args.time_budget),
        max_trials=int(args.max_trials),
        descriptions=[],  # empty: autoresearch_rl random-mutation mode
        rank_metric=str(args.rank_metric),
        periods_per_year=float(args.periods_per_year),
        max_steps_override=int(args.max_steps_override),
        max_timesteps_per_sample=int(args.max_timesteps_per_sample),
        fee_rate_override=float(args.fee_rate_override),
        holdout_data=str(args.holdout_data),
        holdout_eval_steps=int(args.holdout_eval_steps),
        holdout_n_windows=int(args.holdout_n_windows),
        holdout_fee_rate=float(args.holdout_fee_rate),
        holdout_fill_buffer_bps=float(args.holdout_fill_buffer_bps),
        leaderboard_path=LEADERBOARD_NAME,
        checkpoint_root=CHECKPOINT_ROOT,
        stocks_mode=True,  # use STOCK_EXPERIMENTS pool (not crypto EXPERIMENTS)
        start_from=int(args.start_from),
        seed_only=bool(args.seed_only),
    )
    pipeline_script = render_remote_pipeline_script(
        remote_dir=str(args.remote_dir),
        remote_env=str(args.remote_env),
        plan=plan,
    )
    rsync_cmd = _build_rsync_cmd(
        str(args.remote_host),
        str(args.remote_dir),
        extra_paths=[str(args.train_data), str(args.val_data)],
    )
    manifest_path = REPO / f"manifest_stocks_{args.run_id}.json"
    _write_local_manifest(
        manifest_path=manifest_path,
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
        f"'tail -f {args.remote_dir}/{plan.remote_log_path}'"
    )
    print(
        "Pull checkpoints:\n"
        f'rsync -az -e "ssh -o StrictHostKeyChecking=no" '
        f"{args.remote_host}:{args.remote_dir}/{CHECKPOINT_ROOT}/ "
        f"{manifest_path.parent}/checkpoints/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
