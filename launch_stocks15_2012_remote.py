#!/usr/bin/env python3
"""Launch stocks15_2012 RL autoresearch on a remote RTX 4090 RunPod pod.

Explores 15-symbol universe (vs 11 in the base launcher) with the full
2012-2025 training history.  More symbols = more opportunity for the agent
to exploit cross-symbol patterns (crypto12 > crypto8 > crypto5 precedent).

Data: stocks15_daily_{train,val}_2012.bin
  train: 15 syms × 4840 days (2012-06-01 to 2025-08-31) = 72,600 samples
  val:   15 syms × 201 days  (2025-09-01 to 2026-03-20)

Usage:
    # Dry run — inspect generated script without SSH
    python launch_stocks15_2012_remote.py --dry-run --gpu-type rtx4090

    # Real launch to 4090 RunPod pod
    python launch_stocks15_2012_remote.py --gpu-type rtx4090 --max-trials 500

    # Pull leaderboard back
    rsync -az -e "ssh -o StrictHostKeyChecking=no" \\
        <host>:<remote_dir>/autoresearch_stocks15_2012_leaderboard.csv ./

    # Pull top checkpoints
    rsync -az -e "ssh -o StrictHostKeyChecking=no" \\
        <host>:<remote_dir>/pufferlib_market/checkpoints/autoresearch_stocks15/ ./
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

# stocks15_2012: 15 symbols from 2012-06-01, same val window as stocks11_2012
TRAIN_DATA = "pufferlib_market/data/stocks15_daily_train_2012.bin"   # 15 × 4840 days
VAL_DATA   = "pufferlib_market/data/stocks15_daily_val_2012.bin"     # 15 × 201 days

FEE_OVERRIDE        = 0.001   # Alpaca 10bps
HOLDOUT_EVAL_STEPS  = 90      # fits in 201-day val set
HOLDOUT_N_WINDOWS   = 20
MAX_STEPS_OVERRIDE  = 252     # daily steps per year
PERIODS_PER_YEAR    = 252.0
LEADERBOARD_NAME    = "autoresearch_stocks15_2012_leaderboard.csv"
CHECKPOINT_ROOT     = "pufferlib_market/checkpoints/autoresearch_stocks15"

# Time budget per trial (seconds).
# stocks15_2012: 200 timesteps × 72,600 samples = 14.52M steps
# RTX 4090 ~92k sps → ~158s; +60% headroom = 255s
# H100 ~400k sps → ~36s; +60% headroom = 60s (but eval adds ~30s → 90s)
_GPU_TYPE_TIME_BUDGETS: dict[str, int] = {
    "h100":    120,  # ~400k sps → 36s training + ~60s eval overhead = 120s cap
    "a100":    150,
    "a40":     175,
    "rtx5090": 255,  # ~92k sps → 158s + headroom
    "rtx4090": 255,  # same throughput class as RTX 5090 for this workload
    "default": 255,
}


def _default_run_id() -> str:
    return time.strftime("stocks15_autoresearch_%Y%m%d_%H%M%S")


def _build_rsync_cmd(remote_host: str, remote_dir: str) -> list[str]:
    paths = [
        "launch_stocks15_2012_remote.py",
        "launch_stocks_autoresearch_remote.py",  # shares helper functions via import
        "src/remote_training_pipeline.py",
        "pufferlib_market/",
        "AGENTS.md",
        TRAIN_DATA,
        VAL_DATA,
    ]
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
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "stocks15_2012",
        "args": {key: str(value) for key, value in vars(args).items()},
        "plan": plan_payload,
        "commands": {
            "rsync_push": rsync_cmd,
            "tail_log": [
                "ssh", "-o", "StrictHostKeyChecking=no", str(args.remote_host),
                f"tail -f {args.remote_dir}/{plan_payload['remote_log_path']}",
            ],
            "pull_leaderboard": [
                "rsync", "-az", "-e", "ssh -o StrictHostKeyChecking=no",
                f"{args.remote_host}:{args.remote_dir}/{LEADERBOARD_NAME}",
                str(manifest_path.parent / "stocks15_leaderboard.csv"),
            ],
            "pull_checkpoints": [
                "rsync", "-az", "-e", "ssh -o StrictHostKeyChecking=no",
                f"{args.remote_host}:{args.remote_dir}/{CHECKPOINT_ROOT}/",
                str(manifest_path.parent / "stocks15_checkpoints") + "/",
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch stocks15_2012 RL autoresearch on a remote RTX 4090 RunPod pod."
    )
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--gpu-type", default="rtx4090",
                        help="GPU type alias used to set per-trial time budget.")
    parser.add_argument("--time-budget", type=int, default=0,
                        help="Seconds per trial. 0 = auto from --gpu-type.")
    parser.add_argument("--max-trials", type=int, default=500,
                        help="Number of trials. 500 trials at 255s each ≈ 35h on 4090. "
                             "Run overnight for 8h (~115 trials).")
    parser.add_argument(
        "--rank-metric",
        choices=["auto", "val_return", "holdout_robust_score",
                 "market_goodness_score", "replay_hourly_return_pct"],
        default="holdout_robust_score",
    )
    parser.add_argument("--max-timesteps-per-sample", type=int, default=200,
                        help="Step cap per sample. 200 × 72,600 samples = 14.52M steps. "
                             "RTX 4090 at ~92k sps → ~158s per trial.")
    parser.add_argument("--holdout-eval-steps",    type=int,   default=HOLDOUT_EVAL_STEPS)
    parser.add_argument("--holdout-n-windows",     type=int,   default=HOLDOUT_N_WINDOWS)
    parser.add_argument("--holdout-fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--holdout-fee-rate",      type=float, default=FEE_OVERRIDE)
    parser.add_argument("--fee-rate-override",     type=float, default=FEE_OVERRIDE)
    parser.add_argument("--periods-per-year",      type=float, default=PERIODS_PER_YEAR)
    parser.add_argument("--max-steps-override",    type=int,   default=MAX_STEPS_OVERRIDE)
    parser.add_argument("--start-from", type=int, default=326,
                        help="Start index in STOCK_EXPERIMENTS pool. "
                             "326 = start of random-mutation block (skip deterministic blocks "
                             "tuned for stocks11/stocks12). For stocks15 we go straight to "
                             "seed-only random mutations from the tp05_s123 formula.")
    parser.add_argument("--seed-only", action="store_true", default=True,
                        help="Only change seed, keep all other params from init_best_config. "
                             "Uses tp05_s123 formula (lr=3e-4, wd=0.01, tp=0.05, slip=0bps).")
    parser.add_argument("--init-best-config", default="stock_trade_pen_05_s123",
                        help="Seed the mutation search from this proven config. "
                             "stock_trade_pen_05_s123: h=1024, tp=0.05, anneal_lr, seed=123 "
                             "— best on stocks11/12, likely transfers to stocks15.")
    parser.add_argument("--lock-best-config", action="store_true", default=True,
                        help="Prevent autoresearch from updating best_config after bad trial. "
                             "CRITICAL: without this, one -50 trial can corrupt the search.")
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-dir",  default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--remote-env",  default=DEFAULT_REMOTE_ENV)
    parser.add_argument("--no-sync",  action="store_true")
    parser.add_argument("--dry-run",  action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    time_budget = args.time_budget or _GPU_TYPE_TIME_BUDGETS.get(
        args.gpu_type.lower().strip(), _GPU_TYPE_TIME_BUDGETS["default"]
    )

    plan = build_remote_autoresearch_plan(
        run_id=str(args.run_id),
        train_data_path=TRAIN_DATA,
        val_data_path=VAL_DATA,
        time_budget=time_budget,
        max_trials=int(args.max_trials),
        descriptions=[],
        rank_metric=str(args.rank_metric),
        periods_per_year=float(args.periods_per_year),
        max_steps_override=int(args.max_steps_override),
        max_timesteps_per_sample=int(args.max_timesteps_per_sample),
        fee_rate_override=float(args.fee_rate_override),
        holdout_data=VAL_DATA,
        holdout_eval_steps=int(args.holdout_eval_steps),
        holdout_n_windows=int(args.holdout_n_windows),
        holdout_fee_rate=float(args.holdout_fee_rate),
        holdout_fill_buffer_bps=float(args.holdout_fill_buffer_bps),
        leaderboard_path=LEADERBOARD_NAME,
        checkpoint_root=CHECKPOINT_ROOT,
        stocks_mode=True,
        start_from=int(args.start_from),
        seed_only=bool(args.seed_only),
        init_best_config=str(args.init_best_config),
        lock_best_config=bool(args.lock_best_config),
    )
    pipeline_script = render_remote_pipeline_script(
        remote_dir=str(args.remote_dir),
        remote_env=str(args.remote_env),
        plan=plan,
    )
    rsync_cmd = _build_rsync_cmd(str(args.remote_host), str(args.remote_dir))
    manifest_path = REPO / f"manifest_stocks15_{args.run_id}.json"
    _write_local_manifest(
        manifest_path=manifest_path,
        args=args,
        plan_payload=plan.as_dict(),
        pipeline_script=pipeline_script,
        rsync_cmd=rsync_cmd,
    )

    if args.dry_run:
        print(f"Dataset:  stocks15_2012 ({TRAIN_DATA})")
        print(f"Budget:   {time_budget}s/trial × {args.max_trials} trials")
        print(f"Manifest: {manifest_path}")
        print()
        print(pipeline_script[:3000])
        return 0

    if not args.no_sync:
        print(f"Syncing code + data to {args.remote_host}:{args.remote_dir} ...")
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
        input=bootstrap, text=True, capture_output=True, check=True,
    )
    remote_pid = launch.stdout.strip().splitlines()[-1] if launch.stdout.strip() else ""
    print(f"Manifest: {manifest_path}")
    print(f"Remote PID: {remote_pid}")
    print(
        f"\nTail log:\n"
        f"  ssh -o StrictHostKeyChecking=no {args.remote_host} "
        f"'tail -f {args.remote_dir}/{plan.remote_log_path}'"
    )
    print(
        f"\nPull leaderboard:\n"
        f"  rsync -az -e 'ssh -o StrictHostKeyChecking=no' "
        f"{args.remote_host}:{args.remote_dir}/{LEADERBOARD_NAME} "
        f"autoresearch_stocks15_2012_leaderboard.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
