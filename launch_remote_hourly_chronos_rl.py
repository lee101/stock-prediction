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
    build_hourly_train_val_window_from_bounds,
    build_remote_hourly_chronos_rl_plan,
    compute_hourly_overlap_bounds,
    normalize_symbols,
    parse_csv_tokens,
    render_remote_pipeline_script,
)


def _default_run_id() -> str:
    return time.strftime("hourly_chronos_rl_%Y%m%d_%H%M%S")


def _build_rsync_cmd(remote_host: str, remote_dir: str) -> list[str]:
    sync_items = [
        "scripts/",
        "src/",
        "pufferlib_market/",
        "binanceneural/",
        "preaug/",
        "chronos2_trainer.py",
        "docs/",
    ]
    if (REPO / "AGENTS.md").exists():
        sync_items.append("AGENTS.md")
    cmd = [
        "rsync",
        "-az",
        "-e",
        "ssh -S none -o ControlMaster=no -o StrictHostKeyChecking=no",
        *sync_items,
        f"{remote_host}:{remote_dir}/",
    ]
    return cmd


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
            f"cat > {json.dumps(str(script_path))} <<'__CODEx_REMOTE_PIPELINE__'",
            pipeline_script.rstrip("\n"),
            "__CODEx_REMOTE_PIPELINE__",
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
    remote_host: str,
    remote_dir: str,
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
                remote_host,
                f"cd {remote_dir} && tail -n 80 {plan_payload['remote_log_path']}",
            ],
            "pull_run_dir": [
                "rsync",
                "-az",
                "-e",
                "ssh -o StrictHostKeyChecking=no",
                f"{remote_host}:{remote_dir}/{plan_payload['remote_run_dir']}/",
                str(manifest_dir) + "/remote_run/",
            ],
            "pull_checkpoints": [
                "rsync",
                "-az",
                "-e",
                "ssh -o StrictHostKeyChecking=no",
                f"{remote_host}:{remote_dir}/{plan_payload['checkpoint_root']}/",
                str(manifest_dir) + "/checkpoints/",
            ],
            "pull_leaderboard": [
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                f"{remote_host}:{remote_dir}/{plan_payload['leaderboard_path']}",
                str(manifest_dir / "leaderboard.csv"),
            ],
        },
    }
    manifest_path = manifest_dir / "launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def _load_remote_overlap_bounds(
    *,
    remote_host: str,
    remote_dir: str,
    remote_env: str,
    remote_data_root: str,
    symbols: Sequence[str],
) -> tuple[str, str]:
    symbol_list = ",".join(normalize_symbols(symbols))
    snippet = f"""
set -euo pipefail
cd {json.dumps(str(remote_dir))}
source {json.dumps(str(remote_env).rstrip('/'))}/bin/activate
python - <<'PY'
import json
from pathlib import Path
from src.remote_training_pipeline import compute_hourly_overlap_bounds

earliest, latest = compute_hourly_overlap_bounds(
    symbols={symbol_list!r}.split(","),
    data_root=Path({str(remote_data_root)!r}),
)
print(json.dumps({{"earliest_common": earliest, "latest_common": latest}}))
PY
"""
    result = subprocess.run(
        ["ssh", "-S", "none", "-o", "ControlMaster=no", "-o", "StrictHostKeyChecking=no", remote_host, snippet],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return str(payload["earliest_common"]), str(payload["latest_common"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the hourly Chronos2 -> RL pipeline on the 5090 box.")
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument(
        "--symbols",
        default="BTCUSD,ETHUSD,SOLUSD,AVAXUSD,LINKUSD,UNIUSD,SHIBUSD,DOGEUSD,XRPUSD,DOTUSD",
    )
    parser.add_argument("--local-data-root", type=Path, default=Path("trainingdatahourly/crypto"))
    parser.add_argument("--remote-data-root", default="trainingdatahourly/crypto")
    parser.add_argument("--preaugs", default="baseline,percent_change,log_returns")
    parser.add_argument("--context-lengths", default="128,256")
    parser.add_argument("--learning-rates", default="5e-5,1e-4")
    parser.add_argument("--num-steps", type=int, default=800)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--train-hours", type=int, default=24 * 180)
    parser.add_argument("--val-hours", type=int, default=24 * 60)
    parser.add_argument("--gap-hours", type=int, default=0)
    parser.add_argument("--feature-lag", type=int, default=1)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--forecast-lookback-hours", type=float, default=None)
    parser.add_argument("--time-budget", type=int, default=6 * 3600)
    parser.add_argument("--max-trials", type=int, default=8)
    parser.add_argument("--descriptions", default="baseline_anneal_lr,ent_anneal,clip_vloss,wd_01,slip_5bps")
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--remote-env", default=DEFAULT_REMOTE_ENV)
    parser.add_argument("--skip-remote-window-check", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = normalize_symbols(parse_csv_tokens(args.symbols))
    preaugs = [str(item).strip() for item in parse_csv_tokens(args.preaugs)]
    context_lengths = [int(item) for item in parse_csv_tokens(args.context_lengths, cast=int)]
    learning_rates = [float(item) for item in parse_csv_tokens(args.learning_rates, cast=float)]
    descriptions = [str(item).strip() for item in parse_csv_tokens(args.descriptions)]

    local_earliest, local_latest = compute_hourly_overlap_bounds(
        symbols=symbols,
        data_root=Path(args.local_data_root),
    )
    effective_earliest = local_earliest
    effective_latest = local_latest
    if not args.skip_remote_window_check:
        remote_earliest, remote_latest = _load_remote_overlap_bounds(
            remote_host=str(args.remote_host),
            remote_dir=str(args.remote_dir),
            remote_env=str(args.remote_env),
            remote_data_root=str(args.remote_data_root),
            symbols=symbols,
        )
        effective_earliest = max(local_earliest, remote_earliest)
        effective_latest = min(local_latest, remote_latest)

    plan = build_remote_hourly_chronos_rl_plan(
        run_id=str(args.run_id),
        symbols=symbols,
        local_data_root=Path(args.local_data_root),
        remote_data_root=str(args.remote_data_root),
        train_hours=int(args.train_hours),
        val_hours=int(args.val_hours),
        gap_hours=int(args.gap_hours),
        preaugs=preaugs,
        context_lengths=context_lengths,
        learning_rates=learning_rates,
        num_steps=int(args.num_steps),
        prediction_length=int(args.prediction_length),
        lora_r=int(args.lora_r),
        feature_lag=int(args.feature_lag),
        min_coverage=float(args.min_coverage),
        time_budget=int(args.time_budget),
        max_trials=int(args.max_trials),
        descriptions=descriptions,
        forecast_lookback_hours=args.forecast_lookback_hours,
        earliest_common_override=effective_earliest,
        latest_common_override=effective_latest,
    )
    pipeline_script = render_remote_pipeline_script(
        remote_dir=str(args.remote_dir),
        remote_env=str(args.remote_env),
        plan=plan,
    )
    rsync_cmd = _build_rsync_cmd(str(args.remote_host), str(args.remote_dir))
    manifest_dir = REPO / "analysis" / "remote_runs" / str(args.run_id)
    manifest_path = _write_local_manifest(
        manifest_dir=manifest_dir,
        args=args,
        plan_payload=plan.as_dict(),
        pipeline_script=pipeline_script,
        rsync_cmd=rsync_cmd,
        remote_host=str(args.remote_host),
        remote_dir=str(args.remote_dir),
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
        ["ssh", "-S", "none", "-o", "ControlMaster=no", "-o", "StrictHostKeyChecking=no", str(args.remote_host)],
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
