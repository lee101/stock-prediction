#!/usr/bin/env python3
"""Launch a remote screened32 Chronos2->daily-RL upgrade pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.daily_stock_defaults import DEFAULT_SYMBOLS
from src.remote_training_pipeline import (
    DEFAULT_REMOTE_DIR,
    DEFAULT_REMOTE_ENV,
    DEFAULT_REMOTE_HOST,
    build_remote_large_universe_stock_plan,
    compute_daily_overlap_bounds,
    compute_hourly_overlap_bounds,
    normalize_symbols,
    render_remote_pipeline_script,
)


def _default_run_id() -> str:
    return time.strftime("screened32_chronos_upgrade_%Y%m%d_%H%M%S")


def _build_rsync_cmd(remote_host: str, remote_dir: str) -> list[str]:
    sync_items = [
        "scripts/",
        "src/",
        "pufferlib_market/",
        "chronos2_trainer.py",
        "docs/",
        "AGENTS.md",
    ]
    return [
        "rsync",
        "-az",
        "-e",
        "ssh -S none -o ControlMaster=no -o StrictHostKeyChecking=no",
        *sync_items,
        f"{remote_host}:{remote_dir}/",
    ]


def _run_rsync(command: list[str], *, password_env: str, cwd: Path) -> None:
    password = os.getenv(str(password_env))
    if not password:
        subprocess.run(command, cwd=cwd, check=True)
        return
    remote_shell = 'sshpass -p "$REMOTE_PASS" ssh -S none -o ControlMaster=no -o StrictHostKeyChecking=no'
    items = [shlex.quote(str(part)) for part in command]
    rewritten: list[str] = []
    replaced = False
    idx = 0
    while idx < len(items):
        if idx + 1 < len(items) and command[idx] == "-e":
            rewritten.extend(["-e", shlex.quote(remote_shell)])
            idx += 2
            replaced = True
            continue
        rewritten.append(items[idx])
        idx += 1
    if not replaced:
        rewritten.insert(2, shlex.quote(remote_shell))
        rewritten.insert(2, "-e")
    shell_cmd = " ".join(rewritten)
    subprocess.run(
        ["bash", "-lc", shell_cmd],
        cwd=cwd,
        check=True,
        env={**os.environ, "REMOTE_PASS": password},
    )


def _with_sshpass(command: list[str], *, password_env: str) -> list[str]:
    password = os.getenv(str(password_env))
    if not password:
        return command
    return ["sshpass", "-p", password, *command]


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


def _write_manifest(
    *,
    manifest_dir: Path,
    args: argparse.Namespace,
    plan_payload: dict[str, object],
    pipeline_script: str,
    rsync_cmd: list[str],
) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "pipeline.sh").write_text(pipeline_script)
    evaluation_dir = manifest_dir / "candidate_evals"
    local_remote_run_dir = manifest_dir / "remote_run"
    local_daily_ckpt_dir = manifest_dir / "daily_checkpoints"
    selector_cmd = [
        sys.executable,
        "scripts/evaluate_screened32_candidates.py",
        "--leaderboard",
        str(local_remote_run_dir / Path(str(plan_payload["daily_leaderboard_path"])).name),
        "--checkpoint-root",
        str(local_daily_ckpt_dir),
        "--out-dir",
        str(evaluation_dir),
    ]
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
                str(local_remote_run_dir) + "/",
            ],
            "pull_daily_checkpoints": [
                "rsync",
                "-az",
                "-e",
                "ssh -o StrictHostKeyChecking=no",
                f"{args.remote_host}:{args.remote_dir}/{plan_payload['daily_checkpoint_root']}/",
                str(local_daily_ckpt_dir) + "/",
            ],
            "post_eval_top_candidates": selector_cmd,
        },
    }
    path = manifest_dir / "launch_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--local-hourly-data-root", type=Path, default=Path("trainingdatahourly"))
    parser.add_argument("--remote-hourly-data-root", default="trainingdatahourly")
    parser.add_argument("--local-daily-data-root", type=Path, default=Path("trainingdata"))
    parser.add_argument("--remote-daily-data-root", default="trainingdata")
    parser.add_argument("--train-hours", type=int, default=24 * 365)
    parser.add_argument("--val-hours", type=int, default=24 * 140)
    parser.add_argument("--gap-hours", type=int, default=24)
    parser.add_argument("--time-budget", type=int, default=2700)
    parser.add_argument("--max-trials", type=int, default=8)
    parser.add_argument("--forecast-lookback-hours", type=float, default=None)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--remote-env", default=DEFAULT_REMOTE_ENV)
    parser.add_argument("--remote-password-env", default="REMOTE_PASS")
    parser.add_argument("--skip-remote-window-check", action="store_true")
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = normalize_symbols(str(args.symbols).split(","))

    effective_earliest = None
    effective_latest = None
    overlap_summary: dict[str, str] | None = None
    if not args.skip_remote_window_check:
        hourly_earliest, hourly_latest = compute_hourly_overlap_bounds(
            symbols=symbols,
            data_root=Path(args.local_hourly_data_root),
        )
        daily_earliest, daily_latest = compute_daily_overlap_bounds(
            symbols=symbols,
            data_root=Path(args.local_daily_data_root),
        )
        effective_earliest = max(hourly_earliest, daily_earliest)
        effective_latest = min(hourly_latest, daily_latest)
        overlap_summary = {
            "hourly_earliest": hourly_earliest,
            "hourly_latest": hourly_latest,
            "daily_earliest": daily_earliest,
            "daily_latest": daily_latest,
            "effective_earliest": effective_earliest,
            "effective_latest": effective_latest,
        }

    plan = build_remote_large_universe_stock_plan(
        run_id=str(args.run_id),
        symbols=symbols,
        local_hourly_data_root=Path(args.local_hourly_data_root),
        remote_hourly_data_root=str(args.remote_hourly_data_root),
        local_daily_data_root=Path(args.local_daily_data_root),
        remote_daily_data_root=str(args.remote_daily_data_root),
        train_hours=int(args.train_hours),
        val_hours=int(args.val_hours),
        gap_hours=int(args.gap_hours),
        time_budget=int(args.time_budget),
        max_trials=int(args.max_trials),
        forecast_lookback_hours=args.forecast_lookback_hours,
        earliest_common_override=effective_earliest,
        latest_common_override=effective_latest,
    )
    pipeline_script = render_remote_pipeline_script(
        remote_dir=str(args.remote_dir),
        remote_env=str(args.remote_env),
        plan=plan,
    )
    manifest_dir = REPO / "analysis" / "remote_runs" / str(args.run_id)
    rsync_cmd = _build_rsync_cmd(str(args.remote_host), str(args.remote_dir))
    manifest_path = _write_manifest(
        manifest_dir=manifest_dir,
        args=args,
        plan_payload={
            **plan.as_dict(),
            "overlap_summary": overlap_summary,
        },
        pipeline_script=pipeline_script,
        rsync_cmd=rsync_cmd,
    )

    if args.dry_run:
        print(f"Manifest: {manifest_path}")
        if overlap_summary:
            print(json.dumps(overlap_summary, indent=2))
        print(pipeline_script)
        return 0

    if not args.no_sync:
        _run_rsync(rsync_cmd, password_env=str(args.remote_password_env), cwd=REPO)

    bootstrap = _build_remote_bootstrap_script(
        remote_dir=str(args.remote_dir),
        script_path=plan.remote_script_path,
        log_path=plan.remote_log_path,
        pid_path=plan.remote_pid_path,
        pipeline_script=pipeline_script,
    )
    launch = subprocess.run(
        _with_sshpass(
            ["ssh", "-S", "none", "-o", "ControlMaster=no", "-o", "StrictHostKeyChecking=no", str(args.remote_host)],
            password_env=str(args.remote_password_env),
        ),
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
    print(
        "Pull daily checkpoints:\n"
        f'rsync -az -e "ssh -o StrictHostKeyChecking=no" '
        f"{args.remote_host}:{args.remote_dir}/{plan.daily_checkpoint_root}/ {manifest_dir}/daily_checkpoints/"
    )
    print(
        "Post-eval:\n"
        f"{sys.executable} scripts/evaluate_screened32_candidates.py "
        f"--leaderboard {manifest_dir}/remote_run/{Path(plan.daily_leaderboard_path).name} "
        f"--checkpoint-root {manifest_dir}/daily_checkpoints "
        f"--out-dir {manifest_dir}/candidate_evals"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
