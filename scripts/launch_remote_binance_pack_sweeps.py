#!/usr/bin/env python3
"""Launch Binance hourly portfolio-pack sweeps on remote GPU machines.

The launcher is deliberately conservative with remote repositories: it does
not pull into a dirty checkout.  Each target fetches `origin/main`, creates a
detached worktree, symlinks the large local data directory, then runs a
disjoint config-sample shard in the background.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RemoteTarget:
    name: str
    host: str
    repo: str
    env: str
    gpu: str = "0"


DEFAULT_TARGETS = {
    "daisy": RemoteTarget(
        name="daisy",
        host="lee@daisy",
        repo="/media/lee/pcd/code/stock",
        env=".venv313",
        gpu="0",
    ),
    "leaf5090": RemoteTarget(
        name="leaf5090",
        host="administrator@93.127.141.100",
        repo="/nvme0n1-disk/code/stock-prediction",
        env=".venv313",
        gpu="0",
    ),
}


def _default_run_id() -> str:
    return time.strftime("binance_pack_remote_%Y%m%d_%H%M%S")


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _parse_targets(value: str) -> list[RemoteTarget]:
    targets: list[RemoteTarget] = []
    for token in str(value).split(","):
        name = token.strip()
        if not name:
            continue
        if name not in DEFAULT_TARGETS:
            known = ", ".join(sorted(DEFAULT_TARGETS))
            raise ValueError(f"unknown target {name!r}; known targets: {known}")
        targets.append(DEFAULT_TARGETS[name])
    if not targets:
        raise ValueError("no remote targets selected")
    return targets


def _sweep_command(
    *,
    run_id: str,
    target: RemoteTarget,
    seed: int,
    args: argparse.Namespace,
) -> list[str]:
    out_dir = f"analysis/remote_runs/{run_id}/{target.name}"
    command = [
        "python",
        "scripts/sweep_binance_hourly_portfolio_pack.py",
        "--min-bars",
        str(args.min_bars),
        "--min-symbols-per-hour",
        str(args.min_symbols_per_hour),
        "--train-days",
        str(args.train_days),
        "--eval-days",
        str(args.eval_days),
        "--label-horizon",
        str(args.label_horizon),
        "--rounds",
        str(args.rounds),
        "--device",
        str(args.device),
        "--max-configs",
        str(args.max_configs_per_target),
        "--config-sample-seed",
        str(seed),
        "--max-leverage-grid",
        str(args.max_leverage_grid),
        "--out",
        f"{out_dir}/portfolio_pack.csv",
    ]
    if args.symbols:
        command.extend(["--symbols", str(args.symbols)])
    if args.extra_args:
        command.extend(shlex.split(str(args.extra_args)))
    if args.render_html:
        command.extend(
            [
                "--html-out",
                f"{out_dir}/best.html",
                "--trace-json-out",
                f"{out_dir}/best.json",
                "--mp4-out",
                "",
            ]
        )
    else:
        command.extend(["--html-out", "", "--trace-json-out", "", "--mp4-out", ""])
    return command


def _remote_pipeline_script(
    *,
    target: RemoteTarget,
    run_id: str,
    remote_ref: str,
    sweep_command: list[str],
) -> str:
    worktree_parent = str(Path(target.repo).parent / "codex_worktrees")
    worktree = f"{worktree_parent}/{run_id}_{target.name}"
    run_dir = f"{worktree}/analysis/remote_runs/{run_id}/{target.name}"
    cmd = " ".join(_q(part) for part in sweep_command)
    return f"""#!/usr/bin/env bash
set -euo pipefail

BASE_REPO={_q(target.repo)}
WORKTREE={_q(worktree)}
RUN_DIR={_q(run_dir)}
REMOTE_REF={_q(remote_ref)}
ENV_DIR="$BASE_REPO/{target.env.strip('/')}"

mkdir -p "$(dirname "$WORKTREE")"
cd "$BASE_REPO"
git fetch origin main
if git worktree list --porcelain | grep -Fxq "worktree $WORKTREE"; then
  git worktree remove --force "$WORKTREE"
fi
rm -rf "$WORKTREE"
git worktree add --detach "$WORKTREE" "$REMOTE_REF"

for rel in binance_spot_hourly; do
  if [ -e "$BASE_REPO/$rel" ] && [ ! -e "$WORKTREE/$rel" ]; then
    ln -s "$BASE_REPO/$rel" "$WORKTREE/$rel"
  fi
done

mkdir -p "$RUN_DIR"
cd "$WORKTREE"
source "$ENV_DIR/bin/activate"
python - <<'PY'
import importlib.util
missing = [name for name in ("xgboost", "numpy", "pandas") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing python deps: " + ",".join(missing))
PY

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES={_q(target.gpu)}
{cmd}
"""


def _bootstrap_script(target: RemoteTarget, *, run_id: str, pipeline_script: str) -> tuple[str, dict[str, str]]:
    remote_run_dir = f"{target.repo}/analysis/remote_runs/{run_id}/{target.name}"
    remote_script = f"{remote_run_dir}/pipeline.sh"
    remote_log = f"{remote_run_dir}/pipeline.log"
    remote_pid = f"{remote_run_dir}/pipeline.pid"
    bootstrap = "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {_q(remote_run_dir)}",
            f"cat > {_q(remote_script)} <<'__CODEX_REMOTE_BINANCE_PACK__'",
            pipeline_script.rstrip("\n"),
            "__CODEX_REMOTE_BINANCE_PACK__",
            f"chmod +x {_q(remote_script)}",
            f"nohup bash {_q(remote_script)} > {_q(remote_log)} 2>&1 &",
            f"echo $! > {_q(remote_pid)}",
            f"cat {_q(remote_pid)}",
        ]
    ) + "\n"
    paths = {
        "remote_run_dir": remote_run_dir,
        "remote_script": remote_script,
        "remote_log": remote_log,
        "remote_pid": remote_pid,
        "worktree": f"{Path(target.repo).parent}/codex_worktrees/{run_id}_{target.name}",
    }
    return bootstrap, paths


def _run_ssh(host: str, script: str, *, dry_run: bool) -> str:
    cmd = ["ssh", "-S", "none", "-o", "ControlMaster=no", "-o", "StrictHostKeyChecking=no", host]
    if dry_run:
        return ""
    result = subprocess.run(cmd, input=script, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _write_manifest(
    *,
    run_id: str,
    args: argparse.Namespace,
    records: list[dict[str, object]],
) -> Path:
    manifest_dir = REPO / "analysis" / "remote_runs" / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id,
        "args": {key: str(value) for key, value in vars(args).items()},
        "targets": records,
    }
    path = manifest_dir / "launch_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def fetch_manifest(manifest_path: Path) -> int:
    payload = json.loads(manifest_path.read_text())
    run_id = str(payload["run_id"])
    local_dir = REPO / "analysis" / "remote_runs" / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    for target in payload.get("targets", []):
        name = str(target["name"])
        host = str(target["host"])
        worktree = str(target["paths"]["worktree"])
        remote_dir = f"{worktree}/analysis/remote_runs/{run_id}/{name}/"
        dest = local_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rsync",
            "-az",
            "-e",
            "ssh -o StrictHostKeyChecking=no",
            f"{host}:{remote_dir}",
            str(dest) + "/",
        ]
        subprocess.run(cmd, check=True)
        print(f"fetched {name} -> {dest}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch remote Binance hourly portfolio-pack sweep shards.")
    parser.add_argument("--fetch", type=Path, default=None, help="Fetch results using a launch_manifest.json and exit.")
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--targets", default="daisy,leaf5090")
    parser.add_argument("--remote-ref", default="origin/main")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--min-bars", type=int, default=5000)
    parser.add_argument("--min-symbols-per-hour", type=int, default=20)
    parser.add_argument("--train-days", type=int, default=720)
    parser.add_argument("--eval-days", type=int, default=120)
    parser.add_argument("--label-horizon", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-configs-per-target", type=int, default=48)
    parser.add_argument("--config-sample-seed", type=int, default=20260427)
    parser.add_argument("--max-leverage-grid", default="1.0")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--render-html", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fetch is not None:
        return fetch_manifest(args.fetch)

    targets = _parse_targets(str(args.targets))
    records: list[dict[str, object]] = []
    for shard_idx, target in enumerate(targets):
        seed = int(args.config_sample_seed) + shard_idx * 1009
        sweep_command = _sweep_command(run_id=str(args.run_id), target=target, seed=seed, args=args)
        pipeline = _remote_pipeline_script(
            target=target,
            run_id=str(args.run_id),
            remote_ref=str(args.remote_ref),
            sweep_command=sweep_command,
        )
        bootstrap, paths = _bootstrap_script(target, run_id=str(args.run_id), pipeline_script=pipeline)
        stdout = _run_ssh(target.host, bootstrap, dry_run=bool(args.dry_run))
        remote_pid = stdout.splitlines()[-1] if stdout else ""
        record: dict[str, object] = {
            **asdict(target),
            "seed": seed,
            "sweep_command": sweep_command,
            "paths": paths,
            "remote_pid": remote_pid,
            "tail_log": f"ssh -o StrictHostKeyChecking=no {target.host} 'tail -n 80 {paths['remote_log']}'",
            "fetch_command": (
                f"rsync -az -e 'ssh -o StrictHostKeyChecking=no' "
                f"{target.host}:{paths['worktree']}/analysis/remote_runs/{args.run_id}/{target.name}/ "
                f"analysis/remote_runs/{args.run_id}/{target.name}/"
            ),
        }
        records.append(record)
        print(f"{target.name}: pid={remote_pid or 'dry-run'} log={paths['remote_log']}")

    manifest_path = _write_manifest(run_id=str(args.run_id), args=args, records=records)
    print(f"Manifest: {manifest_path}")
    print(f"Fetch all: python scripts/launch_remote_binance_pack_sweeps.py --fetch {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
