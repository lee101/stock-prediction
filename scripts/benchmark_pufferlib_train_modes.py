#!/usr/bin/env python3
"""Benchmark a few pufferlib_market training modes and summarize SPS.

Example:
  source .venv/bin/activate
  PYTHONPATH=. python scripts/benchmark_pufferlib_train_modes.py \
    --data-path pufferlib_market/data/fdusd3_daily_train.bin \
    --output analysis/pufferlib_train_modes_20260402.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


SUMMARY_RE = re.compile(
    r"^\[\s*(?P<update>\d+)/(?P<total>\d+)\]\s+step=\s*(?P<step>[\d,]+)\s+sps=(?P<sps>\d+)"
)


def _mode_args(mode: str) -> list[str]:
    if mode == "base":
        return []
    if mode == "bf16":
        return ["--use-bf16"]
    if mode == "graph_bf16":
        return ["--use-bf16", "--cuda-graph-ppo"]
    raise ValueError(f"Unsupported mode: {mode}")


def _parse_summary(log_text: str) -> dict[str, int] | None:
    last_match: dict[str, int] | None = None
    for line in log_text.splitlines():
        match = SUMMARY_RE.search(line)
        if not match:
            continue
        last_match = {
            "update": int(match.group("update")),
            "total_updates": int(match.group("total")),
            "step": int(match.group("step").replace(",", "")),
            "sps": int(match.group("sps")),
        }
    return last_match


def _gpu_snapshot() -> dict[str, object]:
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        procs = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        gpu_line = gpu.stdout.strip().splitlines()[0] if gpu.stdout.strip() else ""
        if gpu_line:
            name, total_mem, used_mem, util = [part.strip() for part in gpu_line.split(",")]
            gpu_info = {
                "name": name,
                "memory_total_mb": int(total_mem),
                "memory_used_mb": int(used_mem),
                "memory_free_mb": int(total_mem) - int(used_mem),
                "utilization_gpu_pct": int(util),
            }
        else:
            gpu_info = {}
        proc_rows: list[dict[str, object]] = []
        for line in procs.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            pid, process_name, used_memory = parts
            proc_rows.append(
                {
                    "pid": int(pid),
                    "process_name": process_name,
                    "used_memory_mb": int(used_memory),
                }
            )
        return {"gpu": gpu_info, "processes": proc_rows}
    except Exception as exc:
        return {"error": str(exc)}


def _is_oom(log_text: str) -> bool:
    lowered = log_text.lower()
    return "out of memory" in lowered or "cudaerrormemoryallocation" in lowered


def _attempt_grid(args: argparse.Namespace) -> list[tuple[int, int]]:
    envs = [args.num_envs]
    if args.oom_retry:
        while envs[-1] > args.min_num_envs:
            next_env = max(args.min_num_envs, envs[-1] // 2)
            if next_env == envs[-1]:
                break
            envs.append(next_env)

    hidden_sizes = [args.hidden_size]
    if args.oom_retry:
        while hidden_sizes[-1] > args.min_hidden_size:
            next_hidden = max(args.min_hidden_size, hidden_sizes[-1] // 2)
            if next_hidden == hidden_sizes[-1]:
                break
            hidden_sizes.append(next_hidden)

    attempts: list[tuple[int, int]] = []
    for hidden in hidden_sizes:
        for env_count in envs:
            attempts.append((env_count, hidden))
    return attempts


def run_mode(args: argparse.Namespace, mode: str) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "." if not env.get("PYTHONPATH") else f".:{env['PYTHONPATH']}"

    attempts_out: list[dict[str, object]] = []
    preflight_gpu = _gpu_snapshot()
    free_mb = int(preflight_gpu.get("gpu", {}).get("memory_free_mb", 0) or 0)
    if args.min_free_gpu_mb > 0 and free_mb and free_mb < args.min_free_gpu_mb:
        return {
            "mode": mode,
            "returncode": 2,
            "wall_time_s": 0.0,
            "summary": None,
            "log_path": None,
            "num_envs": int(args.num_envs),
            "hidden_size": int(args.hidden_size),
            "attempts": attempts_out,
            "skipped": True,
            "skip_reason": (
                f"gpu_free_mb={free_mb} below min_free_gpu_mb={args.min_free_gpu_mb}; "
                "benchmark skipped to avoid futile OOM retries"
            ),
            "gpu_before": preflight_gpu,
        }

    for num_envs, hidden_size in _attempt_grid(args):
        ckpt_dir = Path("/tmp") / f"pufferlib_bench_{mode}_{num_envs}env_{hidden_size}hid"
        log_path = Path("/tmp") / f"pufferlib_bench_{mode}_{num_envs}env_{hidden_size}hid.log"
        cmd = [
            sys.executable,
            "pufferlib_market/train.py",
            "--data-path", args.data_path,
            "--num-envs", str(num_envs),
            "--hidden-size", str(hidden_size),
            "--rollout-len", str(args.rollout_len),
            "--minibatch-size", str(args.minibatch_size),
            "--ppo-epochs", str(args.ppo_epochs),
            "--total-timesteps", str(args.total_timesteps),
            "--disable-shorts",
            "--checkpoint-dir", str(ckpt_dir),
            "--save-every", "999999",
            "--log-interval", str(args.log_interval),
            *_mode_args(mode),
        ]

        gpu_before = _gpu_snapshot()
        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=args.project_root,
            env=env,
            text=True,
            capture_output=True,
        )
        wall_time = time.perf_counter() - start
        log_text = result.stdout + ("\n" + result.stderr if result.stderr else "")
        log_path.write_text(log_text)
        summary = _parse_summary(log_text)
        oom = _is_oom(log_text)
        attempt = {
            "num_envs": int(num_envs),
            "hidden_size": int(hidden_size),
            "returncode": int(result.returncode),
            "wall_time_s": round(wall_time, 3),
            "summary": summary,
            "log_path": str(log_path),
            "oom": oom,
            "gpu_before": gpu_before,
        }
        attempts_out.append(attempt)
        if result.returncode == 0:
            return {
                "mode": mode,
                "returncode": 0,
                "wall_time_s": round(wall_time, 3),
                "summary": summary,
                "log_path": str(log_path),
                "num_envs": int(num_envs),
                "hidden_size": int(hidden_size),
                "attempts": attempts_out,
            }
        if not oom or not args.oom_retry:
            break

    last = attempts_out[-1]
    return {
        "mode": mode,
        "returncode": int(last["returncode"]),
        "wall_time_s": float(last["wall_time_s"]),
        "summary": last["summary"],
        "log_path": last["log_path"],
        "num_envs": int(last["num_envs"]),
        "hidden_size": int(last["hidden_size"]),
        "attempts": attempts_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pufferlib_market training modes.")
    parser.add_argument("--project-root", default=".", help="Repo root.")
    parser.add_argument("--data-path", required=True, help="Path to market .bin file.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument("--modes", default="base,bf16,graph_bf16", help="Comma-separated benchmark modes.")
    parser.add_argument("--total-timesteps", type=int, default=131072)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--oom-retry", action="store_true", default=False, help="Retry smaller shapes on CUDA OOM.")
    parser.add_argument("--min-num-envs", type=int, default=8, help="Lower bound when halving env count for OOM retry.")
    parser.add_argument("--min-hidden-size", type=int, default=256, help="Lower bound when halving hidden size for OOM retry.")
    parser.add_argument("--min-free-gpu-mb", type=int, default=0, help="Skip benchmarking when free GPU memory is below this threshold.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.project_root = str(project_root)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = [run_mode(args, mode) for mode in modes]
    payload = {
        "data_path": args.data_path,
        "total_timesteps": args.total_timesteps,
        "num_envs": args.num_envs,
        "hidden_size": args.hidden_size,
        "rollout_len": args.rollout_len,
        "minibatch_size": args.minibatch_size,
        "ppo_epochs": args.ppo_epochs,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
