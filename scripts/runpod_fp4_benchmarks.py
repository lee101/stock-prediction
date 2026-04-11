#!/usr/bin/env python3
"""Launch a RunPod 5090, run fp4 Phase 5 sweep + fused bench + gemini A/B, tear down.

Benchmarks (sequential, abort-on-failure between stages but NEVER skip teardown):

    A. python fp4/bench/sweep.py --algos ppo,sac,qr_ppo --constrained both \
         --seeds 0,1,2,3,4 --steps 2000000
       then: python fp4/bench/make_leaderboard.py
    B. python fp4/bench/bench_fused.py
    C. python fp4/bench/bench_gemini_ab.py --steps 50000 --seeds 0,1

Results are rsync'd back to fp4/bench/results/runpod_<date>/ on the local repo.
The pod is always terminated in the finally block; if teardown fails, the
script loudly prints 'Pod STILL RUNNING: <id>' to stderr.
"""

from __future__ import annotations

import datetime as _dt
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.runpod_client import (  # noqa: E402
    DEFAULT_POD_READY_POLL_INTERVAL_SECONDS,
    DEFAULT_POD_READY_TIMEOUT_SECONDS,
    PodConfig,
    RunPodClient,
    resolve_gpu_preferences,
)
from src.runpod_remote_utils import SSH_OPTIONS, ssh_run  # noqa: E402

REMOTE_WS = "/workspace/stock-prediction"
REMOTE_VENV = ".venv"
DATE_TAG = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
RESULTS_LOCAL = REPO / "fp4" / "bench" / "results" / f"runpod_{DATE_TAG}"


def _ssh_exec(ready, remote_cmd: str, *, stage: str) -> subprocess.CompletedProcess[str]:
    print(f"\n=== [{stage}] ssh exec ===\n  $ {remote_cmd[:200]}{'...' if len(remote_cmd) > 200 else ''}", flush=True)
    # Stream output directly; do NOT capture (benches can run for hours)
    result = ssh_run(
        ssh_host=ready.ssh_host,
        ssh_port=ready.ssh_port,
        remote_cmd=remote_cmd,
        capture_output=False,
    )
    print(f"=== [{stage}] exit={result.returncode} ===", flush=True)
    return result


def _rsync_push(ready) -> None:
    """Rsync local repo -> pod (exclude heavy dirs)."""
    cmd = [
        "rsync", "-az",
        "--delete",
        "--exclude", ".git/",
        "--exclude", "__pycache__/",
        "--exclude", "*.pyc",
        "--exclude", ".venv*/",
        "--exclude", "trainingdata/",
        "--exclude", "data/",
        "--exclude", "fp4/bench/results/",
        "--exclude", "pufferlib_market/checkpoints/",
        "--exclude", "build/",
        "--exclude", "dist/",
        "--exclude", "*.egg-info/",
        "-e", f"ssh {' '.join(SSH_OPTIONS)} -p {ready.ssh_port}",
        f"{REPO}/",
        f"root@{ready.ssh_host}:{REMOTE_WS}/",
    ]
    print(f"\n=== rsync push (excludes trainingdata/data/results) ===", flush=True)
    r = subprocess.run(cmd, check=False, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"rsync push failed exit={r.returncode}")


def _rsync_pull_results(ready) -> None:
    RESULTS_LOCAL.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-az",
        "-e", f"ssh {' '.join(SSH_OPTIONS)} -p {ready.ssh_port}",
        f"root@{ready.ssh_host}:{REMOTE_WS}/fp4/bench/results/",
        f"{RESULTS_LOCAL}/",
    ]
    print(f"\n=== rsync pull results -> {RESULTS_LOCAL} ===", flush=True)
    r = subprocess.run(cmd, check=False, text=True)
    if r.returncode != 0:
        print(f"WARNING: rsync pull failed exit={r.returncode} (continuing to teardown)", file=sys.stderr)


BOOTSTRAP = r"""
set -euo pipefail
cd /workspace
if [ ! -d stock-prediction ]; then mkdir -p stock-prediction; fi
cd stock-prediction
python3 --version
apt-get update -qq && apt-get install -y -qq rsync build-essential git >/tmp/apt.log 2>&1 || true
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip setuptools wheel ninja 2>&1 | tail -5
python -m pip install "torch>=2.9" --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5 || \
  python -m pip install "torch>=2.9" 2>&1 | tail -5
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
python -m pip install -e fp4/ 2>&1 | tail -10
# Extensions: build with SM120 flags (the setup.py files already carry them per commits 15cdd8b9 / 3a1a29c2)
export TORCH_CUDA_ARCH_LIST="12.0"
(cd gpu_trading_env && python -m pip install -e . 2>&1 | tail -15) || echo "WARN: gpu_trading_env build failed"
(cd pufferlib_cpp_market_sim && python -m pip install -e . 2>&1 | tail -15) || echo "WARN: market_sim_py build failed"
# Other deps bench scripts may need
python -m pip install numpy pandas matplotlib pyyaml tqdm 2>&1 | tail -3
python -c "import fp4; print('fp4 import ok')"
echo BOOTSTRAP_OK
"""


BENCH_A = (
    "set -e; cd /workspace/stock-prediction && . .venv/bin/activate && "
    "python fp4/bench/sweep.py --algos ppo,sac,qr_ppo --constrained both "
    "--seeds 0,1,2,3,4 --steps 2000000 2>&1 | tee /tmp/benchA.log; "
    "python fp4/bench/make_leaderboard.py 2>&1 | tee /tmp/benchA_leaderboard.log"
)

BENCH_B = (
    "set -e; cd /workspace/stock-prediction && . .venv/bin/activate && "
    "python fp4/bench/bench_fused.py 2>&1 | tee /tmp/benchB.log"
)

BENCH_C = (
    "set -e; cd /workspace/stock-prediction && . .venv/bin/activate && "
    "python fp4/bench/bench_gemini_ab.py --steps 50000 --seeds 0,1 2>&1 | tee /tmp/benchC.log"
)

# Smoke: 1 algo, 1 seed, 50k steps -- aborts fast if the env/build is broken
BENCH_SMOKE = (
    "set -e; cd /workspace/stock-prediction && . .venv/bin/activate && "
    "python fp4/bench/sweep.py --algos ppo --constrained off --seeds 0 --steps 50000 2>&1 | tee /tmp/smoke.log"
)


def main() -> int:
    client = RunPodClient()
    gpu_prefs = resolve_gpu_preferences("5090", None)
    print(f"GPU preferences: {gpu_prefs}", flush=True)

    cfg = PodConfig(
        name=f"fp4-bench-{int(time.time())}",
        gpu_type=gpu_prefs[0],
        gpu_count=1,
        volume_size=80,
        container_disk=60,
        cloud_type="COMMUNITY",
    )

    pod_id = None
    start_time = time.monotonic()
    status = "unknown"
    try:
        print(f"[{_dt.datetime.now().isoformat()}] Creating pod...", flush=True)
        pod = client.create_pod_with_fallback(cfg, gpu_prefs)
        pod_id = pod.id
        print(f"Pod created: id={pod_id} (waiting for SSH, up to 25 min)...", flush=True)
        ready = client.wait_for_pod(pod_id, timeout=1500, poll_interval=15)
        print(f"Pod READY: id={ready.id} gpu={ready.gpu_type} ssh={ready.ssh_host}:{ready.ssh_port}", flush=True)

        _rsync_push(ready)

        bs = _ssh_exec(ready, BOOTSTRAP, stage="bootstrap")
        if bs.returncode != 0:
            status = "bootstrap_failed"
            raise RuntimeError(f"bootstrap exit={bs.returncode}")

        # Smoke test before committing to multi-hour sweep
        sm = _ssh_exec(ready, BENCH_SMOKE, stage="smoke")
        if sm.returncode != 0:
            status = "smoke_failed"
            raise RuntimeError(f"smoke exit={sm.returncode}")

        ra = _ssh_exec(ready, BENCH_A, stage="A-sweep")
        a_ok = ra.returncode == 0

        rb = _ssh_exec(ready, BENCH_B, stage="B-fused")
        b_ok = rb.returncode == 0

        rc = _ssh_exec(ready, BENCH_C, stage="C-gemini")
        c_ok = rc.returncode == 0

        status = f"A={'ok' if a_ok else 'fail'} B={'ok' if b_ok else 'fail'} C={'ok' if c_ok else 'fail'}"
        print(f"\n=== benchmarks done: {status} ===", flush=True)

        _rsync_pull_results(ready)
        # Also pull the /tmp logs
        for name in ("benchA.log", "benchA_leaderboard.log", "benchB.log", "benchC.log", "smoke.log", "apt.log"):
            subprocess.run(
                ["rsync", "-az",
                 "-e", f"ssh {' '.join(SSH_OPTIONS)} -p {ready.ssh_port}",
                 f"root@{ready.ssh_host}:/tmp/{name}",
                 f"{RESULTS_LOCAL}/{name}"],
                check=False, text=True,
            )
        return 0 if (a_ok and b_ok and c_ok) else 2
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr, flush=True)
        if status == "unknown":
            status = "exception"
        return 1
    finally:
        uptime = int(time.monotonic() - start_time)
        print(f"\n=== TEARDOWN === pod_id={pod_id} uptime={uptime}s status={status}", flush=True)
        if pod_id:
            terminated = False
            for attempt in range(3):
                try:
                    client.terminate_pod(pod_id)
                    terminated = True
                    break
                except Exception as texc:
                    print(f"terminate attempt {attempt+1} failed: {texc}", file=sys.stderr, flush=True)
                    time.sleep(5)
            # Verify via list_pods
            try:
                remaining = [p for p in client.list_pods() if p.id == pod_id and p.status != "TERMINATED"]
                if remaining:
                    print(f"!!! Pod STILL RUNNING: {pod_id} !!! ({remaining[0].status}) — MANUAL CLEANUP REQUIRED", file=sys.stderr, flush=True)
                else:
                    print(f"Pod terminated: {pod_id}", flush=True)
            except Exception as lexc:
                if terminated:
                    print(f"Pod terminated: {pod_id} (verify failed: {lexc})", flush=True)
                else:
                    print(f"!!! Pod STILL RUNNING: {pod_id} !!! (terminate + verify both failed: {lexc}) — MANUAL CLEANUP REQUIRED", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
