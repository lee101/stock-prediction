"""Benchmark env-steps/sec for the fp4 fast-SPS infra.

Compares three configurations on the same synthetic GPU env (or CPU fallback):
    - eager: env.step in a Python loop (no policy)
    - eager_policy: env.step + policy forward in a Python loop
    - cuda_graph: env.step + policy forward captured in a CUDA graph

The pufferlib BF16 baseline is referenced as a row from
`pufferlib_market/train.py` if a stored SPS is available; otherwise we
record `null` so the comparison is honest. Writes JSON to
`fp4/bench/results/sps_<UTC date>.json`.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

import torch

# Make `import fp4` work when running this script directly.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "fp4"))

from fp4.vec_env import (  # noqa: E402
    SyntheticOHLCEnv,
    _MarketSimPyWrapper,
    _try_import_market_sim_py,
)
from fp4.cuda_graph import capture_step  # noqa: E402


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_policy(obs_dim: int, act_dim: int, hidden: int, device: str) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(obs_dim, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, act_dim),
        torch.nn.Tanh(),
    ).to(device)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def bench_eager_env(env: SyntheticOHLCEnv, iters: int, device: str) -> float:
    action = torch.zeros(env.num_envs, env.act_dim, device=device)
    for _ in range(10):
        env.step(action)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        env.step(action)
    _sync(device)
    return (iters * env.num_envs) / (time.perf_counter() - t0)


def bench_eager_policy(env: SyntheticOHLCEnv, policy: torch.nn.Module, iters: int, device: str) -> float:
    obs = env.reset()
    for _ in range(10):
        with torch.no_grad():
            action = policy(obs)
        obs = env.step(action)["obs"]
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            action = policy(obs)
        obs = env.step(action)["obs"]
    _sync(device)
    return (iters * env.num_envs) / (time.perf_counter() - t0)


def bench_cuda_graph(env: SyntheticOHLCEnv, policy: torch.nn.Module, iters: int) -> float:
    """env.step + policy.forward captured as a single CUDA graph."""
    if not torch.cuda.is_available():
        return float("nan")
    obs = env.reset()
    out_obs = torch.zeros_like(obs)
    out_action = torch.zeros(env.num_envs, env.act_dim, device="cuda")
    out_reward = torch.zeros(env.num_envs, device="cuda")

    def step_fn(inputs):
        with torch.no_grad():
            a = policy(inputs["obs"])
        out_action.copy_(a)
        step_out = env.step(out_action)
        out_obs.copy_(step_out["obs"])
        out_reward.copy_(step_out["reward"])
        return {"obs": out_obs, "action": out_action, "reward": out_reward}

    captured = capture_step(step_fn, {"obs": obs.clone()}, warmup=5)
    # Replay loop. Feed each replay's output back as next input.
    captured.copy_inputs(obs=obs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = captured.replay()
        captured.copy_inputs(obs=out["obs"])
    torch.cuda.synchronize()
    return (iters * env.num_envs) / (time.perf_counter() - t0)


def bench_marketsim_real_eager(env, iters: int) -> float:
    """Real `market_sim_py` env stepped in a Python loop with a tiny policy
    forward pass — keeps everything on CUDA, no host syncs in the loop body.
    """
    device = "cuda"
    obs = env.reset()
    policy = _make_policy(env.obs_dim, env.act_dim, 64, device)
    for _ in range(10):
        with torch.no_grad():
            action = policy(obs)
        obs = env.step(action)["obs"]
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            action = policy(obs)
        obs = env.step(action)["obs"]
    _sync(device)
    return (iters * env.num_envs) / (time.perf_counter() - t0)


def bench_marketsim_real_cuda_graph(env, iters: int) -> float:
    """Capture marketsim env.step + policy forward into a single CUDA graph.

    The pybind11 binding launches CUDA kernels into the current stream, so it
    is graph-capturable as long as we don't allocate or sync mid-step. We
    pre-allocate output buffers and reuse them across replays.
    """
    device = "cuda"
    obs = env.reset()
    policy = _make_policy(env.obs_dim, env.act_dim, 64, device)
    out_obs = torch.zeros_like(obs)
    out_action = torch.zeros(env.num_envs, env.act_dim, device=device)
    out_reward = torch.zeros(env.num_envs, device=device)

    def step_fn(inputs):
        with torch.no_grad():
            a = policy(inputs["obs"])
        out_action.copy_(a)
        step_out = env.step(out_action)
        out_obs.copy_(step_out["obs"])
        out_reward.copy_(step_out["reward"])
        return {"obs": out_obs, "action": out_action, "reward": out_reward}

    try:
        captured = capture_step(step_fn, {"obs": obs.clone()}, warmup=5)
    except Exception as exc:  # pragma: no cover - graph capture incompatible
        print(f"  marketsim cuda graph capture failed: {exc}")
        return float("nan")
    captured.copy_inputs(obs=obs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = captured.replay()
        captured.copy_inputs(obs=out["obs"])
    torch.cuda.synchronize()
    return (iters * env.num_envs) / (time.perf_counter() - t0)


def bench_pufferlib_bf16_baseline(target_steps: int = 1_000_000) -> float | None:
    """Measure SPS for the pufferlib_market C env on the same box.

    Reuses the synthetic-MKTD path from `pufferlib_market.bench_env` so we
    don't need real training data on disk. Returns None if the binding can't
    be imported (e.g., wheel mismatch).
    """
    try:
        import tempfile
        from pathlib import Path

        import pufferlib_market.binding as binding
        from pufferlib_market.bench_env import _write_mktd_bin, run_benchmark
    except Exception as exc:  # pragma: no cover - optional baseline
        print(f"  pufferlib_bf16 baseline unavailable: {exc}")
        return None
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "bench_data.bin"
        _write_mktd_bin(data_path, num_symbols=12, num_timesteps=5000)
        binding.shared(data_path=str(data_path))
        return run_benchmark(
            data_path, num_envs=128, target_steps=target_steps, trials=3
        )


def main() -> int:
    device = _device()
    num_envs = 4096 if device == "cuda" else 256
    obs_dim, act_dim, hidden = 16, 1, 64
    iters = 200 if device == "cuda" else 50

    env = SyntheticOHLCEnv(num_envs=num_envs, obs_dim=obs_dim, act_dim=act_dim,
                           device=device, seed=0)
    env.reset()
    policy = _make_policy(obs_dim, act_dim, hidden, device)

    results = {
        "device": device,
        "num_envs": num_envs,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hidden": hidden,
        "iters": iters,
        "torch_version": torch.__version__,
        "rows": {},
        # Reference baseline — populated by hand once we measure pufferlib BF16
        # on this exact box. null means "not yet recorded".
        "pufferlib_bf16_sps": None,
    }

    results["rows"]["fp4_eager_env_only"] = bench_eager_env(env, iters, device)
    results["rows"]["fp4_eager_env_plus_policy"] = bench_eager_policy(env, policy, iters, device)
    if device == "cuda":
        results["rows"]["fp4_cuda_graph"] = bench_cuda_graph(env, policy, iters)
    else:
        results["rows"]["fp4_cuda_graph"] = None

    # ---- Real marketsim (Unit C bindings) ----
    if device == "cuda" and _try_import_market_sim_py() is not None:
        try:
            real_env = _MarketSimPyWrapper(
                num_envs=4096, device="cuda", seed=0, action_mode="dps"
            )
            results["marketsim_num_envs"] = real_env.num_envs
            results["marketsim_obs_dim"] = real_env.obs_dim
            results["marketsim_act_dim"] = real_env.act_dim
            results["rows"]["marketsim_real_eager"] = bench_marketsim_real_eager(
                real_env, iters
            )
            results["rows"]["marketsim_real_cuda_graph"] = (
                bench_marketsim_real_cuda_graph(real_env, iters)
            )
        except Exception as exc:
            print(f"  marketsim real bench failed: {exc}")
            results["rows"]["marketsim_real_eager"] = None
            results["rows"]["marketsim_real_cuda_graph"] = None
    else:
        results["rows"]["marketsim_real_eager"] = None
        results["rows"]["marketsim_real_cuda_graph"] = None

    # ---- Pufferlib BF16 baseline (C env on the same box) ----
    pl_sps = bench_pufferlib_bf16_baseline()
    results["rows"]["pufferlib_bf16_baseline"] = pl_sps
    results["pufferlib_bf16_sps"] = pl_sps

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.utcnow().strftime("%Y%m%d")
    out_path = out_dir / f"sps_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"device={device} num_envs={num_envs} iters={iters}")
    for k, v in results["rows"].items():
        if v is None:
            print(f"  {k}: skipped")
        else:
            print(f"  {k}: {v:,.0f} env-steps/sec")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
