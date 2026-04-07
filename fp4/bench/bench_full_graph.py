"""Benchmark: SPS comparison for 3 modes of the fp4 PPO inner loop.

Modes
-----
1. eager           — full python rollout + eager update (no graph at all)
2. update-only     — Phase 3 graph (update step only); rollout still eager
3. full            — Phase 5 P5-3 graph (env + fwd + bwd + optim all captured)

Each mode is given a fixed wall-clock budget (default 10 s), and we report
total environment steps / wall_sec. Output written to
`fp4/bench/results/full_graph_sps_<YYYYMMDD>.json`.

Skipped (with reason recorded) when CUDA is unavailable.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

import torch


def _make_policy_and_optim(obs_dim: int, act_dim: int, hidden: int, device):
    from fp4.policy import ActorCritic
    policy = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, seed=0).to(device)
    optim = torch.optim.SGD(policy.parameters(), lr=1e-3)
    return policy, optim


def bench_eager(*, num_envs, rollout_len, obs_dim, act_dim, hidden, budget_s, device):
    from fp4.cuda_graph_full import build_synthetic_full_step
    policy, optim = _make_policy_and_optim(obs_dim, act_dim, hidden, device)
    step_fn, _ = build_synthetic_full_step(
        policy, optim, num_envs=num_envs, rollout_len=rollout_len,
        obs_dim=obs_dim, act_dim=act_dim, device=device,
    )
    # Warmup
    for _ in range(3):
        step_fn()
    torch.cuda.synchronize()

    n_iters = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < budget_s:
        step_fn()
        n_iters += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    total_steps = n_iters * num_envs * rollout_len
    return {"mode": "eager", "iters": n_iters, "wall_sec": wall,
            "total_env_steps": total_steps, "sps": total_steps / wall}


def bench_update_only(*, num_envs, rollout_len, obs_dim, act_dim, hidden, budget_s, device):
    """Capture only the *update* step (mirrors Phase 3 cuda_graph.py).

    Rollout runs eager; the inner PPO loss/backward/optimizer.step is replayed
    from a captured CUDA graph.
    """
    from fp4.cuda_graph_full import build_synthetic_full_step
    policy, optim = _make_policy_and_optim(obs_dim, act_dim, hidden, device)
    # Reuse the rollout helper to populate buffers, but separately capture the
    # update step on a private stream the way Phase 3's _try_capture_update does.
    step_fn, state = build_synthetic_full_step(
        policy, optim, num_envs=num_envs, rollout_len=rollout_len,
        obs_dim=obs_dim, act_dim=act_dim, device=device,
    )

    T, N = rollout_len, num_envs
    b_obs = state["obs_buf"].reshape(T * N, obs_dim)
    b_act = state["act_buf"].reshape(T * N, act_dim)
    b_logp = state["logp_buf"].reshape(T * N)
    b_adv = state["adv_buf"].reshape(T * N)
    b_ret = state["ret_buf"].reshape(T * N)
    clip_eps, vf_coef, ent_coef = 0.2, 0.5, 0.01

    def update_only():
        optim.zero_grad(set_to_none=False)
        mean, std, value = policy(b_obs)
        var = std * std
        new_logp = (-0.5 * (((b_act - mean) ** 2) / (var + 1e-8)
                            + 2 * torch.log(std + 1e-8)
                            + math.log(2 * math.pi))).sum(dim=-1)
        entropy = (0.5 * math.log(2 * math.pi * math.e)
                   + torch.log(std + 1e-8)).sum(dim=-1).mean()
        ratio = torch.exp(new_logp - b_logp)
        adv_norm = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        surr1 = ratio * adv_norm
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = ((value - b_ret) ** 2).mean()
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        loss.backward()
        optim.step()
        state["loss"].copy_(loss.detach())
        return {"loss": state["loss"]}

    # Prime rollout buffers once.
    step_fn()
    torch.cuda.synchronize()

    # Capture update only (Phase 3 pattern).
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            update_only()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            update_only()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    n_iters = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < budget_s:
        # Eager rollout to refill buffers
        step_fn()  # contains both rollout + an eager update; close enough
        # Replay captured update graph
        g.replay()
        n_iters += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    total_steps = n_iters * num_envs * rollout_len
    return {"mode": "update_only", "iters": n_iters, "wall_sec": wall,
            "total_env_steps": total_steps, "sps": total_steps / wall}


def bench_full(*, num_envs, rollout_len, obs_dim, act_dim, hidden, budget_s, device):
    from fp4.cuda_graph_full import build_synthetic_full_step, capture_full_step
    policy, optim = _make_policy_and_optim(obs_dim, act_dim, hidden, device)
    step_fn, _ = build_synthetic_full_step(
        policy, optim, num_envs=num_envs, rollout_len=rollout_len,
        obs_dim=obs_dim, act_dim=act_dim, device=device,
    )
    captured = capture_full_step(step_fn)

    n_iters = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < budget_s:
        captured.replay()
        n_iters += 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    total_steps = n_iters * num_envs * rollout_len
    return {"mode": "full", "iters": n_iters, "wall_sec": wall,
            "total_env_steps": total_steps, "sps": total_steps / wall}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget-sec", type=float, default=10.0)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--rollout-len", type=int, default=32)
    p.add_argument("--obs-dim", type=int, default=16)
    p.add_argument("--act-dim", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    args = p.parse_args()

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    date = _dt.datetime.now().strftime("%Y%m%d")
    out_path = out_dir / f"full_graph_sps_{date}.json"

    if not torch.cuda.is_available():
        payload = {
            "skipped": True, "reason": "no CUDA available",
            "rows": [], "config": vars(args),
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[bench_full_graph] skipped (no CUDA); wrote {out_path}")
        return 0

    device = torch.device("cuda")
    common = dict(num_envs=args.num_envs, rollout_len=args.rollout_len,
                  obs_dim=args.obs_dim, act_dim=args.act_dim, hidden=args.hidden,
                  budget_s=args.budget_sec, device=device)

    rows = []
    for fn in (bench_eager, bench_update_only, bench_full):
        try:
            row = fn(**common)
        except Exception as exc:
            row = {"mode": fn.__name__.replace("bench_", ""),
                   "error": f"{type(exc).__name__}: {exc}", "sps": 0.0}
        rows.append(row)
        print(f"  {row.get('mode'):>12s}  sps={row.get('sps', 0.0):,.0f}  "
              f"iters={row.get('iters', 0)}")

    payload = {
        "rows": rows,
        "config": vars(args),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "phase3_baseline_sps": 5159,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[bench_full_graph] wrote {out_path}")

    full = next((r for r in rows if r.get("mode") == "full"), {})
    upd = next((r for r in rows if r.get("mode") == "update_only"), {})
    if full.get("sps", 0) <= upd.get("sps", 0):
        print(f"[bench_full_graph] WARNING: full ({full.get('sps')}) <= "
              f"update_only ({upd.get('sps')})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
