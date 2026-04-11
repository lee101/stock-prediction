"""fp4 replay renderer (P6-1).

Runs a short rollout of an fp4 env (default: ``gpu_trading_env`` via
``fp4.env_adapter.make_env``), captures the per-step trajectory with
``fp4.replay_recorder.ReplayRecorder``, converts it to a
``src.marketsim_video.MarketsimTrace``, and produces an MP4 + HTML index
under ``models/artifacts/<run_name>/videos/``.

The renderer path goes through the existing marketsim video utilities used
by ``pufferlib_market.intrabar_replay.build_hourly_marketsim_trace`` → so
this script is the "gpu_trading_env → intrabar replay video" bridge the
Phase 6 plan asks for. When the plotly HTML dependency is missing we still
write the MP4 and a minimal ``index.html``.

Usage::

    python fp4/bench/render_replay.py --trainer fp4 --steps 4096 --seed 0
    # → models/artifacts/fp4_<YYYY-MM-DD>/videos/replay.mp4
    #   models/artifacts/fp4_<YYYY-MM-DD>/videos/index.html
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Repo root on sys.path so ``import src.marketsim_video`` works when running
# this file as a script from any cwd.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from fp4.fp4.env_adapter import make_env  # noqa: E402
from fp4.fp4.replay_recorder import (  # noqa: E402
    ReplayRecorder,
    ReplayTrajectory,
    trajectory_to_marketsim_trace,
)


def _default_run_name() -> str:
    return f"fp4_{dt.date.today().isoformat()}"


def _simple_policy(obs: torch.Tensor, act_dim: int) -> torch.Tensor:
    """Trivial signed-momentum policy: ``sign(close - open)`` sized at 0.5.

    This is NOT trained — it exists purely so the recorder + renderer see a
    non-trivial stream of fills to verify the pipeline end-to-end. ``obs`` is
    the flattened gpu_trading_env obs ``[N, 7]``: open, high, low, close,
    equity, pos_qty, drawdown.
    """
    if obs.dim() == 2 and obs.shape[-1] >= 4:
        sig = torch.tanh((obs[:, 3] - obs[:, 0]) * 10.0)
    else:
        sig = torch.zeros(obs.shape[0], device=obs.device)
    act = torch.zeros(obs.shape[0], max(act_dim, 1), device=obs.device, dtype=torch.float32)
    act[:, 0] = 0.5 * sig
    return act


def run_rollout(
    *,
    steps: int,
    seed: int,
    num_envs: int = 8,
    backend: str = "auto",
    device: Optional[str] = None,
) -> tuple[ReplayTrajectory, str]:
    """Build an env, run ``steps`` steps with a trivial policy, return trajectory."""
    cfg = {"env": backend} if backend != "auto" else {}
    dev = torch.device(device) if device else None
    env = make_env(cfg, num_envs=int(num_envs), seed=int(seed), device=dev)
    rec = ReplayRecorder.attach(env, max_steps=int(steps), env_index=0)
    obs = env.reset()
    if obs.dtype != torch.float32:
        obs = obs.to(torch.float32)
    rec.on_reset(obs)
    for _ in range(int(steps)):
        action = _simple_policy(obs, env.action_dim)
        obs, reward, done, _cost = env.step(action)
        if obs.dtype != torch.float32:
            obs = obs.to(torch.float32)
        rec.on_step(action=action, obs=obs, reward=reward, done=done)
    traj = rec.trajectory()
    return traj, env.backend_name


def render_videos(
    traj: ReplayTrajectory,
    out_dir: Path,
    *,
    title: str,
    fps: int = 8,
    num_pairs: int = 1,
) -> dict:
    """Render MP4 + HTML under ``out_dir``. Returns a dict of produced paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = trajectory_to_marketsim_trace(traj, symbol_name="ENV0")

    from src.marketsim_video import render_mp4  # local import, heavy deps

    mp4_path = out_dir / "replay.mp4"
    render_mp4(
        trace,
        mp4_path,
        num_pairs=num_pairs,
        fps=fps,
        title=title,
        nvenc=False,  # portability first; fall back to libx264 everywhere
    )

    html_path = out_dir / "index.html"
    produced = {"mp4": str(mp4_path)}
    try:
        from src.marketsim_video import render_html_plotly
        render_html_plotly(trace, html_path, num_pairs=num_pairs, title=title)
        produced["html"] = str(html_path)
    except Exception as exc:
        # Plotly optional; fall back to a minimal HTML linking the MP4.
        minimal = (
            "<!doctype html><meta charset='utf-8'>"
            f"<title>{title}</title>"
            f"<h1>{title}</h1>"
            f"<video controls autoplay loop src='{mp4_path.name}'></video>"
            f"<p>plotly unavailable: <code>{type(exc).__name__}</code></p>"
        )
        html_path.write_text(minimal)
        produced["html"] = str(html_path)
        produced["html_fallback_reason"] = f"{type(exc).__name__}: {exc}"

    meta_path = out_dir / "replay_meta.json"
    meta_path.write_text(json.dumps({
        "num_steps": int(traj.num_steps),
        "backend": traj.backend_name,
        "initial_equity": float(traj.initial_equity),
        "final_equity": float(traj.equity[-1]) if traj.num_steps else None,
        "title": title,
        **produced,
    }, indent=2))
    produced["meta"] = str(meta_path)
    return produced


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", default="fp4",
                        help="Label for the run directory (default: fp4).")
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "gpu_trading_env", "market_sim_py", "stub"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--artifacts-root", default=None,
                        help="Override the artifacts root (default: <repo>/models/artifacts).")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--device", default=None,
                        help="Force torch device (e.g. cpu). Default: auto.")
    args = parser.parse_args(argv)

    run_name = args.run_name or _default_run_name()
    root = Path(args.artifacts_root) if args.artifacts_root else (_REPO / "models" / "artifacts")
    out_dir = root / run_name / "videos"

    print(f"[render_replay] running {args.trainer} backend={args.backend} "
          f"steps={args.steps} seed={args.seed}")
    traj, backend_name = run_rollout(
        steps=args.steps, seed=args.seed,
        num_envs=args.num_envs, backend=args.backend, device=args.device,
    )
    print(f"[render_replay] recorded {traj.num_steps} steps on backend={backend_name}")
    print(f"[render_replay] writing videos to {out_dir}")

    produced = render_videos(
        traj, out_dir,
        title=f"{args.trainer} replay ({backend_name}, seed={args.seed})",
        fps=int(args.fps),
    )
    print(f"[render_replay] done: {json.dumps(produced, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
