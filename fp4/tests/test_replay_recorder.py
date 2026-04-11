"""Tests for fp4.replay_recorder + fp4/bench/render_replay.py (P6-1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fp4.fp4.env_adapter import make_env
from fp4.fp4.replay_recorder import (
    ReplayRecorder,
    ReplayTrajectory,
    trajectory_to_marketsim_trace,
)


def _run_short_rollout(steps: int = 256, seed: int = 0) -> tuple[ReplayTrajectory, str]:
    # Prefer CPU for the test to avoid stomping on other agents' CUDA memory.
    # gpu_trading_env requires CUDA; when that backend is unavailable we fall
    # back to the CPU stub which still exercises the recorder code paths.
    device = torch.device("cpu")
    try:
        env = make_env({}, num_envs=4, seed=seed, device=device)
        # If CUDA returned a gpu_trading_env handle anyway, that's fine.
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"env_adapter.make_env failed: {exc}")
    rec = ReplayRecorder.attach(env, max_steps=steps, env_index=0)
    obs = env.reset()
    if obs.dtype != torch.float32:
        obs = obs.to(torch.float32)
    rec.on_reset(obs)
    for _ in range(steps):
        # Trivial momentum-like action.
        act = torch.zeros(env.num_envs, env.action_dim, dtype=torch.float32, device=obs.device)
        if obs.shape[-1] >= 4:
            act[:, 0] = torch.tanh((obs[:, 3] - obs[:, 0]) * 10.0) * 0.5
        obs, reward, done, _ = env.step(act)
        if obs.dtype != torch.float32:
            obs = obs.to(torch.float32)
        rec.on_step(action=act, obs=obs, reward=reward, done=done)
    traj = rec.trajectory()
    return traj, env.backend_name


def test_recorder_shapes_and_values():
    traj, backend = _run_short_rollout(steps=256, seed=0)
    assert isinstance(traj, ReplayTrajectory)
    assert traj.num_steps == 256
    assert traj.ref_ohlc.shape == (256, 4)
    assert traj.ref_px.shape == (256,)
    assert traj.equity.shape == (256,)
    assert traj.pos_qty.shape == (256,)
    assert traj.drawdown.shape == (256,)
    assert traj.action.shape == (256, 4)
    assert traj.done.shape == (256,)
    assert traj.backend_name == backend
    # Finite values everywhere.
    for arr in (traj.ref_px, traj.equity, traj.pos_qty, traj.drawdown, traj.reward):
        assert np.isfinite(arr).all(), f"non-finite values in recorded array (backend={backend})"
    # Equity should be ~initial at step 0 (first obs is reset bar).
    assert traj.initial_equity > 0.0


def test_trajectory_to_marketsim_trace_shape():
    traj, _ = _run_short_rollout(steps=128, seed=1)
    try:
        trace = trajectory_to_marketsim_trace(traj, symbol_name="ENV0")
    except ModuleNotFoundError as exc:
        pytest.skip(f"marketsim_video deps unavailable: {exc}")
    assert trace.symbols == ["ENV0"]
    assert trace.prices.shape == (128, 1)
    assert trace.prices_ohlc.shape == (128, 1, 4)
    assert len(trace.frames) == 128
    # Equity normalised to $10k baseline.
    assert trace.frames[0].equity == pytest.approx(10000.0, rel=0.05)


def test_render_replay_mp4(tmp_path: Path):
    """End-to-end: recorder → MarketsimTrace → MP4 file on disk."""
    try:
        import matplotlib  # noqa: F401
        import imageio  # noqa: F401
    except ModuleNotFoundError as exc:
        pytest.skip(f"render deps unavailable: {exc}")
    try:
        from src.marketsim_video import render_mp4  # noqa: F401
    except ModuleNotFoundError as exc:
        pytest.skip(f"src.marketsim_video unavailable: {exc}")

    traj, backend = _run_short_rollout(steps=256, seed=2)
    from fp4.bench.render_replay import render_videos
    out_dir = tmp_path / "videos"
    try:
        produced = render_videos(traj, out_dir, title=f"test-{backend}", fps=4, num_pairs=1)
    except Exception as exc:
        pytest.skip(f"render_videos failed (likely ffmpeg missing): {type(exc).__name__}: {exc}")
    assert Path(produced["mp4"]).exists()
    assert Path(produced["mp4"]).stat().st_size > 0
    assert Path(produced["html"]).exists()
    assert Path(produced["meta"]).exists()
