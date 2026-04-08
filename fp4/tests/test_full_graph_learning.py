"""Regression test: `train_ppo` with `full_graph_capture=True` on the real
`gpu_trading_env` backend must actually learn (non-zero entropy), close
episodes (n_episodes > 0), produce a finite non-zero sortino, and leave a
`best.pt` checkpoint behind that `scripts/eval_100d.py` can consume.

This test guards against the regression where the full-graph branch built a
synthetic step (`build_synthetic_full_step`) instead of the real env step,
hard-coded `final_sortino=0.0` / `n_episodes=0` / `last_entropy=0.0`, and
never wrote a checkpoint.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from fp4 import trainer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_graph_real_env_learns(tmp_path: Path) -> None:
    try:
        import gpu_trading_env as gte  # noqa: F401
        if gte._load_ext() is None:
            pytest.skip("gpu_trading_env ext not built")
    except Exception:
        pytest.skip("gpu_trading_env not importable")

    cfg = {
        "full_graph_capture": True,
        "env": {
            "name": "stocks12_v5_rsi",
            "backend": "gpu_trading_env",
            "fee_rate": 0.001,
            "max_leverage": 5.0,
        },
        "ppo": {
            "hidden_size": 64,
            "num_envs": 64,
            "rollout_len": 256,
            "ppo_epochs": 1,
            "minibatch_size": 2048,
            "lr": 3e-4,
            "ent_coef": 0.05,
            "clip_eps": 0.2,
        },
    }
    ckpt_dir = tmp_path / "fg"
    metrics = trainer.train_ppo(cfg, total_timesteps=10_000, seed=0,
                                checkpoint_dir=str(ckpt_dir))

    assert metrics["full_graph_used"] is True
    assert metrics["n_episodes"] > 0, f"no episodes closed: {metrics}"
    assert metrics["last_entropy"] > 0.01, f"entropy collapsed: {metrics}"
    sortino = float(metrics["final_sortino"])
    assert math.isfinite(sortino) and sortino != 0.0, f"bad sortino: {metrics}"
    assert (ckpt_dir / "best.pt").exists(), "best.pt not written"
    # Checkpoint format sanity: scripts/eval_100d -> evaluate_policy_file ->
    # _find_state_dict expects one of {"policy", "model", ...} keys.
    payload = torch.load(ckpt_dir / "best.pt", map_location="cpu",
                         weights_only=False)
    assert "policy" in payload
    assert any("pi_head" in k or "policy_head" in k for k in payload["policy"].keys())
