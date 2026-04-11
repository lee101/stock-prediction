"""Smoke test for the HF Trainer marketsim adapter.

The adapter is allowed to *skip* with a clear reason if any dependency is
unavailable (transformers missing, pufferlib_market binding not built, train
data not present).  In all other cases it must complete an offline rollout +
a few PPO update epochs and return finite metrics.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _tiny_cfg() -> dict:
    return {
        "env": {
            "train_data": "pufferlib_market/data/stocks12_daily_v5_rsi_train.bin",
            "val_data": "pufferlib_market/data/stocks12_daily_v5_rsi_val.bin",
            "fee_rate": 0.001,
            "max_leverage_scalar_fallback": 1.5,
        },
        "ppo": {
            "hidden_size": 64,
            "lr": 3.0e-4,
            "clip_eps": 0.2,
            "ent_coef": 0.02,
            "num_envs": 4,
            "rollout_len": 32,
            "ppo_epochs": 1,
            "minibatch_size": 32,
        },
    }


def test_hf_adapter_smoke(tmp_path):
    pytest.importorskip("torch")
    from fp4.bench.adapters import hf_adapter

    rec = hf_adapter.run(cfg=_tiny_cfg(), steps=128, seed=0, ckpt_dir=tmp_path)
    assert isinstance(rec, dict)
    assert "status" in rec
    if rec["status"] == "skip":
        pytest.skip(f"hf_adapter skipped: {rec.get('reason', '?')}")
    assert rec["status"] == "ok", rec
    import math

    assert math.isfinite(rec["sortino"]), rec
    assert math.isfinite(rec["mean_episode_return"]), rec
    assert rec["samples"] > 0
    assert rec["global_step"] >= 1
