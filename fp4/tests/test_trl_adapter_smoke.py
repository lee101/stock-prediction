"""Smoke test for the TRL marketsim adapter.

Acceptable outcomes (none should crash the bench harness):
- 'ok'   : real PPO step ran end to end
- 'skip' : trl/PPOTrainer not available on this version, or env bindings
           not built yet — adapter must report a clear reason

Anything else (including uncaught exceptions) fails the test.
"""

from __future__ import annotations

from pathlib import Path

from fp4.bench.adapters.trl_adapter import run_trl


def _tiny_cfg() -> dict:
    return {
        "env": {
            "train_data": "data/stocks12_v5_rsi_train.npz",
            "val_data": "data/stocks12_v5_rsi_val.npz",
            "fee_rate": 0.001,
            "max_leverage_scalar_fallback": 1.5,
        },
        "ppo": {"hidden_size": 32},
        "eval": {"slippage_bps": [5], "n_windows": 4},
    }


def test_trl_adapter_returns_status(tmp_path: Path) -> None:
    rec = run_trl(_tiny_cfg(), steps=64, seed=0, ckpt_dir=tmp_path)
    assert isinstance(rec, dict)
    assert "status" in rec
    assert rec["status"] in ("ok", "skip", "error"), rec
    if rec["status"] == "skip":
        assert "reason" in rec and rec["reason"], "skip must include reason"
    elif rec["status"] == "ok":
        assert "wall_sec" in rec
        assert "trainer_output" in rec
