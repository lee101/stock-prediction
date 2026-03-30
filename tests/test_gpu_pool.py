from __future__ import annotations

from pathlib import Path

import pytest

from pufferlib_market import gpu_pool
from pufferlib_market.autoresearch_rl import TrialConfig


def test_stocks12_fidelity_longrun_preset_exists() -> None:
    assert "stocks12_fidelity_longrun" in gpu_pool.PRESETS


def test_stocks12_fidelity_longrun_preset_uses_high_fidelity_flags() -> None:
    jobs = gpu_pool.PRESETS["stocks12_fidelity_longrun"]
    assert len(jobs) == 6
    for job in jobs:
        assert job["use_bf16"] is False
        assert job["no_tf32"] is True
        assert job["no_cuda_graph"] is True
        assert job["periods_per_year"] == 252.0
        assert job["max_steps"] == 252
        assert job["time_budget_override"] == 1800


def test_resolve_baseline_prune_settings_auto_enables_stock_floor() -> None:
    settings = gpu_pool.resolve_baseline_prune_settings(
        baseline_profile="auto",
        stocks_mode=True,
        baseline_val_return_floor=None,
        baseline_combined_floor=None,
        projection_clip_abs=None,
    )

    assert settings["profile_name"] == "stocks_daily_candidate"
    assert settings["baseline_val_return_floor"] == pytest.approx(0.35)
    assert settings["baseline_combined_floor"] == pytest.approx(1.0)


def test_worker_loop_passes_baseline_settings_to_run_trial(monkeypatch, tmp_path: Path) -> None:
    captured: list[dict[str, float]] = []
    pending_job = {"id": "job1", "overrides": {"description": "trial1"}}
    state = {"claimed": False}

    def _fake_claim_next_job(queue_path: Path, worker_id: str):
        if not state["claimed"]:
            state["claimed"] = True
            return pending_job
        return None

    def _fake_run_trial(*args, **kwargs):
        captured.append(
            {
                "baseline_val_return_floor": kwargs["baseline_val_return_floor"],
                "baseline_combined_floor": kwargs["baseline_combined_floor"],
                "projection_clip_abs": kwargs["projection_clip_abs"],
            }
        )
        return {
            "val_return": 0.1,
            "val_sortino": 0.2,
            "val_wr": 0.55,
            "rank_score": 0.3,
        }

    monkeypatch.setattr(gpu_pool, "claim_next_job", _fake_claim_next_job)
    monkeypatch.setattr(gpu_pool, "mark_job_done", lambda *args, **kwargs: None)
    monkeypatch.setattr(gpu_pool, "append_leaderboard_row", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gpu_pool,
        "build_config",
        lambda overrides: TrialConfig(description=overrides["description"]),
    )
    monkeypatch.setattr(gpu_pool, "run_trial", _fake_run_trial)
    monkeypatch.setattr(gpu_pool.time, "sleep", lambda *args, **kwargs: None)

    gpu_pool.worker_loop(
        gpu_id=0,
        queue_path=tmp_path / "queue.jsonl",
        leaderboard_path=tmp_path / "leaderboard.csv",
        train_data="train.bin",
        val_data="val.bin",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        time_budget=60,
        holdout_data=None,
        holdout_n_windows=0,
        holdout_eval_steps=0,
        holdout_fill_buffer_bps=5.0,
        stocks_mode=True,
        baseline_profile="auto",
        baseline_val_return_floor=None,
        baseline_combined_floor=1.1,
        projection_clip_abs=5.0,
    )

    assert len(captured) == 1
    assert captured[0]["baseline_val_return_floor"] == pytest.approx(0.35)
    assert captured[0]["baseline_combined_floor"] == pytest.approx(1.1)
    assert captured[0]["projection_clip_abs"] == pytest.approx(5.0)
