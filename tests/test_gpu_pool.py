from __future__ import annotations

from pufferlib_market import gpu_pool


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
