from __future__ import annotations

import pandas as pd
import pytest

from fastalgorithms.eth_risk_ppo.reproducibility_sweep import (
    build_sweep_run_specs,
    resolve_mode_overrides,
    summarize_results,
)
from fastalgorithms.eth_risk_ppo.retune_recent_eth import build_default_candidates


def test_resolve_mode_overrides_applies_threads_to_deterministic_modes() -> None:
    overrides = resolve_mode_overrides("deterministic_cuda", torch_num_threads=1)

    assert overrides["DEVICE"] == "cuda"
    assert overrides["POLICY_DTYPE"] == "float32"
    assert overrides["DETERMINISTIC_TRAINING"] == "1"
    assert overrides["DISABLE_TF32"] == "1"
    assert overrides["TORCH_NUM_THREADS"] == "1"


def test_resolve_mode_overrides_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="Unsupported mode"):
        resolve_mode_overrides("not_a_mode")


def test_build_sweep_run_specs_expands_modes_repeats_and_families() -> None:
    base_candidates = build_default_candidates(seeds=[42])

    runs = build_sweep_run_specs(
        base_candidates=base_candidates,
        families=["chronos2_h6_ctx1024"],
        modes=["fast_cuda", "deterministic_cuda"],
        repeats=2,
        torch_num_threads=1,
    )

    assert len(runs) == 4
    assert {run.mode for run in runs} == {"fast_cuda", "deterministic_cuda"}
    assert {run.repeat for run in runs} == {1, 2}
    assert all(run.family == "chronos2_h6_ctx1024" for run in runs)
    assert all(run.candidate.seed == 42 for run in runs)
    assert all(run.candidate.env_overrides["SEED"] == "42" for run in runs)
    assert any(run.candidate.env_overrides["DETERMINISTIC_TRAINING"] == "1" for run in runs if run.mode == "deterministic_cuda")


def test_summarize_results_penalizes_noisier_modes() -> None:
    results = pd.DataFrame(
        [
            {
                "family": "chronos2_h6_ctx1024",
                "mode": "deterministic_cuda",
                "seed": 42,
                "repeat": 1,
                "robust_score": 20.0,
                "long_return": 0.03,
                "long_sortino": 1.0,
                "meta_return_pct": 6.0,
                "meta_sortino": 1.1,
                "meta_max_drawdown_pct": 3.0,
                "pnl_smoothness": 0.0010,
                "eval_reward_delta_std": 0.2,
            },
            {
                "family": "chronos2_h6_ctx1024",
                "mode": "deterministic_cuda",
                "seed": 42,
                "repeat": 2,
                "robust_score": 20.0,
                "long_return": 0.03,
                "long_sortino": 1.0,
                "meta_return_pct": 6.0,
                "meta_sortino": 1.1,
                "meta_max_drawdown_pct": 3.0,
                "pnl_smoothness": 0.0010,
                "eval_reward_delta_std": 0.2,
            },
            {
                "family": "chronos2_h6_ctx1024",
                "mode": "fast_cuda",
                "seed": 42,
                "repeat": 1,
                "robust_score": 22.0,
                "long_return": 0.04,
                "long_sortino": 1.2,
                "meta_return_pct": 8.0,
                "meta_sortino": 1.3,
                "meta_max_drawdown_pct": 4.0,
                "pnl_smoothness": 0.0100,
                "eval_reward_delta_std": 1.4,
            },
            {
                "family": "chronos2_h6_ctx1024",
                "mode": "fast_cuda",
                "seed": 42,
                "repeat": 2,
                "robust_score": 15.0,
                "long_return": 0.01,
                "long_sortino": 0.4,
                "meta_return_pct": 1.0,
                "meta_sortino": 0.2,
                "meta_max_drawdown_pct": 8.0,
                "pnl_smoothness": 0.0100,
                "eval_reward_delta_std": 1.6,
            },
        ]
    )

    summary, repeat_summary = summarize_results(results)

    assert list(summary["mode"]) == ["deterministic_cuda", "fast_cuda"]
    det = summary.iloc[0]
    noisy = summary.iloc[1]
    assert det["repeat_robust_span_mean"] == pytest.approx(0.0)
    assert noisy["repeat_robust_span_mean"] == pytest.approx(7.0)
    assert det["stability_score"] > noisy["stability_score"]
    assert set(repeat_summary["mode"]) == {"deterministic_cuda", "fast_cuda"}
