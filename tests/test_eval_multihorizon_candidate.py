from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "eval_multihorizon_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_multihorizon_candidate", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_start_indices_is_deterministic():
    module = _load_module()
    starts_a = module.build_start_indices(
        num_timesteps=263,
        eval_days=100,
        n_windows=12,
        seed=1337,
    )
    starts_b = module.build_start_indices(
        num_timesteps=263,
        eval_days=100,
        n_windows=12,
        seed=1337,
    )
    assert starts_a == starts_b
    assert len(starts_a) == 12
    assert starts_a == sorted(starts_a)
    assert len(set(starts_a)) == len(starts_a)


def test_build_start_indices_recent_tail_bounds():
    module = _load_module()
    starts = module.build_start_indices(
        num_timesteps=263,
        eval_days=60,
        n_windows=10,
        seed=7,
        recent_within_days=90,
    )
    assert len(starts) == 10
    assert min(starts) >= 112
    assert max(starts) <= 202


def test_choose_recommendation_flags_promising_additive():
    module = _load_module()
    report = {
        "scenarios": {
            "baseline": {
                "aggregate": {
                    "worst_cell": {"median_monthly_return": 0.10},
                }
            },
            "candidate": {
                "aggregate": {
                    "worst_cell": {"median_monthly_return": 0.09},
                }
            },
            "baseline_plus_candidate": {
                "aggregate": {
                    "worst_cell": {"median_monthly_return": 0.13},
                }
            },
        },
        "comparisons": {
            "candidate_vs_baseline": {
                "mean_delta_median_monthly_return": -0.01,
                "mean_delta_negative_windows": 0.5,
            },
            "baseline_plus_candidate_vs_baseline": {
                "mean_delta_median_monthly_return": 0.03,
                "mean_delta_negative_windows": -1.0,
            },
        },
    }
    rec = module.choose_recommendation(report)
    assert rec["status"] == "promising_additive"


def test_choose_recommendation_rejects_unproven_candidate():
    module = _load_module()
    report = {
        "scenarios": {
            "baseline": {
                "aggregate": {
                    "worst_cell": {"median_monthly_return": 0.11},
                }
            },
            "candidate": {
                "aggregate": {
                    "worst_cell": {"median_monthly_return": 0.08},
                }
            },
            "baseline_plus_candidate": {
                "aggregate": {
                    "worst_cell": {"median_monthly_return": 0.10},
                }
            },
        },
        "comparisons": {
            "candidate_vs_baseline": {
                "mean_delta_median_monthly_return": -0.02,
                "mean_delta_negative_windows": 1.0,
            },
            "baseline_plus_candidate_vs_baseline": {
                "mean_delta_median_monthly_return": -0.01,
                "mean_delta_negative_windows": 0.25,
            },
        },
    }
    rec = module.choose_recommendation(report)
    assert rec["status"] == "not_proven"
