from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from benchmark_chronos2 import (
    Chronos2Candidate,
    CandidateReport,
    EvaluationResult,
    _build_cross_product_candidates,
    _build_direct_search_space,
    _maybe_update_hyperparams,
    _parse_bool_tokens,
)


def _args(**overrides):
    defaults = dict(
        context_lengths=[256],
        batch_sizes=[32],
        aggregations=["median"],
        sample_counts=[0],
        scalers=["none"],
        multivariate_modes=[False, True],
        quantile_levels=[0.1, 0.5, 0.9],
        max_candidates=None,
        auto_context_lengths=False,
        auto_context_min=128,
        auto_context_max=4096,
        auto_context_step=128,
        auto_context_guard=40,
        direct_batch_sizes=None,
        direct_aggregations=None,
        direct_sample_counts=None,
        direct_scalers=None,
        direct_multivariate_modes=None,
        update_hyperparams=True,
        hyperparam_root="hyperparams",
        chronos2_subdir="hourly",
        model_id="amazon/chronos-2",
        device_map="cuda",
        force_update=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_parse_bool_tokens_accepts_mixed_styles():
    assert _parse_bool_tokens(["false", "1", "off"], flag_name="--modes") == (False, True)


def test_build_cross_product_candidates_sweeps_multivariate_modes():
    candidates = _build_cross_product_candidates(_args(), {})
    assert [candidate.use_multivariate for candidate in candidates] == [False, True]
    assert candidates[0].name.endswith("uni")
    assert candidates[1].name.endswith("mv")


def test_build_direct_search_space_keeps_multivariate_modes():
    space = _build_direct_search_space(series_length=600, args=_args())
    assert space.multivariate_modes == (False, True)


def test_maybe_update_hyperparams_persists_use_multivariate(tmp_path: Path):
    args = _args(hyperparam_root=str(tmp_path), chronos2_subdir="hourly")
    report = CandidateReport(
        symbol="AAPL",
        candidate=Chronos2Candidate(
            name="ctx256_bs32_median_mv",
            context_length=256,
            batch_size=32,
            aggregation="median",
            sample_count=0,
            scaler="none",
            use_multivariate=True,
        ),
        validation=EvaluationResult(
            price_mae=1.0,
            rmse=1.1,
            pct_return_mae=0.01,
            latency_s=0.5,
            mae_percent=0.4,
            predictions=[1.0],
        ),
        test=EvaluationResult(
            price_mae=1.2,
            rmse=1.3,
            pct_return_mae=0.02,
            latency_s=0.6,
            mae_percent=0.5,
            predictions=[1.2],
        ),
        windows={"val_window": 20, "test_window": 20, "forecast_horizon": 1},
    )

    _maybe_update_hyperparams([report], args)

    payload = json.loads((tmp_path / "chronos2" / "hourly" / "AAPL.json").read_text())
    assert payload["config"]["use_multivariate"] is True
