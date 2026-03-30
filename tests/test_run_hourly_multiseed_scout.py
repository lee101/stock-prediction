from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

from unified_hourly_experiment.jax_classic_defaults import DEFAULT_JAX_CLASSIC_PRELOAD


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "run_hourly_multiseed_scout.py"


def _load_module():
    spec = spec_from_file_location("run_hourly_multiseed_scout", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_combos_cross_product() -> None:
    mod = _load_module()
    args = SimpleNamespace(
        learning_rates="1e-4,3e-4",
        weight_decays="0.0,0.05",
        return_weights="0.08",
        fill_temperatures="0.0005,0.001",
    )
    combos = mod.build_combos(args)
    labels = [mod.combo_label(combo.as_dict()) for combo in combos]

    assert len(combos) == 8
    assert "lr_0p0001_wd_0p0_rw_0p08_ft_0p0005" in labels
    assert "lr_0p0003_wd_0p05_rw_0p08_ft_0p001" in labels


def test_compare_metric_requires_margin() -> None:
    mod = _load_module()
    comparison = mod.compare_metric(
        candidate=[20.0, 21.0, 22.0],
        baseline=[18.0, 18.5, 19.0],
        higher_is_better=True,
        abs_tolerance=1.0,
    )

    assert comparison["delta_mean"] > 0
    assert comparison["significance_margin"] >= 1.0
    assert comparison["significant"] is True


def test_summarize_config_runs_adds_baseline_comparison() -> None:
    mod = _load_module()
    combo = mod.ExperimentCombo(
        learning_rate=3e-4,
        weight_decay=0.0,
        return_weight=0.08,
        fill_temperature=5e-4,
    )
    args = SimpleNamespace(
        market_return_weight=0.10,
        market_std_penalty=1.0,
        market_drawdown_penalty=0.25,
        sortino_tolerance=1.0,
        return_tolerance_pct=0.5,
        drawdown_tolerance_pct=0.25,
    )
    candidate_runs = [
        {
            "seed": 1337,
            "train": {"checkpoint_metric": 17.0},
            "backtest": {"avg_sortino": 20.0, "avg_return": 3.0, "avg_max_dd": -2.0},
        },
        {
            "seed": 42,
            "train": {"checkpoint_metric": 18.0},
            "backtest": {"avg_sortino": 21.0, "avg_return": 3.2, "avg_max_dd": -2.1},
        },
        {
            "seed": 7,
            "train": {"checkpoint_metric": 19.0},
            "backtest": {"avg_sortino": 22.0, "avg_return": 3.1, "avg_max_dd": -1.9},
        },
    ]
    baseline_runs = [
        {
            "seed": 1337,
            "train": {"checkpoint_metric": 14.0},
            "backtest": {"avg_sortino": 16.0, "avg_return": 2.0, "avg_max_dd": -2.1},
        },
        {
            "seed": 42,
            "train": {"checkpoint_metric": 15.0},
            "backtest": {"avg_sortino": 17.0, "avg_return": 2.1, "avg_max_dd": -2.0},
        },
        {
            "seed": 7,
            "train": {"checkpoint_metric": 16.0},
            "backtest": {"avg_sortino": 18.0, "avg_return": 2.2, "avg_max_dd": -1.8},
        },
    ]

    summary = mod.summarize_config_runs(
        combo=combo,
        runs=candidate_runs,
        baseline_runs=baseline_runs,
        args=args,
    )

    assert summary["market_robust_score"] > 0
    assert summary["market_avg_sortino"]["mean"] == 21.0
    assert summary["baseline_comparison"]["sortino"]["significant"] is True


def test_parse_args_uses_shared_default_preload() -> None:
    mod = _load_module()
    args = mod.parse_args([])
    assert args.preload == DEFAULT_JAX_CLASSIC_PRELOAD
