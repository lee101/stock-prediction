from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    path = Path("scripts/refit_chronos2_pair_configs.py")
    spec = importlib.util.spec_from_file_location("refit_chronos2_pair_configs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    defaults = dict(
        frequency="hourly",
        asset_kind="stocks",
        run_id="test_run",
        preaug_selection_metric="mae_percent",
        preaug_strategies=["baseline", "rolling_norm"],
        context_lengths=[256, 512],
        batch_sizes=[32, 64],
        aggregations=["median"],
        sample_counts=[0],
        scalers=["none", "meanstd"],
        multivariate_modes=["false", "true"],
        device_map="cuda",
        torch_dtype=None,
        torch_compile=False,
        pipeline_backend="chronos",
        predict_batches_jointly=False,
        search_method="grid",
        lora_preaugs=["baseline", "rolling_norm"],
        lora_context_lengths=[128, 256],
        lora_learning_rates=["1e-5", "5e-5"],
        lora_rs=[8, 16],
        lora_batch_size=32,
        lora_num_steps=1000,
        lora_prediction_length=24,
        lora_improvement_threshold=5.0,
        promote_selection_strategy="stable_family",
        promote_metric="val_mae_percent",
        python_bin="python",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_resolve_paths_hourly_stocks():
    mod = _load_module()
    paths = mod._resolve_paths(frequency="hourly", asset_kind="stocks", report_root=Path("reports/refits"))
    assert paths.data_dir == Path("trainingdatahourly/stocks")
    assert paths.hyperparam_dir == Path("hyperparams/chronos2/hourly")
    assert paths.preaug_dir == Path("preaugstrategies/chronos2/hourly")


def test_build_benchmark_command_includes_multivariate_modes():
    mod = _load_module()
    paths = mod._resolve_paths(frequency="hourly", asset_kind="stocks", report_root=Path("reports/refits"))
    cmd = mod.build_benchmark_command(
        python_bin="python",
        symbols=["AAPL", "MSFT"],
        paths=paths,
        args=_args(),
    )
    assert "--multivariate-modes" in cmd
    idx = cmd.index("--multivariate-modes")
    assert cmd[idx + 1 : idx + 3] == ["false", "true"]
    assert "--chronos2-subdir" in cmd


def test_build_preaug_and_lora_commands_use_frequency_specific_paths():
    mod = _load_module()
    paths = mod._resolve_paths(frequency="hourly", asset_kind="stocks", report_root=Path("reports/refits"))
    args = _args()

    preaug_cmd = mod.build_preaug_command(
        python_bin="python",
        symbols=["AAPL"],
        paths=paths,
        args=args,
    )
    assert "preaug_sweeps/evaluate_preaug_chronos.py" in preaug_cmd
    assert str(paths.hyperparam_dir) in preaug_cmd
    assert str(paths.preaug_dir) in preaug_cmd

    lora_cmd = mod.build_lora_command(
        python_bin="python",
        symbols=["AAPL"],
        paths=paths,
        args=args,
    )
    assert "scripts/chronos2_lora_improvement_sweep.py" in lora_cmd
    assert str(paths.data_dir) in lora_cmd
    assert str(paths.lora_results_dir) in lora_cmd
