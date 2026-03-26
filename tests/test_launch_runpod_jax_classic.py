from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "launch_runpod_jax_classic.py"


def _load_module():
    spec = spec_from_file_location("launch_runpod_jax_classic", SCRIPT_PATH)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_sync_manifest_contains_expected_stock_files() -> None:
    mod = _load_module()
    manifest = mod.build_sync_manifest(
        symbols=["AAPL", "TSLA", "NVDA"],
        horizons=(1,),
        preload_path="unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt",
    )
    assert "trainingdatahourly/stocks/AAPL.csv" in manifest["data_files"]
    assert "unified_hourly_experiment/forecast_cache/h1/AAPL.parquet" in manifest["cache_files"]
    assert "binanceneural" in manifest["code_paths"]


def test_build_bootstrap_command_installs_cuda_jax() -> None:
    mod = _load_module()
    command = mod.build_bootstrap_command(remote_run_dir="/workspace/stock-prediction/analysis/remote_runs/test")
    assert "jax[cuda12]==0.9.2" in command
    assert "python3-pip" in command
    assert "torch_cuda.txt" in command
    assert "uv pip freeze" in command
    assert "jax_devices.txt" in command


def test_build_train_command_includes_wandb_and_status_outputs() -> None:
    mod = _load_module()
    command = mod.build_train_command(
        run_name="jax_run",
        remote_run_dir="/workspace/stock-prediction/analysis/remote_runs/jax_run",
        remote_log_path="/workspace/stock-prediction/analysis/remote_runs/jax_run/train.log",
        symbols=["AAPL", "TSLA"],
        horizons=(1,),
        preload_path="unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt",
        validation_days=7,
        epochs=3,
        batch_size=4,
        sequence_length=48,
        wandb_project="stock",
        wandb_entity="lee101p",
        wandb_group="jax_group",
        wandb_tags="runpod,jax",
        wandb_notes="note",
        wandb_mode="offline",
        dry_train_steps=1,
    )
    assert "--wandb-project stock" in command
    assert '--wandb-mode offline' in command
    assert "status.txt" in command
    assert "exit_code.txt" in command
    assert "train_jax_classic.py" in command
