from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


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


def test_build_docker_validate_command_includes_wandb_args(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    args = SimpleNamespace(
        run_name="jax_run",
        symbols="AAPL,TSLA,NVDA",
        preload="unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt",
        sequence_length=48,
        wandb_project="stock",
        wandb_entity="lee101p",
        wandb_group="docker_group",
        wandb_tags="docker,jax",
        wandb_notes="docker verify",
        wandb_mode="offline",
    )
    command = mod.build_docker_validate_command(args)
    rendered = " ".join(command)
    assert "WANDB_API_KEY=test-key" in rendered
    assert "WANDB_PROJECT=stock" in rendered
    assert "--wandb-project stock" in rendered
    assert "--wandb-group docker_group" in rendered
    assert "--wandb-mode offline" in rendered


def test_main_deletes_pod_when_startup_times_out(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    deleted: list[str] = []

    args = SimpleNamespace(
        run_name="timeout_case",
        symbols="AAPL,TSLA",
        forecast_horizons="1",
        preload="unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt",
        gpu_type=mod.DEFAULT_GPU_TYPE,
        image=mod.DEFAULT_DOCKER_IMAGE,
        validation_days=7,
        epochs=1,
        batch_size=2,
        sequence_length=48,
        dry_train_steps=1,
        wandb_project=None,
        wandb_entity=None,
        wandb_group=None,
        wandb_tags="",
        wandb_notes=None,
        wandb_mode="offline",
        key_path=tmp_path / "id_ed25519",
        startup_timeout_sec=1,
        poll_interval_sec=15,
        output_root=tmp_path / "runs",
        docker_validate=False,
        dry_run=False,
        detach=False,
        keep_pod=False,
    )

    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setattr(mod, "parse_args", lambda: args)
    monkeypatch.setattr(mod, "create_pod", lambda api_key, **kwargs: "pod-123")
    monkeypatch.setattr(mod, "wait_for_public_ssh", lambda api_key, pod_id, timeout_sec: (_ for _ in ()).throw(TimeoutError("timed out")))
    monkeypatch.setattr(mod, "delete_pod", lambda api_key, pod_id: deleted.append(pod_id))
    monkeypatch.setattr(
        mod,
        "build_sync_manifest",
        lambda **kwargs: {"code_paths": [], "data_files": [], "cache_files": [], "checkpoint_files": []},
    )

    try:
        mod.main()
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected TimeoutError")

    assert deleted == ["pod-123"]


def test_main_dry_run_skips_docker_validate(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    called = {"docker": False}
    args = SimpleNamespace(
        run_name="dry_run_case",
        symbols="AAPL,TSLA",
        forecast_horizons="1",
        preload="unified_hourly_experiment/checkpoints/wd_0.06_s42/epoch_020.pt",
        gpu_type=mod.DEFAULT_GPU_TYPE,
        image=mod.DEFAULT_DOCKER_IMAGE,
        validation_days=7,
        epochs=1,
        batch_size=2,
        sequence_length=48,
        dry_train_steps=1,
        wandb_project=None,
        wandb_entity=None,
        wandb_group=None,
        wandb_tags="",
        wandb_notes=None,
        wandb_mode="offline",
        key_path=tmp_path / "id_ed25519",
        startup_timeout_sec=1,
        poll_interval_sec=15,
        output_root=tmp_path / "runs",
        docker_validate=True,
        dry_run=True,
        detach=False,
        keep_pod=False,
    )

    monkeypatch.setattr(mod, "parse_args", lambda: args)
    monkeypatch.setattr(
        mod,
        "build_sync_manifest",
        lambda **kwargs: {"code_paths": [], "data_files": [], "cache_files": [], "checkpoint_files": []},
    )
    monkeypatch.setattr(mod, "docker_validate", lambda _args: called.__setitem__("docker", True))

    rc = mod.main()

    assert rc == 0
    assert called["docker"] is False
