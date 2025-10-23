import json
import sys
from pathlib import Path
from typing import List
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "fal" not in sys.modules:
    class _FalAppMeta(type):
        def __new__(mcls, cls_name, bases, namespace, **kwargs):
            cls = super().__new__(mcls, cls_name, bases, dict(namespace))
            cls._fal_options = kwargs
            return cls

    class _FalApp(metaclass=_FalAppMeta):
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def endpoint(cls, path: str):
            def decorator(func):
                return func

            return decorator

    def _fal_endpoint(path: str):
        def decorator(func):
            return func

        return decorator

    fal_stub = ModuleType("fal")
    fal_stub.App = _FalApp
    fal_stub.endpoint = _fal_endpoint
    sys.modules["fal"] = fal_stub

if "transformers" not in sys.modules:
    transformers_stub = ModuleType("transformers")
    transformers_stub.set_seed = lambda seed: None
    sys.modules["transformers"] = transformers_stub

import faltrain.app as fal_app


def test_grid_cartesian_product():
    space = fal_app.SweepSpace(
        learning_rates=[1e-3, 5e-4],
        batch_sizes=[64],
        context_lengths=[256],
        horizons=[30, 60],
        loss=["mse", "mae"],
        crypto_enabled=[False],
    )

    grid = fal_app._grid(space)

    assert len(grid) == 2 * 1 * 1 * 2 * 2 * 1
    first = grid[0]
    assert first["learning_rate"] == pytest.approx(1e-3)
    assert first["batch_size"] == 64
    assert first["context_length"] == 256
    assert first["horizon"] == 30
    assert first["loss"] == "mse"
    assert first["crypto_enabled"] is False
    assert "batch_size_plan" in first
    plan_first = first["batch_size_plan"]
    assert plan_first["initial"] == 64
    assert plan_first["fallbacks"] == [64]
    assert plan_first["candidates_desc"] == [64]

    last = grid[-1]
    assert last["learning_rate"] == pytest.approx(5e-4)
    assert last["batch_size"] == 64
    assert last["context_length"] == 256
    assert last["horizon"] == 60
    assert last["loss"] == "mae"
    assert last["crypto_enabled"] is False
    assert "batch_size_plan" in last
    plan_last = last["batch_size_plan"]
    assert plan_last["initial"] == 64
    assert plan_last["fallbacks"] == [64]
    assert plan_last["candidates_desc"] == [64]


def test_train_hf_selects_best_cfg_and_pushes_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(fal_app, "REPO_ROOT", tmp_path)

    repo_trainingdata = tmp_path / "trainingdata"
    repo_trainingdata.mkdir()

    monkeypatch.setenv("R2_ENDPOINT", "https://endpoint.test")
    monkeypatch.setenv("R2_BUCKET", "models")
    monkeypatch.setenv("WANDB_PROJECT", "stocks")
    monkeypatch.setenv("WANDB_ENTITY", "default")
    models_dir = tmp_path / "models_store"
    monkeypatch.setenv("FALTRAIN_MODELS_DIR", str(models_dir))

    sync_calls = []
    cp_calls = []
    pnl_calls = []
    metrics_values = [0.42, 0.11]

    def fake_aws_sync(src: str, dst: str, endpoint: str, delete: bool = False) -> None:
        sync_calls.append((src, dst, endpoint, delete))

    def fake_aws_cp(src: str, dst: str, endpoint: str, recursive: bool = False) -> None:
        cp_calls.append((src, dst, endpoint, recursive))

    def fake_run_hf_once(workdir: Path, req, cfg):
        value = metrics_values.pop(0)
        outdir = workdir / f"hf_{value:.2f}".replace(".", "_")
        outdir.mkdir(parents=True, exist_ok=True)
        metrics = {"val_loss": value}
        (outdir / "final_metrics.json").write_text(json.dumps(metrics))
        (outdir / "best_model.pt").write_bytes(b"unit-test-model")
        return metrics, outdir

    def fake_evaluate_pnl(model_dir: Path, include_crypto: bool, trainingdata_dir: Path):
        pnl_calls.append((model_dir, include_crypto, trainingdata_dir))
        base_return = 12.34 if not include_crypto else 8.76
        return {"mode": "mock", "crypto": include_crypto, "return_pct": base_return}

    monkeypatch.setattr(fal_app, "_aws_sync", fake_aws_sync)
    monkeypatch.setattr(fal_app, "_aws_cp", fake_aws_cp)
    monkeypatch.setattr(fal_app.StockTrainerApp, "_run_hf_once", staticmethod(fake_run_hf_once))
    monkeypatch.setattr(
        fal_app.StockTrainerApp, "_evaluate_pnl", staticmethod(fake_evaluate_pnl)
    )

    request = fal_app.TrainRequest(
        run_name="unit-test-run",
        trainer="hf",
        do_sweeps=True,
        sweeps=fal_app.SweepSpace(
            learning_rates=[1e-3, 5e-4],
            batch_sizes=[128],
            context_lengths=[512],
            horizons=[30],
            loss=["mse"],
            crypto_enabled=[False],
        ),
        output_root=str(tmp_path / "outputs"),
        trainingdata_prefix="trainingdata",
        epochs=2,
        parallel_trials=1,
    )

    app = fal_app.create_app()
    response = app.train(request)

    assert sync_calls == [("s3://models/trainingdata/", "/data/trainingdata/", "https://endpoint.test", False)]
    assert cp_calls[0][0].endswith("outputs/unit-test-run/")
    assert cp_calls[0][1] == "s3://models/checkpoints/unit-test-run/"
    assert cp_calls[0][3] is True

    assert len(cp_calls) == 2
    assert cp_calls[1][0].startswith(str(models_dir))
    assert cp_calls[1][1].startswith("s3://models/models/")
    assert cp_calls[1][3] is False

    assert response.run_name == "unit-test-run"
    assert len(response.sweep_results) == 2
    assert response.best_metrics["val_loss"] == pytest.approx(0.11)
    assert response.best_cfg["learning_rate"] == pytest.approx(5e-4)

    best_dir = Path(response.artifact_root.local_dir) / Path(
        response.sweep_results[1].artifacts.local_dir
    ).name
    assert best_dir.exists()

    assert len(pnl_calls) == len(response.sweep_results) * 2
    assert all(call[2] == Path("/data/trainingdata") for call in pnl_calls)
    assert response.pnl_summary["stock_only"]["crypto"] is False
    assert response.pnl_summary["stock_plus_crypto"]["crypto"] is True

    exported_files = list(models_dir.glob("*.pt"))
    assert len(exported_files) == 1
    exported_name = exported_files[0].name
    assert "loss0.11" in exported_name
    assert "pnl12.34" in exported_name


def test_batch_size_fallback_on_oom(monkeypatch, tmp_path):
    monkeypatch.setattr(fal_app, "REPO_ROOT", tmp_path)

    repo_trainingdata = tmp_path / "trainingdata"
    repo_trainingdata.mkdir()

    monkeypatch.setenv("R2_ENDPOINT", "https://endpoint.test")
    monkeypatch.setenv("R2_BUCKET", "models")

    selection = fal_app.BatchSizeSelection(
        signature="fake:141g",
        selected=512,
        descending_candidates=(512, 256, 128),
        user_candidates=(512, 256, 128),
        context_length=512,
        horizon=30,
        exhaustive=False,
    )
    monkeypatch.setattr(fal_app, "auto_tune_batch_sizes", lambda **_: selection)
    monkeypatch.setattr(fal_app, "get_cached_batch_selection", lambda **_: None)

    sync_calls = []
    cp_calls = []
    monkeypatch.setattr(fal_app, "_aws_sync", lambda *args, **kwargs: sync_calls.append(args))
    monkeypatch.setattr(fal_app, "_aws_cp", lambda *args, **kwargs: cp_calls.append(args))

    persisted: List[int] = []

    def fake_persist(selection_obj, *, batch_size=None):
        value = selection_obj.selected if batch_size is None else int(batch_size)
        persisted.append(value)

    monkeypatch.setattr(fal_app, "persist_batch_size", fake_persist)

    attempts: List[int] = []

    def fake_run_hf_once(workdir: Path, req, cfg):
        attempts.append(cfg["batch_size"])
        if cfg["batch_size"] == 512:
            raise fal_app.TrainingOOMError(["python"], "CUDA out of memory")
        outdir = workdir / f"hf_{cfg['batch_size']}"
        outdir.mkdir(parents=True, exist_ok=True)
        metrics = {"val_loss": 0.5}
        (outdir / "final_metrics.json").write_text(json.dumps(metrics))
        return metrics, outdir

    monkeypatch.setattr(fal_app.StockTrainerApp, "_run_hf_once", staticmethod(fake_run_hf_once))
    monkeypatch.setattr(fal_app.StockTrainerApp, "_evaluate_pnl", staticmethod(lambda *args, **kwargs: {}))

    request = fal_app.TrainRequest(
        run_name="oom-fallback",
        trainer="hf",
        do_sweeps=False,
        sweeps=fal_app.SweepSpace(
            learning_rates=[1e-3],
            batch_sizes=[512, 256, 128],
            context_lengths=[512],
            horizons=[30],
            loss=["mse"],
            crypto_enabled=[False],
        ),
        output_root=str(tmp_path / "outputs"),
        trainingdata_prefix="trainingdata",
        parallel_trials=1,
    )

    app = fal_app.create_app()
    response = app.train(request)

    assert attempts == [512, 256]
    assert persisted[-1] == 256
    assert response.best_cfg["batch_size"] == 256
    assert response.best_cfg["transaction_cost_bps"] == fal_app.DEFAULT_STOCK_TRADING_FEE_BPS


def test_run_toto_once_uses_runner(monkeypatch, tmp_path):
    monkeypatch.setattr(fal_app, "REPO_ROOT", tmp_path)

    data_root = tmp_path / "data"
    (data_root / "train").mkdir(parents=True)

    captured = {}

    def fake_run_training(**kwargs):
        captured.update(kwargs)
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = out_dir / "final_metrics.json"
        summary.write_text(json.dumps({"val": {"loss": 0.12}}))
        return {"val": {"loss": 0.12}}, summary

    monkeypatch.setattr("tototrainingfal.runner.run_training", fake_run_training)

    request = fal_app.TrainRequest(
        run_name="toto-run",
        trainer="toto",
        trainingdata_prefix=str(data_root),
        output_root=str(tmp_path / "out"),
        sweeps=fal_app.SweepSpace(
            learning_rates=[2e-4],
            batch_sizes=[32],
            context_lengths=[128],
            horizons=[24],
            loss=["huber"],
            crypto_enabled=[False],
        ),
        do_sweeps=False,
        parallel_trials=1,
        epochs=2,
    )

    app = fal_app.create_app()
    cfg = {
        "batch_size": 32,
        "context_length": 128,
        "horizon": 24,
        "loss": "huber",
        "learning_rate": 2e-4,
        "transaction_cost_bps": fal_app.DEFAULT_STOCK_TRADING_FEE_BPS,
        "crypto_enabled": False,
    }

    metrics, outdir = app._run_toto_once(Path(tmp_path / "artifacts"), request, cfg)

    assert metrics == {"val": {"loss": 0.12}}
    assert Path(outdir).exists()
    assert captured["context_length"] == 128
    assert captured["prediction_length"] == 24
    assert captured["batch_size"] == 32
    assert captured["epochs"] == 2
    assert captured["loss"] == "huber"
def test_setup_injects_training_modules(monkeypatch):
    torch_stub = SimpleNamespace(tag="torch")
    numpy_stub = SimpleNamespace(tag="numpy")
    calls = []

    for module_name in fal_app._TRAINING_INJECTION_MODULES:
        module = SimpleNamespace()

        def _make_hook(name: str):
            def _hook(torch_mod, numpy_mod):
                calls.append((name, torch_mod, numpy_mod))

            return _hook

        module.setup_training_imports = _make_hook(module_name)
        monkeypatch.setitem(sys.modules, module_name, module)

    fal_app._inject_training_modules(torch_stub, numpy_stub)

    expected = [
        (name, torch_stub, numpy_stub) for name in fal_app._TRAINING_INJECTION_MODULES
    ]
    assert calls == expected


def test_auto_tune_batch_sizes_prefers_feasible_candidate(monkeypatch, tmp_path):
    import faltrain.batch_size_tuner as tuner

    props = SimpleNamespace(total_memory=8 * 1024**3)

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_name(_):
            return "FakeGPU"

        @staticmethod
        def get_device_properties(_):
            return props

    torch_stub = SimpleNamespace(cuda=FakeCuda)

    monkeypatch.setattr(tuner, "_CACHE", {})
    monkeypatch.setattr(tuner, "_PERSIST_PATHS", (tmp_path / "best.json",))
    monkeypatch.setattr(tuner, "_PERSISTED", {})
    monkeypatch.setattr(tuner, "_load_torch", lambda: torch_stub)

    result = tuner.auto_tune_batch_sizes(
        candidates=[64, 128, 192, 256],
        context_lengths=[1024],
        horizons=[60],
        auto_tune=True,
        safety_margin=0.8,
    )

    assert isinstance(result, tuner.BatchSizeSelection)
    assert result.selected == 128
    assert result.descending_candidates == (256, 192, 128, 64)
    assert result.user_candidates == (64, 128, 192, 256)
    assert result.signature == "FakeGPU:8589934592"
    assert result.exhaustive is False
    assert result.fallback_values() == [128, 64]


def test_auto_tune_can_be_disabled(monkeypatch, tmp_path):
    import faltrain.batch_size_tuner as tuner

    monkeypatch.setattr(tuner, "_CACHE", {})
    monkeypatch.setattr(tuner, "_PERSIST_PATHS", (tmp_path / "best.json",))
    monkeypatch.setattr(tuner, "_PERSISTED", {})
    monkeypatch.setattr(tuner, "_load_torch", lambda: None)

    result = tuner.auto_tune_batch_sizes(
        candidates=[32, 16, 8],
        context_lengths=[10],
        horizons=[1],
        auto_tune=False,
        safety_margin=0.5,
    )
    assert isinstance(result, tuner.BatchSizeSelection)
    assert result.exhaustive is True
    assert result.sweep_values() == (32, 16, 8)
