import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
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
    assert grid[0] == {
        "learning_rate": pytest.approx(1e-3),
        "batch_size": 64,
        "context_length": 256,
        "horizon": 30,
        "loss": "mse",
        "crypto_enabled": False,
    }
    assert grid[-1] == {
        "learning_rate": pytest.approx(5e-4),
        "batch_size": 64,
        "context_length": 256,
        "horizon": 60,
        "loss": "mae",
        "crypto_enabled": False,
    }


def test_train_hf_selects_best_cfg_and_pushes_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(fal_app, "REPO_ROOT", tmp_path)

    repo_trainingdata = tmp_path / "trainingdata"
    repo_trainingdata.mkdir()

    monkeypatch.setenv("R2_ENDPOINT", "https://endpoint.test")
    monkeypatch.setenv("R2_BUCKET", "models")
    monkeypatch.setenv("WANDB_PROJECT", "stocks")
    monkeypatch.setenv("WANDB_ENTITY", "default")

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
        return metrics, outdir

    def fake_evaluate_pnl(model_dir: Path, include_crypto: bool, trainingdata_dir: Path):
        pnl_calls.append((model_dir, include_crypto, trainingdata_dir))
        return {"mode": "mock", "crypto": include_crypto}

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
    )

    app = fal_app.create_app()
    response = app.train(request)

    assert sync_calls == [("s3://models/trainingdata/", "/data/trainingdata/", "https://endpoint.test", False)]
    assert cp_calls[0][0].endswith("outputs/unit-test-run/")
    assert cp_calls[0][1] == "s3://models/checkpoints/unit-test-run/"
    assert cp_calls[0][3] is True

    assert response.run_name == "unit-test-run"
    assert len(response.sweep_results) == 2
    assert response.best_metrics["val_loss"] == pytest.approx(0.11)
    assert response.best_cfg["learning_rate"] == pytest.approx(5e-4)

    best_dir = Path(response.artifact_root.local_dir) / Path(
        response.sweep_results[1].artifacts.local_dir
    ).name
    assert best_dir.exists()

    assert len(pnl_calls) == 2
    assert all(call[2] == Path("/data/trainingdata") for call in pnl_calls)
    assert response.pnl_summary["stock_only"]["crypto"] is False
    assert response.pnl_summary["stock_plus_crypto"]["crypto"] is True


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


def test_auto_tune_batch_sizes_prefers_feasible_candidate(monkeypatch):
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
    monkeypatch.setattr(tuner, "_load_torch", lambda: torch_stub)

    result = tuner.auto_tune_batch_sizes(
        candidates=[64, 128, 192, 256],
        context_lengths=[1024],
        horizons=[60],
        auto_tune=True,
        safety_margin=0.8,
    )

    assert result == [128]


def test_auto_tune_can_be_disabled(monkeypatch):
    import faltrain.batch_size_tuner as tuner

    monkeypatch.setattr(tuner, "_CACHE", {})
    monkeypatch.setattr(tuner, "_load_torch", lambda: None)

    result = tuner.auto_tune_batch_sizes(
        candidates=[32, 16, 8],
        context_lengths=[10],
        horizons=[1],
        auto_tune=False,
        safety_margin=0.5,
    )
    assert result == [8, 16, 32]
