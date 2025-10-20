#!/usr/bin/env python3
"""
FAL training orchestration for the production trading stack.

Features:
    * Sync training data from R2 on startup, push artifacts back on completion.
    * Inject heavy dependencies into the training packages so imports stay fast.
    * Run HF / Toto / Puffer trainers sequentially on a GPU-H200 with sweep support.
    * Evaluate PnL for stock-only and stock+crypto configurations.
    * Upload best checkpoints + logs and return a JSON summary.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import time
import uuid
from pathlib import Path
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fal
from pydantic import BaseModel, Field

from faltrain.artifacts import load_artifact_specs, sync_artifacts
from faltrain.batch_size_tuner import auto_tune_batch_sizes
REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("faltrain.app")
LOG.setLevel(logging.INFO)

_TRAINING_INJECTION_MODULES: Tuple[str, ...] = (
    "hftraining.injection",
    "tototraining.injection",
    "pufferlibtraining.injection",
)


def _env(key: str, default: Optional[str] = None) -> str:
    value = os.getenv(key, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


def _run(
    cmd: List[str],
    *,
    cwd: Path = REPO_ROOT,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    LOG.info("• $ %s", " ".join(cmd))
    run_kwargs: Dict[str, Any] = {
        "cwd": str(cwd),
        "env": env,
        "check": check,
        "text": True,
    }
    if capture:
        run_kwargs["stdout"] = subprocess.PIPE
        run_kwargs["stderr"] = subprocess.STDOUT
    return subprocess.run(cmd, **run_kwargs)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _aws_sync(src: str, dst: str, endpoint: str, delete: bool = False) -> None:
    args = ["aws", "s3", "sync", src, dst, "--endpoint-url", endpoint]
    if delete:
        args.append("--delete")
    _run(args)


def _aws_cp(src: str, dst: str, endpoint: str, recursive: bool = False) -> None:
    args = ["aws", "s3", "cp", src, dst, "--endpoint-url", endpoint]
    if recursive:
        args.append("--recursive")
    _run(args)


def _maybe_seed(seed: Optional[int]) -> int:
    value = seed if seed is not None else random.randint(1, 10_000_000)
    random.seed(value)
    return value


def _inject_training_modules(
    torch_mod: Any,
    numpy_mod: Any,
    *,
    module_names: Iterable[str] = _TRAINING_INJECTION_MODULES,
) -> None:
    for mod_path in module_names:
        try:
            module = import_module(mod_path)
        except ImportError:
            continue
        setup_fn = getattr(module, "setup_training_imports", None)
        if callable(setup_fn):
            setup_fn(torch_mod, numpy_mod)
            LOG.info("Injected torch/numpy into %s", mod_path)


def _grid(space: "SweepSpace") -> List[Dict[str, Any]]:
    grid: List[Dict[str, Any]] = []
    batch_sizes = auto_tune_batch_sizes(
        candidates=space.batch_sizes,
        context_lengths=space.context_lengths,
        horizons=space.horizons,
        auto_tune=space.auto_tune_batch_size,
        safety_margin=space.batch_size_safety_margin,
    )
    for lr in space.learning_rates:
        for bs in batch_sizes:
            for cl in space.context_lengths:
                for horizon in space.horizons:
                    for loss in space.loss:
                        for crypto_enabled in space.crypto_enabled:
                            grid.append(
                                {
                                    "learning_rate": lr,
                                    "batch_size": bs,
                                    "context_length": cl,
                                    "horizon": horizon,
                                    "loss": loss,
                                    "crypto_enabled": crypto_enabled,
                                }
                            )
    return grid


class SweepSpace(BaseModel):
    learning_rates: List[float] = [7e-4, 3e-4]
    batch_sizes: List[int] = [128, 256]
    context_lengths: List[int] = [512, 1024]
    horizons: List[int] = [30, 60]
    loss: List[str] = ["mse"]
    crypto_enabled: List[bool] = [False, True]
    auto_tune_batch_size: bool = True
    batch_size_safety_margin: float = 0.8


class TrainRequest(BaseModel):
    run_name: str = Field(default_factory=lambda: f"run_{int(time.time())}")
    trainer: str = Field("hf", description="hf | toto | puffer")
    trainingdata_prefix: str = "trainingdata"
    output_root: str = "/data/experiments"
    do_sweeps: bool = False
    sweeps: SweepSpace = SweepSpace()
    symbols: List[str] = ["SPY"]
    val_days: int = 30
    epochs: int = 2
    transaction_cost_bps: int = 1
    seed: Optional[int] = None


class RunArtifact(BaseModel):
    local_dir: str
    r2_uri: Optional[str] = None
    metrics_path: Optional[str] = None


class SweepResult(BaseModel):
    cfg: Dict[str, Any]
    metrics: Dict[str, Any]
    artifacts: RunArtifact


class TrainResponse(BaseModel):
    run_name: str
    best_cfg: Dict[str, Any]
    best_metrics: Dict[str, Any]
    sweep_results: List[SweepResult]
    pnl_summary: Dict[str, Any]
    artifact_root: RunArtifact


class StockTrainerApp(
    fal.App,
    name="stock-trainer",
    min_concurrency=0,
    max_concurrency=1,
    keep_alive=120,
):
    machine_type = "GPU-H200"
    python_version = "3.12"
    requirements = [
        "fal-client",
        "torch>=2.8.0",
        "torchvision>=0.21.0",
        "numpy>=1.24.4,<2",
        "pandas",
        "tqdm",
        "rich",
        "orjson",
        "awscli",
        "boto3",
        "huggingface-hub[cli]",
        "safetensors",
        "transformers>=4.45.0",
        "accelerate>=1.2.0",
        "bitsandbytes>=0.44.0",
        "xformers",
        "wandb",
    ]
    local_python_modules: List[str] = [
        "faltrain",
        "hftraining",
        "hfshared",
        "hfinference",
        "traininglib",
        "marketsimulator",
        "differentiable_market",
        "toto",
        "pufferlibtraining",
        "pufferlibinference",
        "gymrl",
        "src",
    ]

    def setup(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

        import torch as _torch
        import numpy as _np

        # CUDA / transformer knobs tuned for H200 workloads.
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:1024,expandable_segments:True"
        )
        try:
            _torch.backends.cuda.matmul.allow_tf32 = True
            _torch.backends.cudnn.allow_tf32 = True
            _torch.backends.cudnn.benchmark = True
            _torch.backends.cuda.enable_flash_sdp(True)
            _torch.backends.cuda.enable_math_sdp(True)
            _torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass

        LOG.info("CUDA device: %s", _torch.cuda.get_device_name(0))
        LOG.info("torch version: %s", _torch.__version__)

        # Offer dependency injection to the in-repo training stacks.
        _inject_training_modules(_torch, _np)

        _ensure_dir(Path("/data"))
        _ensure_dir(Path("/data/trainingdata"))
        _ensure_dir(Path("/data/experiments"))
        self._prefetch_reference_artifacts()

    def _prefetch_reference_artifacts(self) -> None:
        try:
            endpoint = _env("R2_ENDPOINT")
        except RuntimeError as exc:
            LOG.warning("Skipping reference artifact download (missing endpoint): %s", exc)
            return

        bucket = os.getenv("R2_BUCKET", "models")
        try:
            specs = load_artifact_specs(repo_root=REPO_ROOT)
        except Exception as exc:
            LOG.warning("Failed to load artifact manifest: %s", exc)
            return

        if not specs:
            LOG.info("No artifact specs configured; skipping reference fetch")
            return

        try:
            sync_artifacts(
                specs,
                direction="download",
                bucket=bucket,
                endpoint_url=endpoint,
                local_root=REPO_ROOT,
                skip_existing=True,
            )
        except Exception as exc:
            LOG.warning("Failed to prefetch reference artifacts: %s", exc)

    def _run_hf_once(
        self, workdir: Path, req: TrainRequest, cfg: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Path]:
        run_id = uuid.uuid4().hex[:8]
        outdir = _ensure_dir(workdir / f"hf_{run_id}")

        config = {
            "seed": cfg.get("seed", req.seed or 42),
            "training": {
                "epochs": req.epochs,
                "batch_size": cfg["batch_size"],
                "learning_rate": cfg["learning_rate"],
                "loss": cfg["loss"],
            },
            "data": {
                "symbols": req.symbols,
                "context_length": cfg["context_length"],
                "horizon": cfg["horizon"],
                "val_days": req.val_days,
                "trainingdata_dir": req.trainingdata_prefix,
                "use_toto_forecasts": cfg.get("crypto_enabled", False),
            },
            "costs": {"transaction_cost_bps": req.transaction_cost_bps},
            "output": {"dir": str(outdir)},
        }

        cfg_path = outdir / "config.json"
        with cfg_path.open("w") as handle:
            json.dump(config, handle, indent=2)

        os.environ.setdefault("WANDB_PROJECT", _env("WANDB_PROJECT", "stock-prediction"))
        os.environ.setdefault("WANDB_ENTITY", _env("WANDB_ENTITY", "default"))
        os.environ.setdefault("WANDB_RUN_GROUP", req.run_name)

        _run(
            ["python", "-m", "hftraining.run_training", "--config", str(cfg_path)],
        )

        metrics_path = outdir / "final_metrics.json"
        metrics: Dict[str, Any] = {}
        if metrics_path.exists():
            with metrics_path.open("r") as handle:
                try:
                    metrics = json.load(handle)
                except json.JSONDecodeError:
                    metrics = {}
        return metrics, outdir

    def _run_toto_once(
        self, workdir: Path, req: TrainRequest, cfg: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Path]:
        run_id = uuid.uuid4().hex[:8]
        outdir = _ensure_dir(workdir / f"toto_{run_id}")

        cmd = [
            "python",
            "tototraining/run_gpu_training.py",
            "--output-dir",
            str(outdir),
            "--epochs",
            str(req.epochs),
            "--batch-size",
            str(cfg["batch_size"]),
            "--context-length",
            str(cfg["context_length"]),
            "--prediction-length",
            str(cfg["horizon"]),
            "--loss",
            cfg["loss"],
        ]
        _run(cmd)

        metrics_path = outdir / "final_metrics.json"
        metrics: Dict[str, Any] = {}
        if metrics_path.exists():
            with metrics_path.open("r") as handle:
                try:
                    metrics = json.load(handle)
                except json.JSONDecodeError:
                    metrics = {}
        return metrics, outdir

    def _run_puffer_once(
        self, workdir: Path, req: TrainRequest, cfg: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Path]:
        run_id = uuid.uuid4().hex[:8]
        outdir = _ensure_dir(workdir / f"puffer_{run_id}")
        logdir = _ensure_dir(outdir / "logs")

        cmd = [
            "python",
            "pufferlibtraining/train_ppo.py",
            "--trainingdata-dir",
            req.trainingdata_prefix,
            "--output-dir",
            str(outdir),
            "--tensorboard-dir",
            str(logdir),
            "--rl-epochs",
            str(req.epochs),
            "--rl-batch-size",
            str(cfg["batch_size"]),
            "--rl-learning-rate",
            str(cfg["learning_rate"]),
        ]
        _run(cmd)

        metrics_path = outdir / "summary.json"
        metrics: Dict[str, Any] = {}
        if metrics_path.exists():
            with metrics_path.open("r") as handle:
                try:
                    metrics = json.load(handle)
                except json.JSONDecodeError:
                    metrics = {}
        return metrics, outdir

    def _evaluate_pnl(
        self, model_dir: Path, include_crypto: bool, trainingdata_dir: Path
    ) -> Dict[str, Any]:
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(REPO_ROOT))

        try:
            args = ["python", "-u", "comprehensive_backtest_real_gpu.py"]
            if include_crypto:
                args.append("--include-crypto")
            args.extend(["--model-dir", str(model_dir)])
            if trainingdata_dir.exists():
                args.extend(["--trainingdata-dir", str(trainingdata_dir)])
            _run(args, env=env, check=False)
        except Exception:
            pass

        try:
            args = ["python", "-u", "enhanced_local_backtester.py"]
            if include_crypto:
                args.append("--include-crypto")
            args.extend(["--model-dir", str(model_dir)])
            _run(args, env=env, check=False)
        except Exception:
            pass

        pnl = {
            "return_pct": None,
            "sharpe": None,
            "max_drawdown_pct": None,
            "mode": "hf_quick_realistic_test",
        }
        try:
            cp = _run(
                ["python", "-u", "hftraining/quick_realistic_test.py"],
                env=env,
                check=False,
                capture=True,
            )
            output = cp.stdout or ""
            import re

            ret_match = re.search(r"Return:\s*([-\d\.]+)%", output)
            sharpe_match = re.search(r"Sharpe Ratio:\s*([-\d\.]+)", output)
            dd_match = re.search(r"Max Drawdown:\s*([-\d\.]+)%", output)
            if ret_match:
                pnl["return_pct"] = float(ret_match.group(1))
            if sharpe_match:
                pnl["sharpe"] = float(sharpe_match.group(1))
            if dd_match:
                pnl["max_drawdown_pct"] = float(dd_match.group(1))
        except Exception:
            pass

        LOG.info("PnL summary (crypto=%s): %s", include_crypto, pnl)
        return pnl

    @fal.endpoint("/api/train")
    def train(self, request: TrainRequest) -> TrainResponse:
        r2_endpoint = _env("R2_ENDPOINT")
        bucket = _env("R2_BUCKET", "models")
        s3_uri = f"s3://{bucket}"

        local_data = Path("/data/trainingdata")
        remote_data = f"{s3_uri}/trainingdata/"
        LOG.info("Syncing training data from %s", remote_data)
        _aws_sync(remote_data, f"{local_data}/", r2_endpoint)

        repo_trainingdata = (REPO_ROOT / request.trainingdata_prefix).resolve()
        if not repo_trainingdata.exists():
            repo_trainingdata.parent.mkdir(parents=True, exist_ok=True)
            try:
                if repo_trainingdata.exists() or repo_trainingdata.is_symlink():
                    repo_trainingdata.unlink()
            except FileNotFoundError:
                pass
            try:
                os.symlink(local_data, repo_trainingdata)
                LOG.info("Symlinked training data -> %s", repo_trainingdata)
            except FileExistsError:
                pass

        seed = _maybe_seed(request.seed)
        base_cfg = {"seed": seed}
        sweep_cfgs = [base_cfg]
        if request.do_sweeps:
            sweep_cfgs = [{**base_cfg, **grid} for grid in _grid(request.sweeps)]
        else:
            sweep_cfgs = [{**base_cfg, **_grid(request.sweeps)[0]}]

        artifact_root = _ensure_dir(Path(request.output_root) / request.run_name)
        results: List[SweepResult] = []
        best_idx, best_score = -1, float("inf")

        for idx, cfg in enumerate(sweep_cfgs):
            LOG.info("=== Sweep %d/%d ===", idx + 1, len(sweep_cfgs))
            LOG.info("Config: %s", cfg)
            start_ts = time.time()

            if request.trainer == "hf":
                metrics, outdir = self._run_hf_once(artifact_root, request, cfg)
            elif request.trainer == "toto":
                metrics, outdir = self._run_toto_once(artifact_root, request, cfg)
            elif request.trainer == "puffer":
                metrics, outdir = self._run_puffer_once(artifact_root, request, cfg)
            else:
                raise ValueError(f"Unknown trainer: {request.trainer}")

            score = None
            for key in ("val_loss", "validation_loss", "loss"):
                if key in metrics:
                    score = metrics[key]
                    break
            if score is None:
                score = float("inf")

            LOG.info(
                "Run finished in %.1fs with score=%s (artifacts=%s)",
                time.time() - start_ts,
                score,
                outdir,
            )

            metrics_path = outdir / "final_metrics.json"
            metrics_file = str(metrics_path) if metrics_path.exists() else None
            results.append(
                SweepResult(
                    cfg=cfg,
                    metrics=metrics,
                    artifacts=RunArtifact(
                        local_dir=str(outdir),
                        metrics_path=metrics_file,
                    ),
                )
            )

            if score < best_score:
                best_score = score
                best_idx = len(results) - 1

        best = results[best_idx]
        pnl_summary = {
            "stock_only": self._evaluate_pnl(
                Path(best.artifacts.local_dir), include_crypto=False, trainingdata_dir=local_data
            ),
            "stock_plus_crypto": self._evaluate_pnl(
                Path(best.artifacts.local_dir), include_crypto=True, trainingdata_dir=local_data
            ),
        }

        LOG.info("Uploading artifacts for run %s", request.run_name)
        run_prefix = f"{s3_uri}/checkpoints/{request.run_name}/"
        _aws_cp(f"{artifact_root}/", run_prefix, r2_endpoint, recursive=True)

        logs_dir = artifact_root / "logs"
        if logs_dir.is_dir():
            _aws_cp(f"{logs_dir}/", f"{s3_uri}/logs/{request.run_name}/", r2_endpoint, recursive=True)

        return TrainResponse(
            run_name=request.run_name,
            best_cfg=best.cfg,
            best_metrics=best.metrics,
            sweep_results=results,
            pnl_summary=pnl_summary,
            artifact_root=RunArtifact(local_dir=str(artifact_root), r2_uri=run_prefix),
        )


def create_app() -> StockTrainerApp:
    if os.getenv("IS_ISOLATE_AGENT"):
        return StockTrainerApp()
    return StockTrainerApp(_allow_init=True)


app = create_app()
