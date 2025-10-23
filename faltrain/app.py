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
import math
import os
import random
import shutil
import subprocess
import time
import uuid
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import inspect

import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import fal
import transformers
from pydantic import BaseModel, Field

from faltrain.artifacts import load_artifact_specs, sync_artifacts
from faltrain.batch_size_tuner import (
    BatchSizeSelection,
    auto_tune_batch_sizes,
    get_cached_batch_selection,
    persist_batch_size,
)
from faltrain.dependencies import bulk_register_fal_dependencies
from faltrain.logger_utils import configure_stdout_logging
from wandboard import WandBoardLogger

from src.dependency_injection import setup_imports as setup_src_imports
from src.tblib_compat import ensure_tblib_pickling_support
from faltrain.shared_logger import get_logger, setup_logging
REPO_ROOT = Path(__file__).resolve().parents[1]
ensure_tblib_pickling_support()
LOG = get_logger("faltrain.app", logging.INFO)

DEFAULT_STOCK_TRADING_FEE = 0.0005
DEFAULT_CRYPTO_TRADING_FEE = 0.0015
DEFAULT_STOCK_TRADING_FEE_BPS = int(round(DEFAULT_STOCK_TRADING_FEE * 10_000))
DEFAULT_CRYPTO_TRADING_FEE_BPS = int(round(DEFAULT_CRYPTO_TRADING_FEE * 10_000))

DEFAULT_STOCK_TRADING_FEE = 0.0005
DEFAULT_CRYPTO_TRADING_FEE = 0.0015
DEFAULT_STOCK_TRADING_FEE_BPS = int(round(DEFAULT_STOCK_TRADING_FEE * 10_000))
DEFAULT_CRYPTO_TRADING_FEE_BPS = int(round(DEFAULT_CRYPTO_TRADING_FEE * 10_000))

_TRAINING_INJECTION_MODULES: Tuple[str, ...] = (
    "hftraining.injection",
    "fal_hftraining.runner",
    "tototraining.injection",
    "pufferlibtraining.injection",
    "fal_pufferlibtraining.runner",
    "fal_marketsimulator.runner",
    "tototrainingfal.runner",
    "faltrain.batch_size_tuner",
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


def _resolve_training_seed(request_seed: Optional[int]) -> int:
    """Return the seed for a training run and seed relevant libraries."""
    seed = int(request_seed) if request_seed is not None else random.randint(1, 10_000_000)
    random.seed(seed)
    transformers.set_seed(seed)
    return seed


def _inject_training_modules(
    torch_mod: Any,
    numpy_mod: Any,
    pandas_mod: Any | None = None,
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
            try:
                params = tuple(inspect.signature(setup_fn).parameters.values())
            except (TypeError, ValueError):
                params = ()
            try:
                if len(params) >= 3:
                    setup_fn(torch_mod, numpy_mod, pandas_mod)
                else:
                    setup_fn(torch_mod, numpy_mod)
            except TypeError:
                setup_fn(torch_mod, numpy_mod)
            LOG.info("Injected torch/numpy into %s", mod_path)


_OOM_PATTERNS: Tuple[str, ...] = (
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "RuntimeError: CUDA out of memory",
    "CUDA error: out of memory",
    "CUBLAS error: CUBLAS_STATUS_ALLOC_FAILED",
    "out of memory",
    "Killed",
)


class TrainingOOMError(RuntimeError):
    def __init__(self, cmd: List[str], output: str) -> None:
        super().__init__("CUDA OOM while executing: " + " ".join(cmd))
        self.cmd = cmd
        self.output = output


def _looks_like_oom(output: str) -> bool:
    text = output.lower()
    for pattern in _OOM_PATTERNS:
        if pattern.lower() in text:
            return True
    return False


def _plan_to_selection(plan: Dict[str, Any], *, selected: int) -> Optional[BatchSizeSelection]:
    def _coerce_sequence(values: Any) -> Tuple[int, ...]:
        seq: List[int] = []
        if isinstance(values, (list, tuple)):
            for value in values:
                try:
                    seq.append(int(value))
                except (TypeError, ValueError):
                    continue
        return tuple(seq)

    descending = _coerce_sequence(plan.get("candidates_desc"))
    if not descending:
        descending = _coerce_sequence(plan.get("candidates"))
    if not descending:
        return None
    user_candidates = _coerce_sequence(plan.get("candidates_user"))
    if not user_candidates:
        user_candidates = descending
    signature = plan.get("signature")
    context_length = int(plan.get("context_length", 1))
    horizon = int(plan.get("horizon", 1))
    exhaustive = bool(plan.get("exhaustive", False))
    return BatchSizeSelection(
        signature=signature,
        selected=int(selected),
        descending_candidates=tuple(dict.fromkeys(descending)),
        user_candidates=tuple(dict.fromkeys(user_candidates)),
        context_length=context_length,
        horizon=horizon,
        exhaustive=exhaustive,
    )


def _score_from_metrics(metrics: Dict[str, Any]) -> Optional[float]:
    for key in ("val_loss", "validation_loss", "loss"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _safe_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value.strip())
        except (TypeError, ValueError):
            return None
    else:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _lookup_path(payload: Any, path: Iterable[str]) -> Any:
    current = payload
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def _extract_loss_metric(metrics: Dict[str, Any]) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None

    preferred_paths = (
        ("val", "loss"),
        ("validation", "loss"),
        ("metrics", "val_loss"),
        ("metrics", "loss"),
        ("val_loss",),
        ("validation_loss",),
        ("best_val_loss",),
        ("loss",),
    )

    for path in preferred_paths:
        value = _lookup_path(metrics, path)
        number = _safe_number(value)
        if number is not None:
            return number

    # Fallback: breadth-first search for the first key containing "loss"
    queue: List[Any] = [metrics]
    while queue:
        current = queue.pop(0)
        if not isinstance(current, dict):
            continue
        for key, value in current.items():
            if isinstance(value, dict):
                queue.append(value)
            if isinstance(key, str) and "loss" in key.lower():
                number = _safe_number(value)
                if number is not None:
                    return number
    return None


def _extract_return_pct(pnl_summary: Dict[str, Any]) -> Optional[float]:
    if not isinstance(pnl_summary, dict):
        return None

    preferred_paths = (
        ("stock_only", "return_pct"),
        ("stock_plus_crypto", "return_pct"),
        ("return_pct",),
        ("test", "return_pct"),
        ("val", "return_pct"),
    )

    for path in preferred_paths:
        value = _lookup_path(pnl_summary, path)
        number = _safe_number(value)
        if number is not None:
            return number

    queue: List[Any] = [pnl_summary]
    while queue:
        current = queue.pop(0)
        if not isinstance(current, dict):
            continue
        for key, value in current.items():
            if isinstance(value, dict):
                queue.append(value)
            if isinstance(key, str) and "pnl" in key.lower():
                number = _safe_number(value)
                if number is not None:
                    return number
            if isinstance(key, str) and "return" in key.lower():
                number = _safe_number(value)
                if number is not None:
                    return number
    return None


def _sanitize_token(token: str) -> str:
    cleaned = token.strip().replace(os.sep, "-")
    cleaned = cleaned.replace(" ", "-")
    cleaned = cleaned.replace("__", "_")
    return cleaned or "model"


def _format_metric_token(prefix: str, value: Optional[float], precision: int) -> Optional[str]:
    if value is None:
        return None
    scaled = f"{value:.{precision}f}"
    scaled = scaled.rstrip("0").rstrip(".") or "0"
    token = f"{prefix}{scaled}"
    token = token.replace("+", "")
    return token


def _build_export_filename(
    *,
    source: Path,
    run_name: str,
    trainer: str,
    loss: Optional[float],
    return_pct: Optional[float],
    postfix: Optional[str] = None,
) -> str:
    tokens = [
        _sanitize_token(run_name),
        _sanitize_token(trainer),
        _sanitize_token(source.stem),
    ]
    if postfix:
        tokens.append(_sanitize_token(postfix))

    loss_token = _format_metric_token("loss", loss, 6)
    if loss_token:
        tokens.append(loss_token)

    pnl_token = _format_metric_token("pnl", return_pct, 2)
    if pnl_token:
        tokens.append(pnl_token)

    filename = "_".join(filter(None, tokens))
    return f"{filename}{source.suffix}"


def _collect_checkpoint_candidates(base_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        if not path.exists():
            return
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(path)

    priority_names = (
        "best_model.pt",
        "deployable.pt",
        "deployable_model.pt",
    )
    for name in priority_names:
        _add(base_dir / name)

    best_dir = base_dir / "best"
    if best_dir.is_dir():
        for path in sorted(best_dir.glob("rank*.pt")):
            _add(path)

    for path in sorted(base_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        _add(path)

    hf_export = base_dir / "hf_export"
    if hf_export.is_dir():
        for path in sorted(hf_export.glob("*.pt")):
            _add(path)

    return candidates


def _grid(space: "SweepSpace") -> List[Dict[str, Any]]:
    grid: List[Dict[str, Any]] = []
    batch_selection: Optional[BatchSizeSelection] = None
    if space.auto_tune_batch_size:
        batch_selection = get_cached_batch_selection(
            candidates=space.batch_sizes,
            context_lengths=space.context_lengths,
            horizons=space.horizons,
        )
        if batch_selection and batch_selection.signature:
            LOG.info(
                "Reusing cached batch size %s for signature %s",
                batch_selection.selected,
                batch_selection.signature,
            )
    if batch_selection is None:
        batch_selection = auto_tune_batch_sizes(
            candidates=space.batch_sizes,
            context_lengths=space.context_lengths,
            horizons=space.horizons,
            auto_tune=space.auto_tune_batch_size,
            safety_margin=space.batch_size_safety_margin,
        )
    batch_meta = batch_selection.meta()
    for lr in space.learning_rates:
        for bs in batch_selection.sweep_values():
            fallbacks = batch_selection.fallback_values(start=bs)
            for cl in space.context_lengths:
                for horizon in space.horizons:
                    for loss in space.loss:
                        for crypto_enabled in space.crypto_enabled:
                            cfg = {
                                "learning_rate": lr,
                                "batch_size": bs,
                                "context_length": cl,
                                "horizon": horizon,
                                "loss": loss,
                                "crypto_enabled": crypto_enabled,
                                "batch_size_plan": {
                                    **batch_meta,
                                    "initial": bs,
                                    "fallbacks": fallbacks,
                                },
                            }
                            grid.append(cfg)
    return grid


class SweepSpace(BaseModel):
    learning_rates: List[float] = [7e-4, 3e-4]
    batch_sizes: List[int] = [2048, 1024, 512, 256]
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
    transaction_cost_bps: Optional[int] = None
    stock_transaction_cost_bps: int = DEFAULT_STOCK_TRADING_FEE_BPS
    crypto_transaction_cost_bps: int = DEFAULT_CRYPTO_TRADING_FEE_BPS
    seed: Optional[int] = None
    parallel_trials: int = 4


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


def _resolve_transaction_cost_bps(req: TrainRequest, *, crypto_enabled: bool) -> int:
    if req.transaction_cost_bps is not None:
        return int(req.transaction_cost_bps)
    if crypto_enabled:
        return int(req.crypto_transaction_cost_bps)
    return int(req.stock_transaction_cost_bps)


class StockTrainerApp(
    fal.App,
    name="stock-trainer",
    min_concurrency=0,
    max_concurrency=1,
    keep_alive=30,
):
    machine_type = "GPU-H200"
    python_version = "3.12"
    requirements = [
        "fal-client",
        "tblib>=3.2",
        "nvidia-ml-py>=13.580.82",
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
        "fal_hftraining",
        "fal_pufferlibtraining",
    ]

    def setup(self) -> None:
        setup_logging(logging.INFO)
        configure_stdout_logging(level=logging.INFO, fmt="%(asctime)s | %(message)s")

        import torch as _torch
        import numpy as _np
        import pandas as _pd

        bulk_register_fal_dependencies(
            {
                "torch": _torch,
                "numpy": _np,
                "pandas": _pd,
            }
        )
        setup_src_imports(_torch, _np, _pd)

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
        _inject_training_modules(_torch, _np, _pd)

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

    def _invoke_training_process(
        self,
        cmd: List[str],
        *,
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        tail = deque(maxlen=200)
        oom_hint = False
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to start training command {cmd}: {exc}") from exc

        assert proc.stdout is not None
        with proc.stdout:
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                LOG.info(line)
                tail.append(line)
                if not oom_hint and _looks_like_oom(line):
                    oom_hint = True
        return_code = proc.wait()
        summary = "\n".join(tail)
        if not oom_hint and summary:
            oom_hint = _looks_like_oom(summary)
        if return_code != 0:
            if oom_hint:
                raise TrainingOOMError(cmd, summary)
            raise RuntimeError(
                f"Training command failed (exit={return_code}): {' '.join(cmd)}\n{summary}"
            )
        return summary

    def _run_with_batch_retry(
        self,
        runner: Callable[[Dict[str, Any]], Tuple[Dict[str, Any], Path]],
        cfg: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Path]:
        plan = cfg.get("batch_size_plan")
        fallback_values: List[int] = []
        if isinstance(plan, dict):
            for value in plan.get("fallbacks", []):
                try:
                    fallback_values.append(int(value))
                except (TypeError, ValueError):
                    continue
        if not fallback_values:
            fallback_values = [int(cfg["batch_size"])]

        last_exc: Optional[TrainingOOMError] = None
        for batch_size in fallback_values:
            cfg["batch_size"] = int(batch_size)
            LOG.info("Launching training with batch_size=%s", batch_size)
            try:
                metrics, outdir = runner(cfg)
            except TrainingOOMError as exc:
                LOG.warning(
                    "CUDA OOM detected for batch_size=%s; falling back to smaller candidate",
                    batch_size,
                )
                if isinstance(plan, dict):
                    plan.setdefault("oom_attempts", []).append(batch_size)
                last_exc = exc
                continue

            if isinstance(plan, dict):
                plan["used"] = batch_size
                selection = _plan_to_selection(plan, selected=batch_size)
                if selection is not None:
                    try:
                        persist_batch_size(selection, batch_size=batch_size)
                    except Exception as exc:
                        LOG.warning(
                            "Failed to persist successful batch size %s: %s", batch_size, exc
                        )
            return metrics, outdir

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Batch size retry logic exhausted without attempting training")

    def _runner_for_request(
        self, request: TrainRequest, artifact_root: Path
    ) -> Callable[[Dict[str, Any]], Tuple[Dict[str, Any], Path]]:
        if request.trainer == "hf":
            return lambda cfg: self._run_hf_once(artifact_root, request, cfg)
        if request.trainer == "toto":
            return lambda cfg: self._run_toto_once(artifact_root, request, cfg)
        if request.trainer == "puffer":
            return lambda cfg: self._run_puffer_once(artifact_root, request, cfg)
        raise ValueError(f"Unknown trainer: {request.trainer}")

    def _run_single_config(
        self,
        idx: int,
        total: int,
        cfg: Dict[str, Any],
        runner: Callable[[Dict[str, Any]], Tuple[Dict[str, Any], Path]],
    ) -> Tuple[int, Dict[str, Any], Dict[str, Any], Path, float]:
        cfg_local = copy.deepcopy(cfg)
        LOG.info("=== Sweep %d/%d ===", idx + 1, total)
        LOG.info("Config: %s", cfg_local)
        start_ts = time.time()
        metrics, outdir = self._run_with_batch_retry(runner, cfg_local)
        duration = time.time() - start_ts
        return idx, cfg_local, metrics, outdir, duration

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
                "transaction_cost_bps": int(
                    cfg.get(
                        "transaction_cost_bps",
                        _resolve_transaction_cost_bps(
                            req, crypto_enabled=cfg.get("crypto_enabled", False)
                        ),
                    )
                ),
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

        from fal_hftraining.runner import run_training as run_hf_training

        metrics, _ = run_hf_training(config=config, run_name=req.run_name, output_dir=outdir)
        return metrics, outdir

    def _run_toto_once(
        self, workdir: Path, req: TrainRequest, cfg: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Path]:
        run_id = uuid.uuid4().hex[:8]
        outdir = _ensure_dir(workdir / f"toto_{run_id}")
        from tototrainingfal.runner import run_training as run_toto_training

        trainingdata_root = Path(req.trainingdata_prefix)
        train_root = trainingdata_root / "train"
        val_root = trainingdata_root / "val"
        if not val_root.exists():
            alt = trainingdata_root / "test"
            val_root = alt if alt.exists() else None

        device_label = "cuda"
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                device_label = "cpu"
        except Exception:
            device_label = "cpu"

        metrics, _ = run_toto_training(
            train_root=train_root,
            val_root=val_root,
            context_length=int(cfg["context_length"]),
            prediction_length=int(cfg["horizon"]),
            stride=int(max(1, cfg["horizon"])),
            batch_size=int(cfg["batch_size"]),
            epochs=int(req.epochs),
            learning_rate=float(cfg.get("learning_rate", req.sweeps.learning_rates[0])),
            loss=str(cfg["loss"]),
            output_dir=outdir,
            device=device_label,
            grad_accum=max(1, cfg.get("grad_accum", 1)),
        )

        return metrics, outdir

    def _run_puffer_once(
        self, workdir: Path, req: TrainRequest, cfg: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Path]:
        run_id = uuid.uuid4().hex[:8]
        outdir = _ensure_dir(workdir / f"puffer_{run_id}")
        logdir = _ensure_dir(outdir / "logs")

        from fal_pufferlibtraining.runner import run_training as run_puffer_training

        summary, summary_path = run_puffer_training(
            trainingdata_dir=Path(req.trainingdata_prefix),
            output_dir=outdir,
            tensorboard_dir=logdir,
            cfg=cfg,
            epochs=req.epochs,
            transaction_cost_bps=float(cfg.get("transaction_cost_bps", req.transaction_cost_bps or 0)),
            run_name=req.run_name,
        )

        metrics: Dict[str, Any] = {}
        if summary_path.exists():
            try:
                metrics = json.loads(summary_path.read_text()).get("portfolio_pairs", {})
            except json.JSONDecodeError:
                metrics = {}
        if not metrics:
            metrics = summary.get("portfolio_pairs", {})
        return metrics, outdir

    def _evaluate_pnl(
        self, model_dir: Path, include_crypto: bool, trainingdata_dir: Path
    ) -> Dict[str, Any]:
        pnl = {
            "return_pct": None,
            "sharpe": None,
            "max_drawdown_pct": None,
            "mode": "hf_quick_realistic_test",
        }
        try:
            from hftraining.quick_realistic_test import quick_test

            _, metrics, _ = quick_test()
            if metrics is not None:
                pnl["return_pct"] = float(getattr(metrics, "total_return", 0.0) * 100.0)
                pnl["sharpe"] = float(getattr(metrics, "sharpe_ratio", 0.0))
                pnl["max_drawdown_pct"] = float(getattr(metrics, "max_drawdown", 0.0) * 100.0)
        except Exception:
            pass

        try:
            from comprehensive_backtest_real_gpu import ComprehensiveBacktester
            from src.fixtures import crypto_symbols

            symbols = [
                "COUR",
                "GOOG",
                "TSLA",
                "NVDA",
                "AAPL",
                "U",
                "ADSK",
                "ADBE",
                "COIN",
                "MSFT",
                "NFLX",
            ]
            if include_crypto:
                symbols.extend(symbol for symbol in crypto_symbols[:3])
            backtester = ComprehensiveBacktester(symbols=symbols)
            backtester.run_comprehensive_backtest()
        except Exception:
            pass

        try:
            from enhanced_local_backtester import run_enhanced_comparison
            from src.fixtures import crypto_symbols

            symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
            if include_crypto:
                symbols.extend(symbol for symbol in crypto_symbols[:2])
            run_enhanced_comparison(symbols, simulation_days=14, compare_with_synthetic=False)
        except Exception:
            pass

        LOG.info("PnL summary (crypto=%s): %s", include_crypto, pnl)
        return pnl

    def _export_best_model(
        self,
        *,
        best: SweepResult,
        pnl_summary: Dict[str, Any],
        trainer: str,
        run_name: str,
        s3_uri: str,
        endpoint: str,
    ) -> Optional[Path]:
        best_dir = Path(best.artifacts.local_dir)
        if not best_dir.exists():
            LOG.info("Best artifact directory %s missing; skipping model export", best_dir)
            return None

        candidates = _collect_checkpoint_candidates(best_dir)
        if not candidates:
            LOG.info("No checkpoint artifacts under %s; skipping model export", best_dir)
            return None

        loss_metric = _extract_loss_metric(best.metrics)
        pnl_metric = _extract_return_pct(pnl_summary)

        models_dir = _ensure_dir(Path(os.getenv("FALTRAIN_MODELS_DIR", "/data/models")))
        source = candidates[0]
        filename = _build_export_filename(
            source=source,
            run_name=run_name,
            trainer=trainer,
            loss=loss_metric,
            return_pct=pnl_metric,
        )
        destination = models_dir / filename

        LOG.info("Exporting compiled model %s -> %s", source, destination)
        shutil.copy2(source, destination)

        remote_uri = f"{s3_uri}/models/{filename}"
        LOG.info("Uploading compiled model to %s", remote_uri)
        _aws_cp(str(destination), remote_uri, endpoint)

        return destination

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

        seed = _resolve_training_seed(request.seed)
        base_cfg = {"seed": seed}
        sweep_cfgs = [base_cfg]
        if request.do_sweeps:
            sweep_cfgs = [{**base_cfg, **grid} for grid in _grid(request.sweeps)]
        else:
            sweep_cfgs = [{**base_cfg, **_grid(request.sweeps)[0]}]

        for cfg in sweep_cfgs:
            crypto_flag = bool(cfg.get("crypto_enabled", False))
            cfg.setdefault("crypto_enabled", crypto_flag)
            cfg["transaction_cost_bps"] = _resolve_transaction_cost_bps(
                request, crypto_enabled=crypto_flag
            )

        artifact_root = _ensure_dir(Path(request.output_root) / request.run_name)
        runner = self._runner_for_request(request, artifact_root)
        total_runs = len(sweep_cfgs)
        parallel_trials = max(1, min(int(request.parallel_trials or 1), total_runs))

        enable_sweep_logging = request.do_sweeps and total_runs > 1
        sweep_logger_ctx = nullcontext(None)
        if enable_sweep_logging:
            aggregate_log_dir = _ensure_dir(artifact_root / "sweep_logs")
            try:
                sweep_logger_ctx = WandBoardLogger(
                    run_name=f"{request.run_name}_sweep",
                    project=os.getenv("WANDB_PROJECT"),
                    entity=os.getenv("WANDB_ENTITY"),
                    group=request.run_name,
                    tags=(f"trainer:{request.trainer}", "faltrain", "sweep"),
                    log_dir=aggregate_log_dir,
                    tensorboard_subdir="aggregate",
                    enable_wandb=True,
                )
            except Exception as exc:
                LOG.warning("Failed to initialise sweep logger: %s", exc)
                sweep_logger_ctx = nullcontext(None)
                enable_sweep_logging = False

        results_by_idx: Dict[int, Tuple[Dict[str, Any], Dict[str, Any], Path, float, float]] = {}
        pnl_per_idx: Dict[int, Dict[str, Dict[str, Any]]] = {}
        results: List[SweepResult] = []
        best_idx, best_score = -1, float("inf")

        with sweep_logger_ctx as sweep_logger:

            def _record_result(
                idx_res: int,
                cfg_res: Dict[str, Any],
                metrics_res: Dict[str, Any],
                outdir_res: Path,
                duration_res: float,
            ) -> None:
                nonlocal best_idx, best_score
                score_val = _score_from_metrics(metrics_res)
                score = score_val if score_val is not None else float("inf")
                score_display = score_val if score_val is not None else "n/a"
                LOG.info(
                    "Run finished in %.1fs with score=%s (artifacts=%s)",
                    duration_res,
                    score_display,
                    outdir_res,
                )
                results_by_idx[idx_res] = (cfg_res, metrics_res, outdir_res, score, duration_res)
                if best_idx == -1 or score < best_score:
                    best_score = score
                    best_idx = idx_res

            if parallel_trials == 1:
                for idx, cfg in enumerate(sweep_cfgs):
                    idx_res, cfg_res, metrics_res, outdir_res, duration_res = self._run_single_config(
                        idx, total_runs, cfg, runner
                    )
                    _record_result(idx_res, cfg_res, metrics_res, outdir_res, duration_res)
            else:
                with ThreadPoolExecutor(max_workers=parallel_trials) as executor:
                    futures = [
                        executor.submit(self._run_single_config, idx, total_runs, cfg, runner)
                        for idx, cfg in enumerate(sweep_cfgs)
                    ]
                    try:
                        for future in as_completed(futures):
                            idx_res, cfg_res, metrics_res, outdir_res, duration_res = future.result()
                            _record_result(idx_res, cfg_res, metrics_res, outdir_res, duration_res)
                    except Exception:
                        for future in futures:
                            future.cancel()
                        raise

            if best_idx == -1:
                raise RuntimeError("No training runs completed successfully")

            for idx in sorted(results_by_idx):
                cfg_res, metrics_res, outdir_res, score_res, _ = results_by_idx[idx]
                metrics_file = None
                outdir_path = Path(outdir_res)
                for candidate in ("final_metrics.json", "summary.json"):
                    candidate_path = outdir_path / candidate
                    if candidate_path.exists():
                        metrics_file = str(candidate_path)
                        break
                results.append(
                    SweepResult(
                        cfg=cfg_res,
                        metrics=metrics_res,
                        artifacts=RunArtifact(
                            local_dir=str(outdir_res),
                            metrics_path=metrics_file,
                        ),
                    )
                )

                should_eval_pnl = enable_sweep_logging or idx == best_idx
                if should_eval_pnl:
                    pnl_per_idx[idx] = {
                        "stock_only": self._evaluate_pnl(
                            Path(outdir_res), include_crypto=False, trainingdata_dir=local_data
                        ),
                        "stock_plus_crypto": self._evaluate_pnl(
                            Path(outdir_res), include_crypto=True, trainingdata_dir=local_data
                        ),
                    }

            if sweep_logger:
                table_name = f"{request.trainer}_sweep"
                for idx in sorted(results_by_idx):
                    cfg_res, metrics_res, outdir_res, score_res, duration_res = results_by_idx[idx]
                    combined_metrics: Dict[str, Any] = {"duration_seconds": duration_res}
                    if metrics_res:
                        combined_metrics["training"] = metrics_res
                    if math.isfinite(score_res):
                        combined_metrics["score"] = score_res
                    if idx in pnl_per_idx:
                        combined_metrics["pnl"] = pnl_per_idx[idx]
                    combined_metrics["artifacts_dir"] = str(outdir_res)
                    sweep_logger.log_sweep_point(
                        hparams=cfg_res,
                        metrics=combined_metrics,
                        step=idx,
                        table_name=table_name,
                    )

        if best_idx == -1:
            raise RuntimeError("No training runs completed successfully")

        best = results[best_idx]
        if best_idx in pnl_per_idx:
            pnl_summary = pnl_per_idx[best_idx]
        else:
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

        exported_model = self._export_best_model(
            best=best,
            pnl_summary=pnl_summary,
            trainer=request.trainer,
            run_name=request.run_name,
            s3_uri=s3_uri,
            endpoint=r2_endpoint,
        )
        if exported_model is not None:
            LOG.info("Exported compiled model available at %s", exported_model)

        return TrainResponse(
            run_name=request.run_name,
            best_cfg=best.cfg,
            best_metrics=best.metrics,
            sweep_results=results,
            pnl_summary=pnl_summary,
            artifact_root=RunArtifact(local_dir=str(artifact_root), r2_uri=run_prefix),
        )


# Ensure the fal runtime sees concrete annotations when introspecting endpoints.
StockTrainerApp.train.__annotations__ = {
    **StockTrainerApp.train.__annotations__,
    "request": TrainRequest,
    "return": TrainResponse,
}


def create_app() -> StockTrainerApp:
    if os.getenv("IS_ISOLATE_AGENT"):
        return StockTrainerApp()
    return StockTrainerApp(_allow_init=True)


app = create_app()
