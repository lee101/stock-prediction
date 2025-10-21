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
from pathlib import Path
from importlib import import_module
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import fal
from pydantic import BaseModel, Field

from faltrain.artifacts import load_artifact_specs, sync_artifacts
from faltrain.batch_size_tuner import (
    BatchSizeSelection,
    auto_tune_batch_sizes,
    persist_batch_size,
)
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
    transaction_cost_bps: int = 1
    seed: Optional[int] = None
    parallel_trials: int = 2


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

        self._invoke_training_process(
            ["python", "-m", "hftraining.run_training", "--config", str(cfg_path)]
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
        self._invoke_training_process(cmd)

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
        self._invoke_training_process(cmd)

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

        seed = _maybe_seed(request.seed)
        base_cfg = {"seed": seed}
        sweep_cfgs = [base_cfg]
        if request.do_sweeps:
            sweep_cfgs = [{**base_cfg, **grid} for grid in _grid(request.sweeps)]
        else:
            sweep_cfgs = [{**base_cfg, **_grid(request.sweeps)[0]}]

        artifact_root = _ensure_dir(Path(request.output_root) / request.run_name)
        runner = self._runner_for_request(request, artifact_root)
        total_runs = len(sweep_cfgs)
        parallel_trials = max(1, min(int(request.parallel_trials or 1), total_runs))

        results_by_idx: Dict[int, Tuple[Dict[str, Any], Dict[str, Any], Path, float]] = {}
        best_idx, best_score = -1, float("inf")

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
            results_by_idx[idx_res] = (cfg_res, metrics_res, outdir_res, score)
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

        results: List[SweepResult] = []
        for idx in sorted(results_by_idx):
            cfg_res, metrics_res, outdir_res, _ = results_by_idx[idx]
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


def create_app() -> StockTrainerApp:
    if os.getenv("IS_ISOLATE_AGENT"):
        return StockTrainerApp()
    return StockTrainerApp(_allow_init=True)


app = create_app()
