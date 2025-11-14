from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from fal_marketsimulator.runner import setup_training_imports, simulate_trading
from falmarket.shared_logger import get_logger, log_timing, setup_logging
from faltrain.artifacts import load_artifact_specs, sync_artifacts
from src.tblib_compat import ensure_tblib_pickling_support
from src.torch_backend import configure_tf32_backends

REPO_ROOT = Path(__file__).resolve().parents[1]

ensure_tblib_pickling_support()
setup_logging(logging.INFO)
LOG = get_logger("runpodmarket.handler", logging.INFO)

def _warmup_cuda(torch_module: Any) -> None:
    """Prime CUDA context to reduce first-request latency."""

    cuda_iface = getattr(torch_module, "cuda", None)
    if cuda_iface is None:
        return
    try:
        if not cuda_iface.is_available():
            LOG.info("CUDA not available; skipping warmup.")
            return
    except Exception as exc:  # pragma: no cover - CUDA presence query failed
        LOG.debug("Unable to query CUDA availability: %s", exc)
        return

    try:
        device_index = cuda_iface.current_device()
    except Exception:
        device_index = 0
        try:
            cuda_iface.set_device(device_index)
        except Exception:
            pass

    try:
        device_name = cuda_iface.get_device_name(device_index)
    except Exception:
        device_name = f"cuda:{device_index}"

    try:
        tensor = torch_module.zeros(1, device=f"cuda:{device_index}")
        tensor.mul_(1.0)
        cuda_iface.synchronize()
        capability = None
        get_capability = getattr(cuda_iface, "get_device_capability", None)
        if callable(get_capability):
            try:
                capability = get_capability(device_index)
            except Exception:
                capability = None
        LOG.info("CUDA warmup complete device=%s capability=%s", device_name, capability)
    except Exception as exc:  # pragma: no cover - device-specific failure
        LOG.warning("CUDA warmup failed on %s: %s", device_name, exc)


_np: ModuleType | None = None
_pd: ModuleType | None = None
_torch: ModuleType | None = None

try:  # Heavy numerics injected for the simulator runtime.
    import numpy as _np
    import pandas as _pd
    import torch as _torch
except Exception as exc:  # pragma: no cover - runtime environment check
    LOG.warning("Failed to import torch/numpy/pandas during startup: %s", exc)
    _np = None
    _pd = None
    _torch = None
else:
    configure_tf32_backends(_torch, logger=LOG)
    setup_training_imports(_torch, _np, _pd)
    try:
        _warmup_cuda(_torch)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("CUDA warmup attempt raised %s", exc)

STRICT_DOWNLOAD = os.getenv("RUNPODMARKET_STRICT_DOWNLOAD", "").lower() in {"1", "true", "yes"}

_MODELS_READY = False
_TRAINING_READY = False


class RunpodSimulationRequest(BaseModel):
    """Validate RunPod job payloads for the market simulator."""

    symbols: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])
    steps: int = Field(default=32, ge=1, le=5_000)
    step_size: int = Field(default=1, ge=1, le=240)
    initial_cash: float = Field(default=100_000.0, ge=1_000.0, le=10_000_000.0)
    top_k: int = Field(default=4, ge=1, le=25)
    kronos_only: bool = False
    compact_logs: bool = True
    download_models: bool = True
    download_training: bool = True
    force_download: bool = False

    @field_validator("symbols")
    @classmethod
    def _normalise_symbols(cls, value: Iterable[str]) -> list[str]:
        cleaned: list[str] = []
        seen = set()
        for symbol in value:
            normalised = symbol.strip().upper()
            if not normalised or normalised in seen:
                continue
            cleaned.append(normalised)
            seen.add(normalised)
        if not cleaned:
            raise ValueError("At least one symbol must be supplied.")
        return cleaned


def _strict_failure(message: str, exc: Optional[BaseException] = None) -> None:
    if not STRICT_DOWNLOAD:
        if exc is None:
            LOG.warning("%s", message)
        else:
            LOG.warning("%s: %s", message, exc)
        return
    if exc is None:
        raise RuntimeError(message)
    raise RuntimeError(message) from exc


def _download_model_artifacts(force: bool) -> None:
    """Download model artifacts declared in faltrain/model_manifest.toml."""

    global _MODELS_READY
    if _MODELS_READY and not force:
        return

    endpoint = os.getenv("R2_ENDPOINT")
    bucket = os.getenv("R2_BUCKET")
    if not endpoint or not bucket:
        _strict_failure("Skipping model artifact download (missing R2_ENDPOINT/R2_BUCKET)")
        _MODELS_READY = True
        return

    try:
        specs = load_artifact_specs(repo_root=REPO_ROOT)
    except Exception as exc:  # pragma: no cover - depends on manifest state
        _strict_failure("Failed to load artifact manifest", exc)
        return

    if not specs:
        LOG.info("Artifact manifest contained no entries; nothing to download.")
        _MODELS_READY = True
        return

    try:
        sync_artifacts(
            specs,
            direction="download",
            bucket=bucket,
            endpoint_url=endpoint,
            local_root=REPO_ROOT,
            skip_existing=not force,
        )
    except Exception as exc:  # pragma: no cover - subprocess/aws failure
        _strict_failure("Model artifact download failed", exc)
        return

    _MODELS_READY = True


def _download_training_data(force: bool) -> None:
    """Download training datasets stored in object storage."""

    global _TRAINING_READY
    if _TRAINING_READY and not force:
        return

    endpoint = os.getenv("TRAININGDATA_ENDPOINT") or os.getenv("R2_ENDPOINT")
    bucket = os.getenv("TRAININGDATA_BUCKET") or os.getenv("R2_BUCKET")
    prefix = os.getenv("TRAININGDATA_PREFIX", "stock/trainingdata")
    local_dir = os.getenv("TRAININGDATA_LOCAL_DIR", "trainingdata")
    if not endpoint or not bucket:
        _strict_failure("Skipping training data download (missing TRAININGDATA or R2 endpoint/bucket)")
        _TRAINING_READY = True
        return

    target = (REPO_ROOT / local_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    try:
        _sync_s3_prefix(
            bucket=bucket,
            prefix=prefix,
            endpoint=endpoint,
            local_root=target,
            force=force,
        )
    except Exception as exc:  # pragma: no cover - network/aws failure
        _strict_failure("Training data download failed", exc)
        return

    _TRAINING_READY = True


def _sync_s3_prefix(
    *,
    bucket: str,
    prefix: str,
    endpoint: str,
    local_root: Path,
    force: bool,
) -> None:
    """Mirror an S3 prefix into ``local_root`` using boto3."""

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
    except Exception as exc:  # pragma: no cover - boto missing
        _strict_failure("boto3 is required to download training data", exc)
        return

    session = boto3.session.Session()
    client = session.client("s3", endpoint_url=endpoint)
    clean_prefix = prefix.strip("/")
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=f"{clean_prefix}/")

    downloaded = 0
    skipped = 0
    for page in pages:
        contents = page.get("Contents", [])
        for entry in contents:
            key = entry.get("Key")
            if not key or key.endswith("/"):
                continue
            relative = key[len(clean_prefix) :].lstrip("/")
            destination = local_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not force:
                size = entry.get("Size")
                try:
                    if size is None or destination.stat().st_size == int(size):
                        skipped += 1
                        continue
                except FileNotFoundError:
                    pass
            try:
                client.download_file(bucket, key, str(destination))
                downloaded += 1
            except (BotoCoreError, ClientError) as exc:  # pragma: no cover - network failure
                _strict_failure(f"Failed to download s3://{bucket}/{key}", exc)
    LOG.info(
        "Training data sync complete bucket=%s prefix=%s downloaded=%s skipped=%s target=%s",
        bucket,
        clean_prefix,
        downloaded,
        skipped,
        local_root,
    )


def handle_job(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the market simulator for a RunPod job payload."""

    LOG.info("Received RunPod job input keys: %s", sorted(job_input.keys()))
    try:
        request = RunpodSimulationRequest.model_validate(job_input)
    except ValidationError as exc:
        LOG.warning("Validation failed for RunPod job payload: %s", exc)
        return {
            "status": "error",
            "error": exc.errors(),
        }

    if request.download_models:
        _download_model_artifacts(force=request.force_download)
    if request.download_training:
        _download_training_data(force=request.force_download)

    if _torch is not None and _np is not None:
        setup_training_imports(_torch, _np, _pd)

    started = datetime.now(timezone.utc)

    with log_timing(LOG, "simulate_trading via RunPod"):
        results = simulate_trading(
            symbols=request.symbols,
            steps=request.steps,
            step_size=request.step_size,
            initial_cash=request.initial_cash,
            top_k=request.top_k,
            kronos_only=request.kronos_only,
            compact_logs=request.compact_logs,
        )

    completed = datetime.now(timezone.utc)
    run_name = f"runpod-sim-{started.strftime('%Y%m%d_%H%M%S')}"

    LOG.info(
        "Simulation finished name=%s run_seconds=%.3f summary_keys=%s",
        run_name,
        results.get("run_seconds", 0.0),
        sorted(results.get("summary", {}).keys()),
    )

    return {
        "status": "success",
        "run": {
            "name": run_name,
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "parameters": request.model_dump(),
            "timeline": results.get("timeline", []),
            "summary": results.get("summary", {}),
            "run_seconds": float(results.get("run_seconds", 0.0)),
        },
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod entrypoint compatible with ``runpod.serverless.start``."""

    try:
        payload = job.get("input", {}) if isinstance(job, dict) else {}
        return handle_job(payload)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.exception("Unhandled error during RunPod job execution: %s", exc)
        return {
            "status": "error",
            "type": exc.__class__.__name__,
            "error": str(exc),
        }


def _start_serverless() -> None:
    auto_start = os.getenv("RUNPODMARKET_DISABLE_SERVERLESS", "").lower() not in {"1", "true", "yes"}
    if not auto_start:
        LOG.info("RUNPOD serverless auto-start disabled via environment flag.")
        return
    try:
        import runpod
    except ImportError as exc:  # pragma: no cover - dependency missing
        _strict_failure("runpod package missing; install it via `uv pip install runpod`.", exc)
        return
    runpod.serverless.start({"handler": handler})


_start_serverless()
