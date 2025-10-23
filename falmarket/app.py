#!/usr/bin/env python3
"""FAL application exposing the marketsimulator trade loop."""

from __future__ import annotations

import logging
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from datetime import datetime, timezone

import fal
from pydantic import BaseModel, Field, field_validator

from fal_marketsimulator.runner import simulate_trading, setup_training_imports
from falmarket.shared_logger import get_logger, log_timing, setup_logging
from faltrain.artifacts import load_artifact_specs, sync_artifacts
from faltrain.dependencies import bulk_register_fal_dependencies
from faltrain.logger_utils import configure_stdout_logging
from src.dependency_injection import setup_imports as setup_src_imports
from src.tblib_compat import ensure_tblib_pickling_support
from src.torch_backend import configure_tf32_backends

REPO_ROOT = Path(__file__).resolve().parents[1]
ensure_tblib_pickling_support()
LOG = get_logger("falmarket.app", logging.INFO)


class SimulationRequest(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])
    steps: int = Field(default=32, ge=1, le=5000)
    step_size: int = Field(default=1, ge=1, le=240)
    initial_cash: float = Field(default=100_000.0, ge=1_000.0, le=10_000_000.0)
    top_k: int = Field(default=4, ge=1, le=25)
    kronos_only: bool = False
    compact_logs: bool = True

    @field_validator("symbols")
    @classmethod
    def _require_symbols(cls, value: List[str]) -> List[str]:
        symbols = [sym.strip().upper() for sym in value if sym.strip()]
        if not symbols:
            raise ValueError("At least one symbol must be provided.")
        return symbols


class SimulationResponse(BaseModel):
    run_name: str
    started_at: datetime
    completed_at: datetime
    timeline: List[Dict[str, Any]]
    summary: Dict[str, Any]
    run_seconds: float


class MarketSimulatorApp(
    fal.App,
    name="market-simulator",
    min_concurrency=0,
    max_concurrency=1,
    keep_alive=5,
):
    machine_type = "GPU-H200"
    python_version = "3.12"
    requirements = [
        "fal-client",
        "tblib>=3.2",
        "nvidia-ml-py>=13.580.82",
        "numpy>=1.24.4,<2",
        "pandas",
        "torch>=2.8.0",
        "scipy",
        "tqdm",
        "loguru",
        "pyarrow",
        "matplotlib",
        "seaborn",
    ]
    local_python_modules = [
        "falmarket",
        "fal_marketsimulator",
        "faltrain",
        "marketsimulator",
        "trade_stock_e2e",
        "trade_stock_e2e_trained",
        "src",
        "stock",
        "utils",
        "traininglib",
    ]

    def setup(self) -> None:
        with log_timing(LOG, "MarketSimulatorApp.setup"):
            setup_logging(logging.INFO)
            configure_stdout_logging(level=logging.INFO, fmt="%(asctime)s | %(message)s")
            warnings.filterwarnings(
                "ignore",
                message="The pynvml package is deprecated.*",
                category=FutureWarning,
                module="torch.cuda",
            )
            LOG.info(
                "Starting MarketSimulatorApp.setup pid=%s cwd=%s",
                os.getpid(),
                os.getcwd(),
            )
            os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
            os.environ.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:1024,expandable_segments:True"
            )
            LOG.debug(
                "Environment CUDA_LAUNCH_BLOCKING=%s PYTORCH_CUDA_ALLOC_CONF=%s",
                os.getenv("CUDA_LAUNCH_BLOCKING"),
                os.getenv("PYTORCH_CUDA_ALLOC_CONF"),
            )

            with log_timing(LOG, "Import torch/numpy/pandas"):
                import torch as _torch
                import numpy as _np
                import pandas as _pd

            with log_timing(LOG, "Configure torch backends"):
                tf32_state = configure_tf32_backends(_torch, logger=LOG)
                LOG.info(
                    "TF32 precision configured new_api=%s legacy_api=%s",
                    tf32_state["new_api"],
                    tf32_state["legacy_api"],
                )
                try:
                    _torch.backends.cudnn.benchmark = True
                    _torch.backends.cuda.enable_flash_sdp(True)
                    _torch.backends.cuda.enable_math_sdp(True)
                    _torch.backends.cuda.enable_mem_efficient_sdp(True)
                except Exception:
                    LOG.debug("Skipping advanced CUDA backend configuration", exc_info=True)

            with log_timing(LOG, "Register shared dependencies"):
                bulk_register_fal_dependencies({"torch": _torch, "numpy": _np, "pandas": _pd})
                setup_training_imports(_torch, _np, _pd)
                setup_src_imports(_torch, _np, _pd)

            os.environ.setdefault("MARKETSIM_ALLOW_MOCK_ANALYTICS", "1")
            os.environ.setdefault("MARKETSIM_SKIP_REAL_IMPORT", "1")
            os.environ.setdefault("FAL_WORKER", "1")
            os.environ.setdefault("COMPILED_MODELS_DIR", str((REPO_ROOT / "compiled_models").resolve()))
            (REPO_ROOT / "compiled_models").mkdir(parents=True, exist_ok=True)

            with log_timing(LOG, "Prefetch reference artifacts"):
                self._prefetch_reference_artifacts()
            with log_timing(LOG, "Sync hyperparameters"):
                self._sync_hyperparams(direction="download")
            with log_timing(LOG, "Sync compiled models"):
                self._sync_compiled_models(direction="download")

            LOG.info(
                "MarketSimulatorApp setup complete cuda_available=%s device='%s' torch=%s",
                _torch.cuda.is_available(),
                _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "cpu",
                _torch.__version__,
            )

    @fal.endpoint("/api/simulate")
    def simulate(self, request: SimulationRequest) -> SimulationResponse:
        started = datetime.now(timezone.utc)
        LOG.info(
            "Simulation request symbols=%s steps=%s step_size=%s top_k=%s kronos_only=%s compact_logs=%s",
            request.symbols,
            request.steps,
            request.step_size,
            request.top_k,
            request.kronos_only,
            request.compact_logs,
        )

        with log_timing(LOG, "simulate_trading execution"):
            results = simulate_trading(
                symbols=request.symbols,
                steps=request.steps,
                step_size=request.step_size,
                initial_cash=request.initial_cash,
                top_k=request.top_k,
                kronos_only=request.kronos_only,
                compact_logs=request.compact_logs,
            )
        finished = datetime.now(timezone.utc)

        LOG.info(
            "Simulation finished timeline_entries=%s run_seconds=%.3f equity=%.2f cash=%.2f",
            len(results["timeline"]),
            results["run_seconds"],
            results["summary"].get("equity", 0.0),
            results["summary"].get("cash", 0.0),
        )

        try:
            self._sync_compiled_models(direction="upload")
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("Failed to upload compiled models: %s", exc)

        return SimulationResponse(
            run_name=f"market-sim-{started.strftime('%Y%m%d_%H%M%S')}",
            started_at=started,
            completed_at=finished,
            timeline=results["timeline"],
            summary=results["summary"],
            run_seconds=float(results["run_seconds"]),
        )

    def _prefetch_reference_artifacts(self) -> None:
        endpoint = os.getenv("R2_ENDPOINT")
        if not endpoint:
            LOG.info("Skipping artifact prefetch (missing R2_ENDPOINT).")
            return
        bucket = os.getenv("R2_BUCKET", "models")
        try:
            specs = load_artifact_specs(repo_root=REPO_ROOT)
        except Exception as exc:
            LOG.warning("Failed to load artifact manifest: %s", exc)
            return
        if not specs:
            LOG.info("Artifact manifest empty; nothing to prefetch.")
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
            LOG.warning("Failed to prefetch artifacts: %s", exc)

    def _sync_hyperparams(self, *, direction: str) -> None:
        endpoint = os.getenv("R2_ENDPOINT")
        if not endpoint:
            LOG.info("Skipping hyperparameter sync (missing R2_ENDPOINT).")
            return
        bucket = os.getenv("R2_BUCKET", "models")
        local = (REPO_ROOT / "hyperparams").resolve()
        local.mkdir(parents=True, exist_ok=True)
        remote = f"s3://{bucket.rstrip('/')}/stock/hyperparams/"
        self._sync_directory(local=local, remote=remote, direction=direction, endpoint=endpoint)

    def _sync_compiled_models(self, *, direction: str) -> None:
        endpoint = os.getenv("R2_ENDPOINT")
        if not endpoint:
            LOG.info("Skipping compiled model sync (missing R2_ENDPOINT).")
            return
        bucket = os.getenv("R2_BUCKET", "models")
        local = (REPO_ROOT / "compiled_models").resolve()
        local.mkdir(parents=True, exist_ok=True)
        remote = f"s3://{bucket.rstrip('/')}/compiled_models/"
        self._sync_directory(local=local, remote=remote, direction=direction, endpoint=endpoint)

    def _sync_directory(self, *, local: Path, remote: str, direction: str, endpoint: str) -> None:
        direction = direction.lower()
        if direction not in {"download", "upload"}:
            raise ValueError(f"Unsupported sync direction: {direction}")
        local = local.resolve()
        if direction == "download":
            source, dest = remote.rstrip("/"), str(local)
        else:
            source, dest = str(local), remote.rstrip("/")
        cmd = ["aws", "s3", "sync", source, dest, "--endpoint-url", endpoint]
        LOG.info("• %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            LOG.warning("AWS CLI not available; skipping sync command.")
        except subprocess.CalledProcessError as exc:
            LOG.warning("aws s3 sync failed (%s): %s", direction, exc)


def create_app() -> MarketSimulatorApp:
    return MarketSimulatorApp()
