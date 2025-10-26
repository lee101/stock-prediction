from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

from src.runtime_imports import setup_src_imports

try:  # pragma: no cover - exercised in production environments
    import torch  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - torch not installed locally
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised in production environments
    import numpy as np  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - numpy not installed locally
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised in production environments
    import pandas as pd  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - pandas optional
    pd = None  # type: ignore[assignment]


def setup_training_imports(
    torch_module: Optional[ModuleType],
    numpy_module: Optional[ModuleType],
    pandas_module: Optional[ModuleType] = None,
) -> None:
    """Register heavy dependencies supplied by the FAL runtime."""

    global torch, np, pd
    if torch_module is not None:
        torch = torch_module
        sys.modules["torch"] = torch_module
    if numpy_module is not None:
        np = numpy_module
        sys.modules["numpy"] = numpy_module
    if pandas_module is not None:
        pd = pandas_module
        sys.modules["pandas"] = pandas_module
    setup_src_imports(torch_module, numpy_module, pandas_module)


def _ensure_injected_modules() -> Tuple[ModuleType, ModuleType, Optional[ModuleType]]:
    global torch, np, pd

    if torch is None:
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Torch is unavailable. Ensure setup_training_imports() has been called inside the FAL app."
            ) from exc
    if np is None:
        try:
            np = importlib.import_module("numpy")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "NumPy is unavailable. Ensure setup_training_imports() has been called inside the FAL app."
            ) from exc
    if pd is None:
        try:
            pd = importlib.import_module("pandas")
        except ModuleNotFoundError:
            pd = None
    setup_training_imports(torch, np, pd)
    return torch, np, pd


def _build_experiment_config(
    config_payload: Dict[str, Any],
    *,
    run_name: Optional[str],
    output_dir: Path,
) -> "ExperimentConfig":
    from hftraining.config import ExperimentConfig, get_default_config

    base_cfg: ExperimentConfig = get_default_config()
    if "seed" in config_payload:
        try:
            base_cfg.system.seed = int(config_payload["seed"])
        except (TypeError, ValueError):
            pass

    training = config_payload.get("training", {})
    if training:
        if "epochs" in training:
            try:
                base_cfg.training.num_epochs = int(training["epochs"])
            except (TypeError, ValueError):
                pass
        if "batch_size" in training:
            try:
                base_cfg.training.batch_size = int(training["batch_size"])
            except (TypeError, ValueError):
                pass
        if "learning_rate" in training:
            try:
                base_cfg.training.learning_rate = float(training["learning_rate"])
            except (TypeError, ValueError):
                pass
        if "transaction_cost_bps" in training:
            try:
                base_cfg.training.transaction_cost_bps = float(training["transaction_cost_bps"])
            except (TypeError, ValueError):
                pass

    data_section = config_payload.get("data", {})
    if data_section:
        if "symbols" in data_section:
            try:
                base_cfg.data.symbols = list(data_section["symbols"])
            except TypeError:
                pass
        if "context_length" in data_section:
            try:
                base_cfg.data.sequence_length = int(data_section["context_length"])
            except (TypeError, ValueError):
                pass
        if "horizon" in data_section:
            try:
                base_cfg.data.prediction_horizon = int(data_section["horizon"])
            except (TypeError, ValueError):
                pass
        if "trainingdata_dir" in data_section:
            base_cfg.data.data_dir = str(data_section["trainingdata_dir"])
        if "validation_data_dir" in data_section and data_section["validation_data_dir"]:
            base_cfg.data.validation_data_dir = str(data_section["validation_data_dir"])
        if "use_toto_forecasts" in data_section:
            base_cfg.data.use_toto_forecasts = bool(data_section["use_toto_forecasts"])

    costs = config_payload.get("costs", {})
    if costs and "transaction_cost_bps" in costs:
        try:
            base_cfg.training.transaction_cost_bps = float(costs["transaction_cost_bps"])
        except (TypeError, ValueError):
            pass

    output_section = config_payload.get("output", {})
    base_cfg.output.output_dir = str(output_dir)
    log_dir = output_section.get("logging_dir") if isinstance(output_section, dict) else None
    cache_dir = output_section.get("cache_dir") if isinstance(output_section, dict) else None
    base_cfg.output.logging_dir = str(log_dir or (output_dir / "logs"))
    base_cfg.output.cache_dir = str(cache_dir or (output_dir / "cache"))
    base_cfg.output.run_name = run_name or base_cfg.output.run_name

    wandb_group = output_section.get("group") if isinstance(output_section, dict) else None
    if wandb_group:
        base_cfg.output.tags = list({*base_cfg.output.tags, wandb_group})

    return base_cfg


def run_training(
    *,
    config: Dict[str, Any],
    run_name: Optional[str],
    output_dir: Path,
) -> Tuple[Dict[str, Any], Path]:
    """Execute HF training in-process and return parsed metrics."""

    _ensure_injected_modules()

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.generated.json"
    config_path.write_text(json.dumps(config, indent=2))

    exp_cfg = _build_experiment_config(config, run_name=run_name, output_dir=output_dir)

    from hftraining.run_training import run_training as hf_run_training

    model, trainer = hf_run_training(exp_cfg)
    # Prevent large CUDA tensors from lingering.
    del model
    del trainer

    metrics_path = output_dir / "final_metrics.json"
    metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            metrics = {}
    return metrics, metrics_path
