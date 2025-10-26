from __future__ import annotations

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

from src.runtime_imports import setup_src_imports

try:  # pragma: no cover - production environments
    import torch  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - production environments
    import numpy as np  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - production environments
    import pandas as pd  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


def setup_training_imports(
    torch_module: Optional[ModuleType],
    numpy_module: Optional[ModuleType],
    pandas_module: Optional[ModuleType] = None,
) -> None:
    """Register heavy dependencies for the RL pipeline."""

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
                "Torch is unavailable. Ensure setup_training_imports() runs inside the FAL worker."
            ) from exc
    if np is None:
        try:
            np = importlib.import_module("numpy")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "NumPy is unavailable. Ensure setup_training_imports() runs inside the FAL worker."
            ) from exc
    if pd is None:
        try:
            pd = importlib.import_module("pandas")
        except ModuleNotFoundError:
            pd = None
    setup_training_imports(torch, np, pd)
    return torch, np, pd


def _default_args() -> Namespace:
    from pufferlibtraining.train_ppo import build_argument_parser

    parser = build_argument_parser()
    return parser.parse_args([])


def _prepare_args(
    *,
    base_args: Namespace,
    trainingdata_dir: Path,
    output_dir: Path,
    tensorboard_dir: Path,
    cfg: Dict[str, Any],
    epochs: int,
    transaction_cost_bps: float,
    run_name: str,
) -> Namespace:
    args = Namespace(**vars(base_args))
    args.trainingdata_dir = str(trainingdata_dir)
    args.output_dir = str(output_dir)
    args.tensorboard_dir = str(tensorboard_dir)
    args.summary_path = str(output_dir / "summary.json")
    args.rl_epochs = int(epochs)
    args.rl_batch_size = int(cfg.get("batch_size", args.rl_batch_size))
    args.rl_learning_rate = float(cfg.get("learning_rate", args.rl_learning_rate))
    args.rl_optimizer = str(cfg.get("optimizer", getattr(args, "rl_optimizer", "adamw")))
    args.transaction_cost_bps = float(cfg.get("transaction_cost_bps", transaction_cost_bps))
    args.wandb_run_name = run_name
    if getattr(args, "wandb_group", None):
        args.wandb_group = args.wandb_group or run_name
    return args


def run_training(
    *,
    trainingdata_dir: Path,
    output_dir: Path,
    tensorboard_dir: Path,
    cfg: Dict[str, Any],
    epochs: int,
    transaction_cost_bps: float,
    run_name: str,
) -> Tuple[Dict[str, Any], Path]:
    """Execute the RL pipeline in-process and return the summary."""

    _ensure_injected_modules()
    trainingdata_dir = trainingdata_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    tensorboard_dir = tensorboard_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    base_args = _default_args()
    args = _prepare_args(
        base_args=base_args,
        trainingdata_dir=trainingdata_dir,
        output_dir=output_dir,
        tensorboard_dir=tensorboard_dir,
        cfg=cfg,
        epochs=epochs,
        transaction_cost_bps=transaction_cost_bps,
        run_name=run_name,
    )

    from pufferlibtraining.train_ppo import run_pipeline

    summary = run_pipeline(args)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary, summary_path
