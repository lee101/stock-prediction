from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

from faltrain.dependencies import bulk_register_fal_dependencies, get_fal_dependencies

_TORCH: Optional[ModuleType] = None
_NUMPY: Optional[ModuleType] = None
_PANDAS: Optional[ModuleType] = None


def setup_training_imports(
    torch_module: Optional[ModuleType],
    numpy_module: Optional[ModuleType],
    pandas_module: Optional[ModuleType] = None,
) -> None:
    """Register heavy dependencies for the RL pipeline."""

    global _TORCH, _NUMPY, _PANDAS
    mapping: Dict[str, ModuleType] = {}
    if torch_module is not None:
        _TORCH = torch_module
        mapping["torch"] = torch_module
    if numpy_module is not None:
        _NUMPY = numpy_module
        mapping["numpy"] = numpy_module
    if pandas_module is not None:
        _PANDAS = pandas_module
        mapping["pandas"] = pandas_module
    if mapping:
        bulk_register_fal_dependencies(mapping)


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

    get_fal_dependencies("torch", "numpy")
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
