from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Dict, Optional, Tuple

_TORCH: Optional[ModuleType] = None
_NUMPY: Optional[ModuleType] = None
_PANDAS: Optional[ModuleType] = None


def setup_training_imports(
    torch_module: Optional[ModuleType],
    numpy_module: Optional[ModuleType],
    pandas_module: Optional[ModuleType] = None,
) -> None:
    """Register heavy modules supplied by the fal runtime."""

    global _TORCH, _NUMPY, _PANDAS
    if torch_module is not None:
        _TORCH = torch_module
    if numpy_module is not None:
        _NUMPY = numpy_module
    if pandas_module is not None:
        _PANDAS = pandas_module


def _ensure_injected_modules() -> None:
    if _TORCH is not None:
        sys.modules.setdefault("torch", _TORCH)
    if _NUMPY is not None:
        sys.modules.setdefault("numpy", _NUMPY)
    if _PANDAS is not None:
        sys.modules.setdefault("pandas", _PANDAS)


def _load_train_module():
    from importlib import import_module

    return import_module("tototraining.train")


def run_training(
    *,
    train_root: Path,
    val_root: Optional[Path],
    context_length: int,
    prediction_length: int,
    stride: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    loss: str,
    output_dir: Path,
    device: str = "cuda",
    grad_accum: int = 1,
    weight_decay: float = 1e-2,
    clip_grad: float = 1.0,
    compile: bool = True,
    ema_decay: float = 0.999,
    quantiles: Optional[list[float]] = None,
) -> Tuple[Dict[str, object], Path]:
    """Run Toto training inside the fal worker and return metrics."""

    _ensure_injected_modules()
    module = _load_train_module()

    train_root = Path(train_root)
    if not train_root.exists():
        raise FileNotFoundError(f"Training root not found: {train_root}")

    val_dir = Path(val_root) if val_root else None
    if val_dir is not None and not val_dir.exists():
        val_dir = None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quantiles = list(quantiles or [0.1, 0.5, 0.9])
    effective_device = device
    if effective_device == "cuda" and _TORCH is not None:
        try:
            if not getattr(_TORCH.cuda, "is_available", lambda: False)():
                effective_device = "cpu"
        except Exception:
            effective_device = "cpu"

    args = SimpleNamespace(
        train_root=train_root,
        val_root=val_dir,
        context_length=int(context_length),
        prediction_length=int(prediction_length),
        stride=int(max(1, stride)),
        batch_size=int(batch_size),
        epochs=int(max(1, epochs)),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        grad_accum=max(1, int(grad_accum)),
        clip_grad=float(clip_grad),
        device=str(effective_device),
        compile=bool(compile),
        compile_mode="max-autotune",
        output_dir=output_dir,
        checkpoint_name=f"fal_toto_{uuid.uuid4().hex[:8]}",
        num_workers=max(2, (os.cpu_count() or 4) - 2),
        prefetch_factor=4,
        profile=False,
        profile_logdir=str(output_dir / "profile"),
        prefetch_to_gpu=bool(str(effective_device).startswith("cuda")),
        ema_decay=float(ema_decay),
        ema_eval=True,
        loss=str(loss),
        huber_delta=0.01,
        quantiles=quantiles,
        cuda_graphs=False,
        cuda_graph_warmup=3,
        global_batch=None,
    )

    if hasattr(module, "run_with_namespace"):
        module.run_with_namespace(args)
    else:  # pragma: no cover - compatibility guard
        module.train_args = args  # type: ignore[attr-defined]
        module.train()

    metrics_path = output_dir / "final_metrics.json"
    metrics: Dict[str, object] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            metrics = {}
    return metrics, metrics_path
