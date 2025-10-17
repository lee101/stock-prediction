#!/usr/bin/env python3
"""
Unified experiment tracker that mirrors metrics to both Weights & Biases and TensorBoard.

The primary goal of this helper is to make it trivial for the training pipelines to keep their
existing TensorBoard integrations while automatically mirroring the same metrics, figures, and
metadata to Weights & Biases when it is available. When `wandb` cannot be imported or the project
configuration is missing, the logger silently falls back to TensorBoard-only mode.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from torch.utils.tensorboard import SummaryWriter

try:  # pragma: no cover - optional dependency
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when wandb missing
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False

Number = Union[int, float]
Scalar = Union[int, float, bool]
logger = logging.getLogger(__name__)


def _ensure_dir(path: Union[str, Path]) -> Path:
    """Create `path` if needed and return it as a Path object."""
    path_obj = Path(path).expanduser().resolve()
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _is_scalar(value: Any) -> bool:
    if isinstance(value, (int, float, bool)):
        return True
    if hasattr(value, "item"):
        try:
            value.item()
            return True
        except Exception:
            return False
    return False


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        return float(value.item())
    raise TypeError(f"Unsupported scalar type: {type(value)!r}")


def _sanitize(obj: Any, max_depth: int = 3) -> Any:
    """Convert complex config objects into something JSON-serialisable."""
    if max_depth <= 0:
        return str(obj)

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, Mapping):
        return {str(k): _sanitize(v, max_depth - 1) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(item, max_depth - 1) for item in obj]

    if hasattr(obj, "__dataclass_fields__"):
        return {
            str(field_name): _sanitize(getattr(obj, field_name), max_depth - 1)
            for field_name in obj.__dataclass_fields__  # type: ignore[attr-defined]
        }

    if hasattr(obj, "__dict__"):
        return {
            str(k): _sanitize(v, max_depth - 1)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }

    return str(obj)


class WandBoardLogger(AbstractContextManager):
    """Mirror metrics to Weights & Biases while keeping TensorBoard writes intact."""

    def __init__(
        self,
        *,
        run_name: Optional[str] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        group: Optional[str] = None,
        notes: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        mode: str = "auto",
        enable_wandb: bool = True,
        log_dir: Optional[Union[str, Path]] = None,
        tensorboard_subdir: Optional[str] = None,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"run_{timestamp}"
        self.project = project or os.getenv("WANDB_PROJECT")
        self.entity = entity or os.getenv("WANDB_ENTITY")
        self.tags = tuple(tags) if tags else tuple()
        self.group = group
        self.notes = notes
        self.mode = (mode or os.getenv("WANDB_MODE") or "auto").lower()
        self.settings = dict(settings or {})

        self._last_error: Optional[Exception] = None
        self._wandb_run = None
        self._wandb_enabled = enable_wandb and _WANDB_AVAILABLE and bool(self.project)

        root_dir = _ensure_dir(log_dir or "tensorboard_logs")
        subdir = tensorboard_subdir or self.run_name
        self.tensorboard_log_dir = _ensure_dir(root_dir / subdir)
        self.tensorboard_writer = SummaryWriter(log_dir=str(self.tensorboard_log_dir))

        if enable_wandb and not _WANDB_AVAILABLE:
            logger.info("wandb package not available; continuing with TensorBoard only.")

        if enable_wandb and _WANDB_AVAILABLE and not self.project:
            logger.info(
                "WANDB project not configured (set WANDB_PROJECT or pass project=); falling back to TensorBoard only."
            )

        if self._wandb_enabled:
            init_kwargs: Dict[str, Any] = {
                "project": self.project,
                "entity": self.entity,
                "name": self.run_name,
                "tags": list(self.tags) if self.tags else None,
                "group": self.group,
                "notes": self.notes,
                "mode": None if self.mode == "auto" else self.mode,
                "config": _sanitize(config) if config is not None else None,
                "settings": dict(self.settings) or None,
            }
            # Remove None values to avoid wandb complaining.
            init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
            try:
                self._wandb_run = wandb.init(**init_kwargs)
            except Exception as exc:  # pragma: no cover - network dependent
                self._last_error = exc
                self._wandb_run = None
                self._wandb_enabled = False
                logger.warning("Failed to initialise wandb run; falling back to TensorBoard only: %s", exc)

    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #
    @property
    def wandb_enabled(self) -> bool:
        return self._wandb_run is not None

    @property
    def last_error(self) -> Optional[Exception]:
        return self._last_error

    def log(
        self,
        metrics: Mapping[str, Any],
        *,
        step: Optional[int] = None,
        commit: Optional[bool] = None,
    ) -> None:
        """Log scalar metrics to both backends."""
        if not metrics:
            return

        scalars: Dict[str, float] = {}
        for key, value in metrics.items():
            if not _is_scalar(value):
                continue
            try:
                scalars[key] = _to_float(value)
            except Exception:
                continue

        if not scalars:
            return

        if self.tensorboard_writer is not None:
            for key, value in scalars.items():
                self.tensorboard_writer.add_scalar(key, value, global_step=step)

        if self._wandb_run is not None:
            log_kwargs: Dict[str, Any] = {}
            if step is not None:
                log_kwargs["step"] = step
            if commit is not None:
                log_kwargs["commit"] = commit
            try:
                self._wandb_run.log(scalars, **log_kwargs)
            except Exception as exc:  # pragma: no cover - network dependent
                self._last_error = exc
                logger.warning("wandb.log failed: %s", exc)

    def add_scalar(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """Compatibility helper mirroring TensorBoard's API."""
        self.log({name: value}, step=step)

    def log_text(self, name: str, text: str, *, step: Optional[int] = None) -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_text(name, text, global_step=step)
        if self._wandb_run is not None:
            try:
                self._wandb_run.log({name: text}, step=step)
            except Exception as exc:  # pragma: no cover
                self._last_error = exc
                logger.warning("wandb.log(text) failed: %s", exc)

    def log_figure(self, name: str, figure: Any, *, step: Optional[int] = None) -> None:
        if self.tensorboard_writer is not None:
            try:
                self.tensorboard_writer.add_figure(name, figure, global_step=step)
            except Exception as exc:
                logger.debug("Failed to add figure to TensorBoard: %s", exc)
        if self._wandb_run is not None:
            try:
                self._wandb_run.log({name: wandb.Image(figure)}, step=step)
            except Exception as exc:  # pragma: no cover
                self._last_error = exc
                logger.warning("wandb.log(figure) failed: %s", exc)

    def log_table(
        self,
        name: str,
        columns: Sequence[str],
        data: Iterable[Sequence[Any]],
        *,
        step: Optional[int] = None,
    ) -> None:
        if self._wandb_run is None:
            return
        try:
            table = wandb.Table(columns=list(columns), data=list(data))
            self._wandb_run.log({name: table}, step=step)
        except Exception as exc:  # pragma: no cover
            self._last_error = exc
            logger.warning("wandb.log(table) failed: %s", exc)

    def watch(self, *args: Any, **kwargs: Any) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.watch(*args, **kwargs)
        except Exception as exc:  # pragma: no cover
            self._last_error = exc
            logger.warning("wandb.watch failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def flush(self) -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()

    def finish(self) -> None:
        """Flush and close both backends."""
        if self.tensorboard_writer is not None:
            try:
                self.tensorboard_writer.flush()
                self.tensorboard_writer.close()
            finally:
                self.tensorboard_writer = None

        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            finally:
                self._wandb_run = None

    def close(self) -> None:
        self.finish()

    def __enter__(self) -> "WandBoardLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()


__all__ = ["WandBoardLogger"]
