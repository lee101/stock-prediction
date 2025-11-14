"""
Helpers for plugging the optimizer registry into Hugging Face `Trainer`.

The Hugging Face API allows overriding optimizers by passing an `(optimizer,
scheduler)` tuple to the `Trainer` constructor or by overriding
`create_optimizer`.  We keep the helpers in this module small and explicit so
they can be reused from scripts as well as notebooks.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple

Trainer: Any | None = None

try:
    from transformers import Trainer as _Trainer
except ModuleNotFoundError:  # pragma: no cover - import guarded at runtime.
    Trainer = None
else:
    Trainer = _Trainer

from .optimizers import create_optimizer, optimizer_registry

SchedulerBuilder = Callable[[Any, int], Any]


def build_hf_optimizers(
    model,
    optimizer_name: str,
    *,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    optimizer_kwargs: Optional[MutableMapping[str, Any]] = None,
    scheduler_builder: Optional[SchedulerBuilder] = None,
    num_training_steps: Optional[int] = None,
) -> Tuple[Any, Optional[Any]]:
    """
    Construct a Hugging Face compatible `(optimizer, scheduler)` tuple.

    Parameters
    ----------
    model:
        The model whose parameters should be optimised.
    optimizer_name:
        Key registered in :mod:`traininglib.optimizers`.
    lr, weight_decay:
        Optional overrides for learning rate / weight decay. If omitted we use
        the defaults associated with the registered optimizer.
    optimizer_kwargs:
        Additional kwargs forwarded to the optimizer factory.
    scheduler_builder:
        Optional callable receiving `(optimizer, num_training_steps)` and
        returning a scheduler instance compatible with `Trainer`.
    num_training_steps:
        Required when `scheduler_builder` needs to know the total number of
        steps up front.
    """
    defaults = optimizer_registry.get_defaults(optimizer_name)
    config = dict(defaults)
    if lr is not None:
        config["lr"] = lr
    if weight_decay is not None:
        config["weight_decay"] = weight_decay
    if optimizer_kwargs:
        config.update(optimizer_kwargs)

    optimizer = create_optimizer(optimizer_name, model.parameters(), **config)
    scheduler = None
    if scheduler_builder is not None:
        if num_training_steps is None:
            raise ValueError(
                "num_training_steps must be provided when using scheduler_builder."
            )
        scheduler = scheduler_builder(optimizer, num_training_steps)
    return optimizer, scheduler


def attach_optimizer_to_trainer(
    trainer: "Trainer",
    optimizer_name: str,
    *,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    optimizer_kwargs: Optional[MutableMapping[str, Any]] = None,
    scheduler_builder: Optional[SchedulerBuilder] = None,
    num_training_steps: Optional[int] = None,
) -> Tuple[Any, Optional[Any]]:
    """
    Mutate an existing Trainer so it uses the registry-backed optimizer.

    This keeps the Trainer lifecycle untouched: once attached, calls to
    `trainer.create_optimizer_and_scheduler` reuse the custom choice.
    """
    if Trainer is None:  # pragma: no cover - defensive branch.
        raise RuntimeError("transformers must be installed to attach optimizers.")

    optimizer, scheduler = build_hf_optimizers(
        trainer.model,
        optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_builder=scheduler_builder,
        num_training_steps=num_training_steps,
    )
    setattr(trainer, "create_optimizer", lambda: optimizer)
    setattr(trainer, "create_optimizer_and_scheduler", lambda _: (optimizer, scheduler))
    trainer.optimizers = (optimizer, scheduler)
    return optimizer, scheduler
