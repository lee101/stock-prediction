"""
Optimizer registry for the project.

The goal here is to make it trivial to experiment with alternative optimizers
without copy/pasting setup code across notebooks or training entry points.  The
registry keeps a map of short names (``"adamw"``, ``"shampoo"``, ``"muon"`` …)
to callables that build the optimizer directly from a set of model parameters.

In practice almost every consumer will interact with the module through
``create_optimizer`` which merges per-optimizer default kwargs with the kwargs
provided at call time.  The defaults live alongside the factory to keep the
logic discoverable and easy to override in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

torch: ModuleType | None = None

try:  # torch is optional at import time so unit tests can guard explicitly.
    import torch as _torch_mod
    from torch.optim import Optimizer as _TorchOptimizer
except ModuleNotFoundError:  # pragma: no cover - exercised when torch missing.
    TorchOptimizer = Any  # type: ignore[misc]
else:
    torch = _torch_mod
    TorchOptimizer = _TorchOptimizer


OptimizerFactory = Callable[[Iterable], TorchOptimizer]


def _ensure_dependency(module: str, install_hint: str) -> Any:
    """Import a module lazily and provide a helpful installation hint."""
    import importlib

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive branch.
        raise RuntimeError(
            f"Optimizer requires '{module}'. Install it with `{install_hint}`."
        ) from exc


@dataclass
class OptimizerSpec:
    """Container keeping metadata around a registered optimizer."""

    name: str
    factory: OptimizerFactory
    defaults: MutableMapping[str, Any] = field(default_factory=dict)

    def build(self, params: Iterable, **overrides: Any) -> TorchOptimizer:
        # Merge without mutating the stored defaults.
        config = dict(self.defaults)
        config.update(overrides)
        return self.factory(params, **config)


class OptimizerRegistry:
    """Simple name → optimizer factory mapping."""

    def __init__(self) -> None:
        self._registry: Dict[str, OptimizerSpec] = {}

    def register(
        self,
        name: str,
        factory: OptimizerFactory,
        *,
        defaults: Optional[Mapping[str, Any]] = None,
        override: bool = False,
    ) -> None:
        key = name.lower()
        if key in self._registry and not override:
            raise ValueError(f"Optimizer '{name}' already registered.")
        self._registry[key] = OptimizerSpec(
            name=key,
            factory=factory,
            defaults=dict(defaults or {}),
        )

    def unregister(self, name: str) -> None:
        self._registry.pop(name.lower())

    def create(self, name: str, params: Iterable, **overrides: Any) -> TorchOptimizer:
        key = name.lower()
        if key not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(f"Optimizer '{name}' is not registered. Known: {available}")
        return self._registry[key].build(params, **overrides)

    def get_defaults(self, name: str) -> Mapping[str, Any]:
        key = name.lower()
        if key not in self._registry:
            raise KeyError(f"Optimizer '{name}' is not registered.")
        return dict(self._registry[key].defaults)

    def names(self) -> Iterable[str]:
        return tuple(sorted(self._registry))

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._registry


optimizer_registry = OptimizerRegistry()


def _register_builtin_optimizers() -> None:
    if torch is None:  # pragma: no cover - torch missing is validated elsewhere.
        return

    def _adamw_factory(params: Iterable, **kwargs: Any) -> TorchOptimizer:
        return torch.optim.AdamW(params, **kwargs)

    optimizer_registry.register(
        "adamw",
        _adamw_factory,
        defaults={"lr": 1e-3, "weight_decay": 0.01},
    )

    def _adam_factory(params: Iterable, **kwargs: Any) -> TorchOptimizer:
        return torch.optim.Adam(params, **kwargs)

    optimizer_registry.register(
        "adam",
        _adam_factory,
        defaults={"lr": 1e-3},
    )

    def _sgd_factory(params: Iterable, **kwargs: Any) -> TorchOptimizer:
        return torch.optim.SGD(params, **kwargs)

    optimizer_registry.register(
        "sgd",
        _sgd_factory,
        defaults={"lr": 1e-2, "momentum": 0.9, "nesterov": True},
    )

    def _shampoo_factory(params: Iterable, **kwargs: Any) -> TorchOptimizer:
        torch_optimizer = _ensure_dependency(
            "torch_optimizer",
            "pip install torch-optimizer",
        )
        return torch_optimizer.Shampoo(params, **kwargs)

    optimizer_registry.register(
        "shampoo",
        _shampoo_factory,
        defaults={
            "lr": 0.05,
            "momentum": 0.0,
            "epsilon": 1e-4,
            "update_freq": 1,
            "weight_decay": 0.0,
        },
    )

    def _muon_factory(params: Iterable, **kwargs: Any) -> TorchOptimizer:
        pytorch_optimizer = _ensure_dependency(
            "pytorch_optimizer",
            "pip install pytorch-optimizer",
        )
        param_list = list(params)
        if not param_list:
            raise ValueError("Muon optimizer received an empty parameter list.")
        param_groups = []
        for tensor in param_list:
            use_muon = getattr(tensor, "ndim", 0) >= 2
            param_groups.append({"params": [tensor], "use_muon": use_muon})
        return pytorch_optimizer.Muon(param_groups, **kwargs)

    optimizer_registry.register(
        "muon",
        _muon_factory,
        defaults={
            "lr": 0.02,
            "momentum": 0.95,
            "weight_decay": 0.0,
            "weight_decouple": True,
            "nesterov": True,
            "ns_steps": 5,
            "use_adjusted_lr": False,
            "adamw_lr": 3e-4,
            "adamw_betas": (0.9, 0.95),
            "adamw_wd": 0.0,
            "adamw_eps": 1e-10,
        },
    )

    def _lion_factory(params: Iterable, **kwargs: Any) -> TorchOptimizer:
        pytorch_optimizer = _ensure_dependency(
            "pytorch_optimizer",
            "pip install pytorch-optimizer",
        )
        return pytorch_optimizer.Lion(params, **kwargs)

    optimizer_registry.register(
        "lion",
        _lion_factory,
        defaults={"lr": 3e-4, "betas": (0.9, 0.95), "weight_decay": 0.0},
    )

    def _adafactor_factory(params: Iterable, **kwargs: Any) -> TorchOptimizer:
        transformers_opt = _ensure_dependency(
            "transformers.optimization",
            "pip install transformers",
        )
        return transformers_opt.Adafactor(params, **kwargs)

    optimizer_registry.register(
        "adafactor",
        _adafactor_factory,
        defaults={
            "lr": None,
            "scale_parameter": True,
            "relative_step": True,
            "warmup_init": True,
        },
    )


_register_builtin_optimizers()


def create_optimizer(name: str, params: Iterable, **kwargs: Any) -> TorchOptimizer:
    """Public helper wrapping ``optimizer_registry.create``."""
    return optimizer_registry.create(name, params, **kwargs)
