from __future__ import annotations

import warnings
from typing import Iterable, Dict, Any, List, Tuple

import torch
from torch.optim import Optimizer


def _maybe_import(module: str, name: str):
    try:
        mod = __import__(module, fromlist=[name])
        return getattr(mod, name)
    except Exception:
        return None


_Lion = _maybe_import("lion_pytorch", "Lion") or _maybe_import("torch_optimizer", "Lion")
_Adafactor = _maybe_import("transformers", "Adafactor")
_Shampoo = _maybe_import("torch_optimizer", "Shampoo")
_Adan = _maybe_import("torch_optimizer", "Adan")
_Muon = _maybe_import("muon", "Muon")


def _patch_muon_single_process() -> None:
    if _Muon is None:
        return
    try:
        import muon  # type: ignore
        import torch.distributed as dist_mod
    except Exception:
        return

    if getattr(muon, "_single_process_patched", False):
        return

    if getattr(dist_mod, "is_available", lambda: False)() and getattr(dist_mod, "is_initialized", lambda: False)():
        return

    class _SingleProcessDist:
        def get_world_size(self) -> int:
            return 1

        def get_rank(self) -> int:
            return 0

        def all_gather(self, output, tensor) -> None:
            if isinstance(output, (list, tuple)):
                for out in output:
                    out.copy_(tensor)
            else:
                output.copy_(tensor)

    muon.dist = _SingleProcessDist()  # type: ignore[attr-defined]
    muon._single_process_patched = True  # type: ignore[attr-defined]


def _no_decay(name: str) -> bool:
    name = name.lower()
    if name.endswith("bias"):
        return True
    if "layernorm" in name or "ln" in name or "norm" in name:
        return True
    if "embedding" in name:
        return True
    return False


def _create_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    extra_no_decay: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    no_decay_set = set(extra_no_decay or [])
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _no_decay(name) or any(token in name for token in no_decay_set) or param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


class MultiOptim(torch.optim.Optimizer):
    """
    Lightweight wrapper to step multiple optimisers together (for Muon mixes).
    """

    def __init__(self, optimizers: List[Optimizer]):
        self.optimizers = optimizers
        self._manual_param_groups = []
        super().__init__([{"params": []}], {})

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value):  # pragma: no cover - setter required for torch internals
        self._manual_param_groups = value

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        if "optimizers" in state_dict and isinstance(state_dict["optimizers"], list):
            for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
                opt.load_state_dict(sd)
            return

        # Backwards compatibility: allow loading a single optimizer state dict.
        if len(self.optimizers) == 1:
            self.optimizers[0].load_state_dict(state_dict)
            return

        for opt in self.optimizers:
            opt.load_state_dict(state_dict)

    def zero_grad(self, set_to_none: bool | None = None):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        for opt in self.optimizers:
            loss = opt.step(closure)
        return loss


def _fused_ok() -> bool:
    return torch.cuda.is_available() and torch.__version__ >= "2.0"


def make_optimizer(
    model: torch.nn.Module,
    name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool = True,
    extra_no_decay: Iterable[str] | None = None,
) -> Optimizer:
    """
    Unified optimiser factory with optional Muon mix support.
    Supported names: adamw, lion, adafactor, shampoo, adan, muon, muon_mix.
    """
    name = name.lower()
    groups = _create_param_groups(model, weight_decay=weight_decay, extra_no_decay=extra_no_decay)

    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused and _fused_ok())

    if name == "lion":
        if _Lion is None:
            warnings.warn("Lion optimizer not available; falling back to AdamW.")
            return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused and _fused_ok())
        return _Lion(groups, lr=lr, weight_decay=weight_decay)

    if name == "adafactor":
        if _Adafactor is None:
            warnings.warn("Adafactor not available; falling back to AdamW.")
            return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused and _fused_ok())
        return _Adafactor(groups, lr=lr, relative_step=False, scale_parameter=False, warmup_init=False)

    if name == "shampoo":
        if _Shampoo is None:
            warnings.warn("Shampoo not available; falling back to AdamW.")
            return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused and _fused_ok())
        return _Shampoo(groups, lr=lr, weight_decay=weight_decay)

    if name == "adan":
        if _Adan is None:
            warnings.warn("Adan not available; falling back to AdamW.")
            return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused and _fused_ok())
        return _Adan(groups, lr=lr, weight_decay=weight_decay)

    if name == "muon":
        if _Muon is None:
            warnings.warn("Muon not available; falling back to AdamW.")
            return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused and _fused_ok())
        _patch_muon_single_process()
        return _Muon(groups, lr=lr, weight_decay=weight_decay)

    if name in {"muon_mix", "muon+adamw"}:
        if _Muon is None:
            warnings.warn("Muon not available; falling back to AdamW.")
            return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused and _fused_ok())
        _patch_muon_single_process()

        muon_groups, adam_groups = [], []
        for g in groups:
            two_d, others = [], []
            for p in g["params"]:
                if not p.requires_grad:
                    continue
                (two_d if getattr(p, "ndim", 0) == 2 else others).append(p)
            if two_d:
                muon_groups.append({"params": two_d, "weight_decay": g["weight_decay"]})
            if others:
                adam_groups.append({"params": others, "weight_decay": g["weight_decay"]})

        muon_opt = None
        if muon_groups:
            unique_wds = {mg["weight_decay"] for mg in muon_groups}
            muon_opts = []
            for wd in unique_wds:
                params = []
                for mg in muon_groups:
                    if mg["weight_decay"] == wd:
                        params.extend(mg["params"])
                if not params:
                    continue
                muon_opts.append(_Muon(params, lr=lr, weight_decay=wd))
            if muon_opts:
                muon_opt = muon_opts[0] if len(muon_opts) == 1 else MultiOptim(muon_opts)

        adam_opt = torch.optim.AdamW(adam_groups, lr=lr, betas=betas, eps=eps, fused=fused and _fused_ok()) if adam_groups else None
        optimizers = [opt for opt in (muon_opt, adam_opt) if opt is not None]
        if len(optimizers) == 1:
            return optimizers[0]
        return MultiOptim(optimizers)

    raise ValueError(f"Unknown optimizer '{name}'.")
