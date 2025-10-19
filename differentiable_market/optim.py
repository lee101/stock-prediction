from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

try:
    from nanochat.nanochat.muon import Muon
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Muon = None  # type: ignore
except RuntimeError:  # pragma: no cover - optional dependency
    # torch.compile is not yet available on Python 3.14+, so skip Muon when import hooks fail
    Muon = None  # type: ignore


@dataclass(slots=True)
class MuonConfig:
    lr_muon: float
    lr_adamw: float
    weight_decay: float
    betas: tuple[float, float]
    momentum: float = 0.95
    ns_steps: int = 5


class CombinedOptimizer:
    """Thin wrapper joining Muon and AdamW optimizers."""

    def __init__(
        self,
        muon_opt: Optional[Muon],
        adam_opt: Optional[torch.optim.AdamW],
        weight_decay: float,
    ):
        self._muon = muon_opt
        self._adam = adam_opt
        self.weight_decay = weight_decay
        self.state = {}
        self.param_groups = []
        if self._muon is not None:
            self.param_groups.extend(self._muon.param_groups)
        if self._adam is not None:
            self.param_groups.extend(self._adam.param_groups)
        self.defaults = {}

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self._muon is not None:
            self._muon.zero_grad(set_to_none=set_to_none)
        if self._adam is not None:
            self._adam.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        if self._muon is not None:
            if self.weight_decay != 0.0:
                for group in self._muon.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.data.add_(param.data, alpha=self.weight_decay)
            self._muon.step()
        if self._adam is not None:
            self._adam.step()

    def state_dict(self) -> dict:
        return {
            "muon": None if self._muon is None else self._muon.state_dict(),
            "adam": None if self._adam is None else self._adam.state_dict(),
            "weight_decay": self.weight_decay,
        }

    def load_state_dict(self, state: dict) -> None:
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        if self._muon is not None and state.get("muon") is not None:
            self._muon.load_state_dict(state["muon"])
        if self._adam is not None and state.get("adam") is not None:
            self._adam.load_state_dict(state["adam"])


def build_muon_optimizer(
    matrix_params: Iterable[torch.nn.Parameter],
    residual_params: Iterable[torch.nn.Parameter],
    cfg: MuonConfig,
) -> Optional[CombinedOptimizer]:
    matrix_params = list(matrix_params)
    residual_params = list(residual_params)
    if not matrix_params or Muon is None:
        return None

    muon_opt = Muon(
        params=matrix_params,
        lr=cfg.lr_muon,
        momentum=cfg.momentum,
        ns_steps=cfg.ns_steps,
    )
    adam_opt = None
    if residual_params:
        adam_opt = torch.optim.AdamW(
            residual_params,
            lr=cfg.lr_adamw,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
    return CombinedOptimizer(muon_opt, adam_opt, weight_decay=cfg.weight_decay)

