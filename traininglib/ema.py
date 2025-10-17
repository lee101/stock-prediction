"""Exponential moving average weights for evaluation stability."""

from __future__ import annotations

from typing import Dict

import torch


class EMA:
    """Keep a shadow copy of model parameters updated with exponential decay."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError("EMA decay must lie in (0, 1).")

        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        self._register(model)

    @torch.no_grad()
    def _register(self, model: torch.nn.Module) -> None:
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if name not in self.shadow or not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


__all__ = ["EMA"]
