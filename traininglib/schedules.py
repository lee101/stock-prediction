from __future__ import annotations

import math
from typing import List

from torch.optim import Optimizer


class WarmupCosine:
    """
    Simple step-based cosine schedule with linear warmup.
    Call step() after each optimizer.step().
    """

    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0):
        assert total_steps > 0, "total_steps must be positive"
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        self._step = 0
        self.base_lrs: List[float] = [group.get("initial_lr", group["lr"]) for group in optimizer.param_groups]
        self._last_lrs: List[float] = list(self.base_lrs)

    def state_dict(self):
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "step": self._step,
            "base_lrs": self.base_lrs,
            "last_lrs": self._last_lrs,
        }

    def load_state_dict(self, state):
        self.warmup_steps = state["warmup_steps"]
        self.total_steps = state["total_steps"]
        self.min_lr = state["min_lr"]
        self._step = state["step"]
        self.base_lrs = state["base_lrs"]
        self._last_lrs = state.get("last_lrs", list(self.base_lrs))

    def _lr_multiplier(self) -> float:
        if self._step < self.warmup_steps and self.warmup_steps > 0:
            return float(self._step) / float(max(1, self.warmup_steps))
        progress = (self._step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self):
        self._step += 1
        mult = self._lr_multiplier()
        updated = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            new_lr = self.min_lr + (base_lr - self.min_lr) * mult
            group["lr"] = new_lr
            updated.append(new_lr)
        self._last_lrs = updated

    def get_last_lr(self) -> List[float]:
        return list(self._last_lrs)
