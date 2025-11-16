from __future__ import annotations

from typing import Iterable

import torch


class Muon(torch.optim.Optimizer):
    """Momentum-based optimizer inspired by NanoChat's Muon variant."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        adaptive: bool = True,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, adaptive=adaptive)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            adaptive = group["adaptive"]
            lr = group["lr"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if weight_decay:
                    grad = grad.add(param, alpha=weight_decay)

                state = self.state[param]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(param)
                buf.mul_(momentum).add_(grad)

                if adaptive:
                    grad_norm = grad.norm().clamp(min=1e-12)
                    step_size = lr * (1.0 / (1.0 + grad_norm))
                else:
                    step_size = lr

                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf
                param.add_(update, alpha=-step_size)

        return loss


__all__ = ["Muon"]
