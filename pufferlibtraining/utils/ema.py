from __future__ import annotations

import torch


class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model weights that can be swapped in for
    evaluation to stabilise long training runs.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if param.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.state_dict().items():
            if name not in self.shadow or not param.dtype.is_floating_point:
                continue
            self.shadow[name].mul_(self.decay).add_(param, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)
