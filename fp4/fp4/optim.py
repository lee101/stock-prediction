"""AdamW with FP32 master weights and BF16 momentum (NeMo NVFP4 recipe)."""
from __future__ import annotations

import torch
from torch.optim import Optimizer


class AdamWMaster(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["master"] = p.data.detach().to(torch.float32).clone()
                    # BF16 momentum (cast to fp32 inside math) — fall back to fp32
                    # if the device doesn't support bf16 (rare).
                    mom_dtype = torch.bfloat16 if p.is_floating_point() else torch.float32
                    try:
                        state["exp_avg"] = torch.zeros_like(state["master"], dtype=mom_dtype)
                        state["exp_avg_sq"] = torch.zeros_like(state["master"], dtype=mom_dtype)
                    except Exception:
                        state["exp_avg"] = torch.zeros_like(state["master"])
                        state["exp_avg_sq"] = torch.zeros_like(state["master"])
                state["step"] += 1
                step = state["step"]
                master = state["master"]
                exp_avg = state["exp_avg"].to(torch.float32)
                exp_avg_sq = state["exp_avg_sq"].to(torch.float32)

                g32 = grad.to(torch.float32)
                if wd != 0.0:
                    master.mul_(1 - lr * wd)
                exp_avg.mul_(beta1).add_(g32, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g32, g32, value=1 - beta2)
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                denom = (exp_avg_sq.sqrt() / (bc2 ** 0.5)).add_(eps)
                master.addcdiv_(exp_avg, denom, value=-lr / bc1)

                state["exp_avg"].copy_(exp_avg.to(state["exp_avg"].dtype))
                state["exp_avg_sq"].copy_(exp_avg_sq.to(state["exp_avg_sq"].dtype))
                p.data.copy_(master.to(p.dtype))
        return loss
