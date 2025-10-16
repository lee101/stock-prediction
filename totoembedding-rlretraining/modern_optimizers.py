#!/usr/bin/env python3
"""
Modern Optimizers for RL Training
Borrowed from HuggingFace training but adapted for RL
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class GPro(torch.optim.Optimizer):
    """
    GPro Optimizer - Gradient Projection with adaptive preconditioning
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, amsgrad=False, projection_factor=0.5):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                       amsgrad=amsgrad, projection_factor=projection_factor)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data).float()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Add weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Update exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group['amsgrad']:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                # Gradient projection step
                direction = exp_avg / denom
                
                # Apply projection factor for better stability
                if group['projection_factor'] != 1.0:
                    direction = direction * group['projection_factor']
                
                p.data.add_(direction, alpha=-step_size)

        return loss


class Lion(torch.optim.Optimizer):
    """
    Lion Optimizer - Discovered through evolutionary search
    Simpler and more memory-efficient than Adam
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.data.add_(update.sign(), alpha=-group['lr'])

                # Momentum update
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


class AdaFactor(torch.optim.Optimizer):
    """
    AdaFactor optimizer from 'Adafactor: Adaptive Learning Rates with Sublinear Memory Cost'
    Memory-efficient alternative to Adam
    """
    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        cliping_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")

        defaults = dict(
            lr=lr,
            eps=eps,
            cliping_threshold=cliping_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)

    def _get_lr(self, param_group, param_state):
        if param_group["lr"] is None:
            step = param_state["step"]
            if param_group["warmup_init"]:
                base_lr = 1e-6 * step
            else:
                base_lr = 1.0

            if param_group["relative_step"]:
                min_step = 1e-10 if param_group["warmup_init"] else 1e-2
                base_lr = base_lr * min(min_step, 1.0 / math.sqrt(step))
            
            param_scale = 1
            if param_group["scale_parameter"]:
                param_scale = math.sqrt(param_state["param_scale"])
            
            return param_scale * base_lr
        
        return param_group["lr"]

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2 and param_shape[0] * param_shape[1] >= 32
        use_first_moment = param_group["beta1"]
        return factored, use_first_moment

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col, update):
        r_factor = (
            ((exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_())
            .unsqueeze(1)
        )
        c_factor = (
            (exp_avg_sq_col.rsqrt()).unsqueeze(0)
        )
        v = r_factor * c_factor

        v.mul_(update)
        return v

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)

                # State initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        state["exp_avg"] = torch.zeros_like(grad)
                    
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[0])
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[1:].numel())
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                    if group["scale_parameter"]:
                        state["param_scale"] = p.data.abs().mean().item() ** 2

                state["step"] += 1
                lr = self._get_lr(group, state)
                
                # Exponential moving average of gradient values
                if use_first_moment:
                    state["exp_avg"].mul_(group["beta1"]).add_(grad, alpha=1 - group["beta1"])

                if factored:
                    eps = group["eps"][0]
                    row_mean = grad.mean(dim=list(range(1, len(grad_shape))))
                    state["exp_avg_sq_row"].mul_(group["decay_rate"]).add_(row_mean ** 2, alpha=1 - group["decay_rate"])
                    col_mean = grad.view(grad_shape[0], -1).mean(dim=0)
                    state["exp_avg_sq_col"].mul_(group["decay_rate"]).add_(col_mean ** 2, alpha=1 - group["decay_rate"])
                    update = grad
                    if use_first_moment:
                        update = state["exp_avg"]
                    
                    update = self._approx_sq_grad(
                        state["exp_avg_sq_row"],
                        state["exp_avg_sq_col"],
                        update,
                    )
                    update.div_((state["RMS"] / group["cliping_threshold"]).clamp(min=1.0))
                else:
                    eps = group["eps"][1]
                    state["exp_avg_sq"].mul_(group["decay_rate"]).add_(grad ** 2, alpha=1 - group["decay_rate"])
                    update = grad
                    if use_first_moment:
                        update = state["exp_avg"]
                    
                    update = update.rsqrt().mul_(update).add_(eps)
                    update.div_((state["RMS"] / group["cliping_threshold"]).clamp(min=1.0))

                state["RMS"] = self._rms(update)

                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * lr)

                p.data.add_(update, alpha=-lr)

        return loss