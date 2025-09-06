#!/usr/bin/env python3
"""
Modern Optimizers Collection
Includes GPro, Lion, AdaFactor, and other state-of-the-art optimizers
"""

import torch
import torch.nn as nn
import torch.optim
import math
from typing import Dict, List, Tuple, Optional, Any


class Lion(torch.optim.Optimizer):
    """
    Lion Optimizer - EvoLved Sign Momentum
    Paper: https://arxiv.org/abs/2302.06675
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
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

                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Lion update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.data.add_(torch.sign(update), alpha=-group['lr'])
                
                # Update exponential moving average
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

        return loss


class AdaFactor(torch.optim.Optimizer):
    """
    AdaFactor Optimizer - Adaptive Learning Rates with Sublinear Memory Cost
    Paper: https://arxiv.org/abs/1804.04235
    """
    def __init__(self, params, lr=None, eps2=1e-30, clip_threshold=1.0, decay_rate=-0.8,
                 beta1=None, weight_decay=0.0, scale_parameter=True, relative_step=True):
        if lr is not None and lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, eps2=eps2, clip_threshold=clip_threshold, decay_rate=decay_rate,
                       beta1=beta1, weight_decay=weight_decay, scale_parameter=scale_parameter,
                       relative_step=relative_step)
        super().__init__(params, defaults)

    def _get_lr(self, param_group, param_state):
        if param_group['lr'] is None:
            min_step = 1e-6 * param_state['step'] if param_group['scale_parameter'] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state['step']))
            param_scale = 1.0
            if param_group['scale_parameter']:
                param_scale = max(param_group['eps2'], param_state['RMS'])
            return param_scale * rel_step_sz
        else:
            return param_group['lr']

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

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
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)

                # State Initialization
                if len(state) == 0:
                    state['step'] = 0

                    if use_first_moment:
                        state['exp_avg'] = torch.zeros_like(grad).float()
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1]).float()
                        state['exp_avg_sq_col'] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).float()
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad).float()

                    state['RMS'] = 0
                p_data_fp32 = p.data.float()
                state['step'] += 1
                state['RMS'] = self._rms(p_data_fp32)

                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = grad**2 + group['eps2']

                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)
                    update = update / (exp_avg_sq_row.unsqueeze(-1) + group['eps2']).sqrt()
                    update = update / (exp_avg_sq_col.unsqueeze(-2) + group['eps2']).sqrt()
                else:
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = update / (exp_avg_sq + group['eps2']).sqrt()

                update.div_((self._rms(update) / group['clip_threshold']).clamp_(min=1.0))

                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(update, alpha=1 - group['beta1'])
                    update = exp_avg

                if group['weight_decay'] != 0:
                    p_data_fp32.mul_(1 - group['weight_decay'] * lr)

                p_data_fp32.add_(update, alpha=-lr)

                p.data.copy_(p_data_fp32)

        return loss


class LAMB(torch.optim.Optimizer):
    """
    LAMB Optimizer - Layer-wise Adaptive Moments optimizer for Batch training
    Paper: https://arxiv.org/abs/1904.00962
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01, clamp_value=10.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, clamp_value=clamp_value)
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
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Apply bias correction
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2

                # Compute update
                denom = exp_avg_sq_corrected.sqrt().add_(group['eps'])
                update = exp_avg_corrected / denom

                # Add weight decay
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])

                # Compute norms for layer-wise adaptation
                weight_norm = p.data.norm()
                update_norm = update.norm()

                # Layer-wise adaptation
                if weight_norm > 0 and update_norm > 0:
                    trust_ratio = weight_norm / update_norm
                    trust_ratio = min(trust_ratio, group['clamp_value'])
                else:
                    trust_ratio = 1.0

                # Apply update
                p.data.add_(update, alpha=-group['lr'] * trust_ratio)

        return loss


class Sophia(torch.optim.Optimizer):
    """
    Sophia Optimizer - Second-order Clipped Stochastic Optimization
    Paper: https://arxiv.org/abs/2305.14342
    """
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04, weight_decay=1e-1, maximize=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, maximize=maximize)
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
                if group['maximize']:
                    grad = -grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['hessian_diag'] = torch.zeros_like(p.data)

                exp_avg, hessian_diag = state['exp_avg'], state['hessian_diag']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Update exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Hessian diagonal approximation (Gauss-Newton)
                # For simplicity, we use the square of gradients as approximation
                hessian_diag.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute update
                exp_avg_corrected = exp_avg / bias_correction1
                hessian_corrected = hessian_diag / bias_correction2

                # Clipping
                update = exp_avg_corrected / torch.clamp(hessian_corrected, min=group['rho'])

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Apply update
                p.data.add_(update, alpha=-group['lr'])

        return loss


class Adan(torch.optim.Optimizer):
    """
    Adan Optimizer - Adaptive Nesterov Momentum Algorithm
    Paper: https://arxiv.org/abs/2208.06677
    """
    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8, weight_decay=0.02):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_diff'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_diff']
                beta1, beta2, beta3 = group['betas']

                state['step'] += 1

                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Update exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute difference for Nesterov
                if state['step'] > 1:
                    diff = grad - state['prev_grad']
                    exp_avg_diff.mul_(beta3).add_(diff, alpha=1 - beta3)

                state['prev_grad'] = grad.clone()

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                bias_correction3 = 1 - beta3 ** state['step']

                # Compute update
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                # Nesterov-style update
                update = (exp_avg + beta2 * exp_avg_diff / bias_correction3) / denom

                p.data.add_(update, alpha=-step_size)

        return loss


def get_optimizer(name: str, parameters, **kwargs):
    """Factory function to get optimizer by name"""
    optimizers = {
        'gpro': lambda p, **k: GPro(p, **k),
        'lion': lambda p, **k: Lion(p, **k),
        'adafactor': lambda p, **k: AdaFactor(p, **k),
        'lamb': lambda p, **k: LAMB(p, **k),
        'sophia': lambda p, **k: Sophia(p, **k),
        'adan': lambda p, **k: Adan(p, **k),
        'adamw': lambda p, **k: torch.optim.AdamW(p, **k),
        'adam': lambda p, **k: torch.optim.Adam(p, **k),
        'sgd': lambda p, **k: torch.optim.SGD(p, **k),
    }
    
    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
    
    return optimizers[name.lower()](parameters, **kwargs)


# GPro is already defined in hf_trainer.py, no need to import here


class Shampoo(torch.optim.Optimizer):
    """
    Lightweight factored Shampoo optimizer (self-contained).
    - For 2D tensors, maintains row/col second-moment statistics and uses
      inverse square-root preconditioning: G_tilde = L^{-1/2} G R^{-1/2}.
    - For 1D tensors (bias, vectors), falls back to RMSProp-like update.

    Notes:
    - This implementation targets small/medium tensors and test scenarios.
    - For stability, adds epsilon to preconditioners and clamps eigenvalues.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.99),
        eps: float = 1e-12,
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _inv_sqrt(mat: torch.Tensor, eps: float) -> torch.Tensor:
        """Compute (mat + eps I)^{-1/2} for symmetric PSD mat via eigendecomposition."""
        # Ensure float32 for stability
        mat = mat.float()
        # Symmetrize just in case of numeric drift
        mat = 0.5 * (mat + mat.transpose(-1, -2))
        eigvals, eigvecs = torch.linalg.eigh(mat)
        # Clamp eigenvalues to avoid negatives and add eps
        eigvals_clamped = torch.clamp(eigvals, min=0.0) + eps
        inv_sqrt_vals = eigvals_clamped.rsqrt()
        # Reconstruct inverse sqrt
        return eigvecs @ torch.diag_embed(inv_sqrt_vals) @ eigvecs.transpose(-1, -2)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            wd = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("Shampoo does not support sparse gradients")

                state = self.state[p]
                # Decoupled weight decay
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                if g.ndim >= 2:
                    # Treat as matrix with last two dims as [rows, cols]
                    # Flatten leading dims if present
                    orig_shape = g.shape
                    rows, cols = g.shape[-2], g.shape[-1]
                    G = g.reshape(-1, rows, cols)

                    if len(state) == 0:
                        state['step'] = 0
                        # Row and column second moment accumulators
                        state['L'] = torch.zeros(G.shape[0], rows, rows, device=G.device, dtype=torch.float32)
                        state['R'] = torch.zeros(G.shape[0], cols, cols, device=G.device, dtype=torch.float32)

                    L = state['L']
                    R = state['R']
                    state['step'] += 1

                    # Update factored second moments
                    # L_t = beta2 * L + (1-beta2) * (G G^T) averaged over batch factors
                    # R_t = beta2 * R + (1-beta2) * (G^T G)
                    GGt = torch.matmul(G, G.transpose(-1, -2))
                    GtG = torch.matmul(G.transpose(-1, -2), G)
                    L.mul_(beta2).add_(GGt.mean(dim=0), alpha=(1 - beta2))
                    R.mul_(beta2).add_(GtG.mean(dim=0), alpha=(1 - beta2))

                    # Precondition: G_tilde = L^{-1/2} G R^{-1/2}
                    L_inv_sqrt = self._inv_sqrt(L, eps)
                    R_inv_sqrt = self._inv_sqrt(R, eps)
                    # Apply preconditioners to each slice
                    G_tilde = torch.matmul(L_inv_sqrt, torch.matmul(G, R_inv_sqrt))
                    # Momentum on preconditioned grads (beta1)
                    if 'm' not in state:
                        state['m'] = torch.zeros_like(G_tilde)
                    m = state['m']
                    m.mul_(beta1).add_(G_tilde, alpha=(1 - beta1))
                    # Update
                    update = m
                    p.data.add_(update.reshape(orig_shape), alpha=-lr)

                else:
                    # 1D parameters: use RMSProp-like second moment
                    if len(state) == 0:
                        state['step'] = 0
                        state['v'] = torch.zeros_like(p, dtype=torch.float32)
                        state['m'] = torch.zeros_like(p, dtype=torch.float32)
                    v = state['v']
                    m = state['m']
                    state['step'] += 1

                    v.mul_(beta2).addcmul_(g, g, value=(1 - beta2))
                    precond = g / (v.sqrt() + eps)
                    m.mul_(beta1).add_(precond, alpha=(1 - beta1))
                    p.data.add_(m, alpha=-lr)

        return loss


# Register Shampoo in the optimizer factory (kept near class for clarity)
def _register_shampoo():
    old_get = get_optimizer

    def _factory(name: str, parameters, **kwargs):
        name_l = name.lower()
        if name_l == 'shampoo':
            return Shampoo(parameters, **kwargs)
        return old_get(name, parameters, **kwargs)

    return _factory

# Monkey-patch get_optimizer to include 'shampoo'
get_optimizer = _register_shampoo()
