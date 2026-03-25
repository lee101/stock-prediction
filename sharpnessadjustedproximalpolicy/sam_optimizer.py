"""Sharpness-Adjusted Proximal optimizer.

Core idea: measure loss-landscape sharpness periodically via random weight
perturbation.  Use sharpness signal to modulate weight decay (not lr):
- Sharp minimum (likely overfit) -> increase weight decay to escape
- Flat minimum (good generalization) -> decrease weight decay to stay

This avoids the numerical instability of lr scaling with Sortino loss.

Speed: sharpness is measured every ``probe_every`` optimiser steps using a
single random perturbation direction -- amortised overhead is ~1/probe_every
extra forward passes (e.g. probe_every=10 => 10% overhead).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class SharpnessState:
    raw: float = 0.0      # latest |L(theta+eps) - L(theta)| / rho
    ema: float = 0.0       # exponential moving average
    lr_scale: float = 1.0  # kept for backwards compat / logging
    wd_scale: float = 1.0  # weight decay multiplier
    step: int = 0


class SharpnessAdjustedOptimizer:
    """Wraps a base optimizer with periodic sharpness probing.

    Instead of scaling lr (which causes NaN with Sortino loss), we scale
    weight decay: high sharpness -> higher wd (push toward flatter minima),
    low sharpness -> lower wd (stay in flat basin).
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        *,
        rho: float = 0.05,
        probe_every: int = 10,
        ema_beta: float = 0.9,
        target_sharpness: float = 1.0,
        min_scale: float = 0.5,
        max_scale: float = 1.5,
        scale_mode: str = "linear",
        warmup_probes: int = 50,
        wd_min_scale: float = 0.5,
        wd_max_scale: float = 2.0,
    ):
        self.base = base_optimizer
        self.rho = rho
        self.probe_every = max(1, probe_every)
        self.ema_beta = ema_beta
        self.target_sharpness = max(target_sharpness, 1e-8)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.wd_min_scale = wd_min_scale
        self.wd_max_scale = wd_max_scale
        self.scale_mode = scale_mode
        self.warmup_probes = warmup_probes
        self._probe_count = 0
        self.state = SharpnessState()
        self._base_lrs = [g["lr"] for g in self.param_groups]
        self._base_wds = [g.get("weight_decay", 0.0) for g in self.param_groups]
        self._saved_params: list[torch.Tensor] = []

    @property
    def param_groups(self):
        return self.base.param_groups

    def zero_grad(self, set_to_none: bool = False):
        self.base.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable | None = None):
        self.base.step(closure=closure)
        self.state.step += 1

    def update_base_lrs(self):
        """Sync base_lrs from current param_groups (call after external lr schedule)."""
        self._base_lrs = [g["lr"] for g in self.param_groups]

    def should_probe(self) -> bool:
        return self.state.step % self.probe_every == 0

    @torch.no_grad()
    def _save_params(self):
        self._saved_params = []
        for group in self.param_groups:
            for p in group["params"]:
                self._saved_params.append(p.data.clone())

    @torch.no_grad()
    def _restore_params(self):
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                p.data.copy_(self._saved_params[idx])
                idx += 1
        self._saved_params = []

    @torch.no_grad()
    def _perturb_params(self) -> float:
        """Add random perturbation of norm rho."""
        total_norm_sq = 0.0
        perturbations = []
        for group in self.param_groups:
            for p in group["params"]:
                eps = torch.randn_like(p)
                perturbations.append(eps)
                total_norm_sq += eps.norm().item() ** 2

        scale = self.rho / max(math.sqrt(total_norm_sq), 1e-12)
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                p.data.add_(perturbations[idx], alpha=scale)
                idx += 1
        return self.rho

    def probe_sharpness(self, loss_fn: Callable[[], torch.Tensor], base_loss: float) -> float:
        """Measure sharpness and adjust weight decay accordingly."""
        if math.isnan(base_loss) or math.isinf(base_loss):
            return self.state.raw

        self._save_params()
        self._perturb_params()
        with torch.inference_mode():
            perturbed_loss = loss_fn().item()
        self._restore_params()

        if math.isnan(perturbed_loss) or math.isinf(perturbed_loss):
            return self.state.raw

        sharpness = abs(perturbed_loss - base_loss) / max(self.rho, 1e-12)
        if math.isnan(sharpness) or math.isinf(sharpness):
            return self.state.raw

        self._probe_count += 1
        self.state.raw = sharpness
        self.state.ema = self.ema_beta * self.state.ema + (1 - self.ema_beta) * sharpness

        if self._probe_count <= self.warmup_probes:
            if self._probe_count == self.warmup_probes:
                self.target_sharpness = max(self.state.ema, 1e-8)
        else:
            self._apply_wd_scaling()
        return sharpness

    def _apply_wd_scaling(self):
        """Scale weight decay based on sharpness: sharp -> more wd, flat -> less wd."""
        if self.state.ema <= 0 or math.isnan(self.state.ema):
            return
        ratio = self.state.ema / self.target_sharpness
        # sharp (ratio>1) -> higher wd to regularize, flat (ratio<1) -> lower wd to stay
        wd_scale = max(self.wd_min_scale, min(self.wd_max_scale, ratio))
        # smooth adjustment
        prev = self.state.wd_scale
        delta = wd_scale - prev
        max_delta = 0.05 * prev  # max 5% change per probe
        delta = max(-max_delta, min(max_delta, delta))
        self.state.wd_scale = prev + delta
        self.state.lr_scale = self.state.wd_scale  # for logging compat

        for group, base_wd in zip(self.param_groups, self._base_wds):
            if base_wd > 0:
                group["weight_decay"] = base_wd * self.state.wd_scale

    def _compute_scale(self, sharpness: float) -> float:
        """Compute raw scale ratio (used by wd scaling internally)."""
        if math.isnan(sharpness) or sharpness <= 0:
            return 1.0
        ratio = sharpness / self.target_sharpness
        if self.scale_mode == "log":
            return 1.0 + math.log(max(ratio, 1e-8))
        return ratio


class SWAWrapper:
    """Stochastic Weight Averaging: maintain EMA of weights during training.

    After warmup_steps, begins averaging weights. Call get_swa_model() at end
    to get the averaged weights for evaluation. Zero overhead except memory.
    """

    def __init__(self, model: torch.nn.Module, *, swa_start_frac: float = 0.5, swa_lr: float | None = None):
        self.model = model
        self.swa_start_frac = swa_start_frac
        self.swa_lr = swa_lr
        self._avg_params: dict[str, torch.Tensor] = {}
        self._n_averaged = 0
        self._started = False

    @torch.no_grad()
    def update(self, step: int, total_steps: int):
        if step < int(total_steps * self.swa_start_frac):
            return
        if not self._started:
            self._started = True
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    self._avg_params[name] = p.data.clone()
            self._n_averaged = 1
            return

        self._n_averaged += 1
        alpha = 1.0 / self._n_averaged
        for name, p in self.model.named_parameters():
            if name in self._avg_params:
                self._avg_params[name].lerp_(p.data, alpha)

    @torch.no_grad()
    def apply_swa_weights(self):
        """Replace model weights with SWA-averaged weights."""
        if not self._avg_params:
            return
        for name, p in self.model.named_parameters():
            if name in self._avg_params:
                p.data.copy_(self._avg_params[name])

    @property
    def n_averaged(self) -> int:
        return self._n_averaged


class GradientNoiseInjector:
    """Add gradient noise scaled by sharpness.

    When landscape is sharp (overfit), inject more noise into gradients to
    help escape. When flat, inject less. Based on the observation that
    gradient noise is equivalent to implicit regularization.

    sigma = base_sigma * (sharpness / target_sharpness)
    """

    def __init__(self, *, base_sigma: float = 0.01, gamma: float = 0.55, target_sharpness: float = 1.0):
        self.base_sigma = base_sigma
        self.gamma = gamma  # noise decay exponent (Neelakantan et al. 2015)
        self.target_sharpness = max(target_sharpness, 1e-8)

    @torch.no_grad()
    def inject(self, model: torch.nn.Module, step: int, sharpness_ema: float):
        if sharpness_ema <= 0 or math.isnan(sharpness_ema):
            return
        ratio = sharpness_ema / self.target_sharpness
        # anneal noise over training, scale by sharpness
        sigma = self.base_sigma * ratio / (1 + step) ** self.gamma
        for p in model.parameters():
            if p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad), alpha=sigma)


class FullSAMOptimizer:
    """Full SAM (Foret et al. 2020) with asymmetric rho.

    Every step:
    1) Compute grad at theta
    2) Ascend: theta' = theta + rho * grad / ||grad||
    3) Compute grad at theta' (the SAM gradient)
    4) Descend using SAM gradient with base optimizer

    rho adapts based on sharpness: sharper = larger perturbation to find
    better escape directions.

    2x compute per step.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        *,
        rho: float = 0.05,
        adaptive: bool = True,
        ema_beta: float = 0.9,
        target_sharpness: float = 1.0,
        rho_min: float = 0.01,
        rho_max: float = 0.2,
    ):
        self.base = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.ema_beta = ema_beta
        self.target_sharpness = max(target_sharpness, 1e-8)
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.state = SharpnessState()
        self._saved_params: list[torch.Tensor] = []
        self._base_lrs = [g["lr"] for g in self.param_groups]

    @property
    def param_groups(self):
        return self.base.param_groups

    def zero_grad(self, set_to_none: bool = False):
        self.base.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def _ascend(self):
        self._saved_params = []
        grad_norm_sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                self._saved_params.append(p.data.clone())
                if p.grad is not None:
                    if self.adaptive:
                        grad_norm_sq += (p.grad * p.abs().clamp(min=1e-12)).norm().item() ** 2
                    else:
                        grad_norm_sq += p.grad.norm().item() ** 2

        eff_rho = self._effective_rho()
        scale = eff_rho / max(math.sqrt(grad_norm_sq), 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if self.adaptive:
                    eps = p.grad * p.abs().clamp(min=1e-12) * scale
                else:
                    eps = p.grad * scale
                p.data.add_(eps)

    @torch.no_grad()
    def _descend(self):
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                p.data.copy_(self._saved_params[idx])
                idx += 1
        self._saved_params = []

    def _effective_rho(self) -> float:
        if self.state.step == 0:
            return self.rho
        ratio = self.state.ema / self.target_sharpness
        eff = self.rho * max(ratio, 0.1)
        return max(self.rho_min, min(self.rho_max, eff))

    def step_with_sam(self, loss_fn: Callable[[], torch.Tensor]) -> tuple[float, float]:
        """Full SAM step. Returns (loss, sharpness)."""
        self._ascend()
        self.base.zero_grad(set_to_none=True)
        perturbed_loss = loss_fn()
        perturbed_loss.backward()
        perturbed_val = perturbed_loss.item()
        self._descend()
        self.base.step()

        sharpness = abs(perturbed_val) / max(self._effective_rho(), 1e-12)
        if not (math.isnan(sharpness) or math.isinf(sharpness)):
            self.state.raw = sharpness
            self.state.ema = self.ema_beta * self.state.ema + (1 - self.ema_beta) * sharpness
        self.state.step += 1
        return perturbed_val, self.state.raw

    def update_base_lrs(self):
        self._base_lrs = [g["lr"] for g in self.param_groups]


class LookSAMOptimizer:
    """LookSAM: reuse the SAM ascent direction for K steps.

    Full SAM every ``sam_every`` steps; for intermediate steps, project
    the current gradient onto the saved SAM direction and mix.
    Overhead: ~1 + 1/sam_every extra forward passes per step.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        *,
        rho: float = 0.05,
        sam_every: int = 5,
        alpha: float = 0.5,
    ):
        self.base = base_optimizer
        self.rho = rho
        self.sam_every = max(1, sam_every)
        self.alpha = alpha
        self.state = SharpnessState()
        self._sam_direction: dict[int, torch.Tensor] = {}

    @property
    def param_groups(self):
        return self.base.param_groups

    def zero_grad(self, set_to_none: bool = False):
        self.base.zero_grad(set_to_none=set_to_none)

    def step(self, loss_fn: Callable[[], torch.Tensor] | None = None):
        do_sam = (self.state.step % self.sam_every == 0) and loss_fn is not None
        if do_sam:
            self._full_sam_step(loss_fn)
        else:
            self._projected_step()
        self.state.step += 1

    @torch.no_grad()
    def _full_sam_step(self, loss_fn: Callable[[], torch.Tensor]):
        saved = []
        grad_norm_sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                saved.append(p.data.clone())
                if p.grad is not None:
                    grad_norm_sq += p.grad.norm().item() ** 2

        scale = self.rho / max(math.sqrt(grad_norm_sq), 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.data.add_(p.grad, alpha=scale)

        with torch.enable_grad():
            self.base.zero_grad(set_to_none=True)
            loss_p = loss_fn()
            loss_p.backward()

        idx = 0
        self._sam_direction = {}
        for group in self.param_groups:
            for p in group["params"]:
                p.data.copy_(saved[idx])
                if p.grad is not None:
                    self._sam_direction[id(p)] = p.grad.clone()
                idx += 1

        self.base.step()

    @torch.no_grad()
    def _projected_step(self):
        if not self._sam_direction:
            self.base.step()
            return
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                sam_d = self._sam_direction.get(id(p))
                if sam_d is None:
                    continue
                proj = (p.grad * sam_d).sum() / max(sam_d.norm().item() ** 2, 1e-12)
                p.grad.add_(sam_d, alpha=self.alpha * proj.item())
        self.base.step()
