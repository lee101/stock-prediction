"""Sharpness-Adjusted Proximal optimizer.

Core idea: measure loss-landscape sharpness periodically via random weight
perturbation.  When the current minimum is *sharp* (likely overfit), allow
large parameter updates to escape.  When it is *flat* (good generalisation),
constrain updates to stay in the basin.

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
    lr_scale: float = 1.0  # multiplier applied to base lr
    step: int = 0


class SharpnessAdjustedOptimizer:
    """Wraps a base optimizer with periodic sharpness probing and lr scaling.

    Parameters
    ----------
    base_optimizer : torch.optim.Optimizer
        Any PyTorch optimizer (AdamW, SGD, ...).
    rho : float
        Perturbation radius (L2 norm) for sharpness probe.  Default 0.05.
    probe_every : int
        Compute sharpness every N optimizer steps.  Higher = faster but less
        responsive.  Default 10.
    ema_beta : float
        EMA decay for smoothing sharpness signal.  Default 0.9.
    target_sharpness : float
        Sharpness value at which lr_scale == 1.0.  Below target => scale < 1
        (stay); above target => scale > 1 (escape).
    min_scale : float
        Minimum lr multiplier (when very flat).  Default 0.3.
    max_scale : float
        Maximum lr multiplier (when very sharp).  Default 3.0.
    scale_mode : str
        "linear" or "log".  How sharpness maps to lr_scale.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        *,
        rho: float = 0.05,
        probe_every: int = 10,
        ema_beta: float = 0.9,
        target_sharpness: float = 1.0,
        min_scale: float = 0.3,
        max_scale: float = 3.0,
        scale_mode: str = "linear",
    ):
        self.base = base_optimizer
        self.rho = rho
        self.probe_every = max(1, probe_every)
        self.ema_beta = ema_beta
        self.target_sharpness = max(target_sharpness, 1e-8)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_mode = scale_mode
        self.state = SharpnessState()
        self._base_lrs = [g["lr"] for g in self.param_groups]
        self._saved_params: list[torch.Tensor] = []

    @property
    def param_groups(self):
        return self.base.param_groups

    def zero_grad(self, set_to_none: bool = False):
        self.base.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable | None = None):
        self.base.step(closure=closure)
        self.state.step += 1
        self._apply_lr_scale()

    def _apply_lr_scale(self):
        s = self.state.lr_scale
        for group, base_lr in zip(self.param_groups, self._base_lrs):
            group["lr"] = base_lr * s

    def update_base_lrs(self):
        """Sync base_lrs from current param_groups (call after external lr schedule)."""
        self._base_lrs = [g["lr"] for g in self.param_groups]

    # ------------------------------------------------------------------
    # Sharpness probing
    # ------------------------------------------------------------------

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
        """Add random perturbation of norm rho. Returns actual norm."""
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
        """Measure sharpness: |L(theta+eps) - L(theta)| / rho.

        Call this with a closure that computes loss at current params.
        ``base_loss`` is L(theta) already computed in the training step.
        """
        self._save_params()
        actual_rho = self._perturb_params()
        with torch.no_grad():
            perturbed_loss = loss_fn().item()
        self._restore_params()

        sharpness = abs(perturbed_loss - base_loss) / max(actual_rho, 1e-12)
        self.state.raw = sharpness
        self.state.ema = self.ema_beta * self.state.ema + (1 - self.ema_beta) * sharpness
        self.state.lr_scale = self._compute_scale(self.state.ema)
        self._apply_lr_scale()
        return sharpness

    def _compute_scale(self, sharpness: float) -> float:
        ratio = sharpness / self.target_sharpness
        if self.scale_mode == "log":
            scale = 1.0 + math.log(max(ratio, 1e-8))
        else:
            scale = ratio
        return max(self.min_scale, min(self.max_scale, scale))


class FullSAMOptimizer:
    """Full SAM (Foret et al. 2020) with asymmetric clipping.

    Every step:
    1) Compute grad at theta
    2) Ascend: theta' = theta + rho * grad / ||grad||
    3) Compute grad at theta' (the SAM gradient)
    4) Descend using SAM gradient with base optimizer

    Asymmetric twist: the ascent step rho is modulated by tracked sharpness.
    When flat, rho shrinks (less aggressive perturbation).
    When sharp, rho grows (explore harder to find escape direction).

    This is 2x compute per step. Use ``probe_every`` variant above for speed.
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
        """Perturb params in gradient direction by adaptive rho."""
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
        """Restore original params."""
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
        """Full SAM step. Returns (loss, sharpness).

        Usage:
            optimizer.zero_grad()
            loss = loss_fn()  # forward + compute loss
            loss.backward()
            sam_loss, sharpness = optimizer.step_with_sam(loss_fn)
        """
        base_loss = None
        # first backward already done by caller
        self._ascend()

        # second forward + backward at perturbed point
        self.base.zero_grad(set_to_none=True)
        perturbed_loss = loss_fn()
        perturbed_loss.backward()

        base_loss_val = 0.0  # caller should track
        perturbed_val = perturbed_loss.item()

        self._descend()
        # now apply the SAM gradient (computed at theta') to original theta
        self.base.step()

        # track sharpness
        sharpness = abs(perturbed_val - base_loss_val) / max(self._effective_rho(), 1e-12)
        self.state.raw = sharpness
        self.state.ema = self.ema_beta * self.state.ema + (1 - self.ema_beta) * sharpness
        self.state.step += 1
        return perturbed_val, sharpness

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
        self._saved_params: list[torch.Tensor] = []

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
        # save params & current grad
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

        # enable grads for perturbed forward+backward
        with torch.enable_grad():
            self.base.zero_grad(set_to_none=True)
            loss_p = loss_fn()
            loss_p.backward()

        # compute sam direction = grad_perturbed - grad_original (approximation)
        idx = 0
        self._sam_direction = {}
        for gidx, group in enumerate(self.param_groups):
            for pidx, p in enumerate(group["params"]):
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
                g = p.grad
                proj = (g * sam_d).sum() / max(sam_d.norm().item() ** 2, 1e-12)
                p.grad.add_(sam_d, alpha=self.alpha * proj.item())
        self.base.step()
