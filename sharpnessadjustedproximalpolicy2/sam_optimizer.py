"""Directional sharpness-aware proximal optimizers.

The primary optimizer in this module measures *directional curvature* around the
current weights and the optimizer's candidate destination, then rescales the
actual parameter step with a proximal pullback:

- moving from sharp -> flatter regions keeps or slightly expands the step
- moving from flat -> sharper regions shrinks the step
- the signal is computed around both the source and candidate points

That is intentionally different from the original package, which only gated
weight decay from a one-sided random probe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

LossFn = Callable[[], torch.Tensor]


@dataclass
class SharpnessState:
    raw: float = 0.0
    ema: float = 0.0
    step_scale: float = 1.0
    wd_scale: float = 1.0
    step: int = 0
    source_raw: float = 0.0
    candidate_loss: float = 0.0
    loss_delta: float = 0.0

    @property
    def lr_scale(self) -> float:
        return self.step_scale

    @lr_scale.setter
    def lr_scale(self, value: float) -> None:
        self.step_scale = value


class DirectionalSharpnessProximalOptimizer:
    """Wrap a base optimizer with directional sharpness-aware proximal control.

    Each step starts with the base optimizer's candidate update. On probe steps,
    the optimizer estimates directional curvature at the current point and the
    candidate point along the *actual update direction* using a symmetric finite
    difference. It then rescales the final parameter move toward the source
    weights whenever the destination looks sharper or the loss worsens.

    Between probe steps, the last measured ``step_scale`` is reused so the
    proximal control still applies on every update.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        *,
        rho: float = 0.05,
        probe_every: int = 10,
        ema_beta: float = 0.9,
        target_sharpness: float = 1.0,
        min_scale: float = 0.35,
        max_scale: float = 1.15,
        warmup_probes: int = 8,
        scale_beta: float = 0.6,
        flat_bonus: float = 0.15,
        sharp_penalty: float = 1.0,
        loss_penalty: float = 0.5,
    ):
        self.base = base_optimizer
        self.rho = max(float(rho), 1e-8)
        self.probe_every = max(1, int(probe_every))
        self.ema_beta = float(ema_beta)
        self.target_sharpness = max(float(target_sharpness), 1e-8)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.warmup_probes = max(0, int(warmup_probes))
        self.scale_beta = float(scale_beta)
        self.flat_bonus = float(flat_bonus)
        self.sharp_penalty = float(sharp_penalty)
        self.loss_penalty = float(loss_penalty)
        self._probe_count = 0
        self.state = SharpnessState()
        self._base_lrs = [g["lr"] for g in self.param_groups]

    @property
    def param_groups(self):
        return self.base.param_groups

    def zero_grad(self, set_to_none: bool = False):
        self.base.zero_grad(set_to_none=set_to_none)

    def update_base_lrs(self):
        self._base_lrs = [g["lr"] for g in self.param_groups]

    def should_probe(self) -> bool:
        return self.state.step % self.probe_every == 0

    @torch.no_grad()
    def _clone_params(self) -> list[torch.Tensor]:
        saved: list[torch.Tensor] = []
        for group in self.param_groups:
            for p in group["params"]:
                saved.append(p.data.clone())
        return saved

    @torch.no_grad()
    def _restore_params(self, saved_params: list[torch.Tensor]) -> None:
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                p.data.copy_(saved_params[idx])
                idx += 1

    @torch.no_grad()
    def _apply_direction(self, directions: list[torch.Tensor], alpha: float) -> None:
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                p.data.add_(directions[idx], alpha=alpha)
                idx += 1

    @torch.no_grad()
    def _build_step_direction(self, source_params: list[torch.Tensor]) -> tuple[list[torch.Tensor], float]:
        deltas: list[torch.Tensor] = []
        total_norm_sq = 0.0
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                delta = p.data - source_params[idx]
                deltas.append(delta.clone())
                total_norm_sq += delta.norm().item() ** 2
                idx += 1

        step_norm = math.sqrt(total_norm_sq)
        if step_norm <= 1e-12:
            return deltas, 0.0

        inv_norm = 1.0 / step_norm
        for delta in deltas:
            delta.mul_(inv_norm)
        return deltas, step_norm

    @torch.no_grad()
    def _evaluate_loss(self, loss_fn: LossFn) -> float:
        value = loss_fn().item()
        if math.isnan(value) or math.isinf(value):
            return float("nan")
        return value

    @torch.no_grad()
    def _directional_curvature(
        self,
        loss_fn: LossFn,
        base_params: list[torch.Tensor],
        base_loss: float,
        directions: list[torch.Tensor],
    ) -> float:
        if math.isnan(base_loss) or math.isinf(base_loss):
            return float("nan")

        self._restore_params(base_params)
        self._apply_direction(directions, self.rho)
        plus_loss = self._evaluate_loss(loss_fn)

        self._restore_params(base_params)
        self._apply_direction(directions, -self.rho)
        minus_loss = self._evaluate_loss(loss_fn)

        self._restore_params(base_params)

        if any(math.isnan(v) or math.isinf(v) for v in (plus_loss, minus_loss)):
            return float("nan")

        return abs(plus_loss + minus_loss - 2.0 * base_loss) / max(self.rho * self.rho, 1e-12)

    def _clamp_scale(self, scale: float) -> float:
        if math.isnan(scale) or math.isinf(scale):
            return self.state.step_scale
        return max(self.min_scale, min(self.max_scale, scale))

    def _smooth_scale(self, target_scale: float) -> float:
        target_scale = self._clamp_scale(target_scale)
        if self._probe_count <= 1:
            return target_scale
        prev = self.state.step_scale
        mixed = self.scale_beta * prev + (1.0 - self.scale_beta) * target_scale
        return self._clamp_scale(mixed)

    def _compute_target_scale(
        self,
        *,
        source_sharpness: float,
        candidate_sharpness: float,
        base_loss: float,
        candidate_loss: float,
    ) -> float:
        sharp_norm = max(self.target_sharpness, 1e-8)
        flat_gain = max(source_sharpness - candidate_sharpness, 0.0) / sharp_norm
        sharp_gain = max(candidate_sharpness - source_sharpness, 0.0) / sharp_norm

        loss_norm = max(abs(base_loss), abs(candidate_loss), 1.0)
        loss_worsening = max(candidate_loss - base_loss, 0.0) / loss_norm

        scale = (1.0 + self.flat_bonus * flat_gain) / (
            1.0 + self.sharp_penalty * sharp_gain + self.loss_penalty * loss_worsening
        )
        return self._clamp_scale(scale)

    @torch.no_grad()
    def _apply_step_scale(self, source_params: list[torch.Tensor], step_scale: float) -> None:
        step_scale = self._clamp_scale(step_scale)
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                source = source_params[idx]
                delta = p.data - source
                p.data.copy_(source)
                p.data.add_(delta, alpha=step_scale)
                idx += 1

    def step(
        self,
        closure: LossFn | None = None,
        *,
        loss_fn: LossFn | None = None,
        base_loss: float | None = None,
    ):
        if loss_fn is None:
            loss_fn = closure

        source_params = self._clone_params()
        self.base.step(closure=closure if loss_fn is None else None)
        self.state.step += 1

        if not source_params:
            return None

        if loss_fn is None or base_loss is None or math.isnan(base_loss) or math.isinf(base_loss):
            self._apply_step_scale(source_params, self.state.step_scale)
            return None

        if not self.should_probe():
            self._apply_step_scale(source_params, self.state.step_scale)
            return None

        candidate_params = self._clone_params()
        directions, step_norm = self._build_step_direction(source_params)
        if step_norm <= 1e-12:
            self._apply_step_scale(source_params, self.state.step_scale)
            return None

        candidate_loss = self._evaluate_loss(loss_fn)
        source_sharpness = self._directional_curvature(loss_fn, source_params, base_loss, directions)
        candidate_sharpness = self._directional_curvature(loss_fn, candidate_params, candidate_loss, directions)

        self._restore_params(candidate_params)

        if any(math.isnan(v) or math.isinf(v) for v in (candidate_loss, source_sharpness, candidate_sharpness)):
            self._apply_step_scale(source_params, self.state.step_scale)
            return candidate_loss

        self._probe_count += 1
        self.state.source_raw = source_sharpness
        self.state.raw = candidate_sharpness
        self.state.candidate_loss = candidate_loss
        self.state.loss_delta = candidate_loss - base_loss
        self.state.ema = self.ema_beta * self.state.ema + (1.0 - self.ema_beta) * candidate_sharpness

        target_scale = 1.0
        if self._probe_count <= self.warmup_probes:
            if self._probe_count == self.warmup_probes:
                self.target_sharpness = max(self.state.ema, 1e-8)
        else:
            target_scale = self._compute_target_scale(
                source_sharpness=source_sharpness,
                candidate_sharpness=candidate_sharpness,
                base_loss=base_loss,
                candidate_loss=candidate_loss,
            )

        self.state.step_scale = self._smooth_scale(target_scale)
        self._apply_step_scale(source_params, self.state.step_scale)
        return candidate_loss


class SWAWrapper:
    """Stochastic Weight Averaging: maintain EMA of weights during training."""

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
        if not self._avg_params:
            return
        for name, p in self.model.named_parameters():
            if name in self._avg_params:
                p.data.copy_(self._avg_params[name])

    @property
    def n_averaged(self) -> int:
        return self._n_averaged


class GradientNoiseInjector:
    """Add gradient noise scaled by sharpness EMA."""

    def __init__(self, *, base_sigma: float = 0.01, gamma: float = 0.55, target_sharpness: float = 1.0):
        self.base_sigma = base_sigma
        self.gamma = gamma
        self.target_sharpness = max(target_sharpness, 1e-8)

    @torch.no_grad()
    def inject(self, model: torch.nn.Module, step: int, sharpness_ema: float):
        if sharpness_ema <= 0 or math.isnan(sharpness_ema):
            return
        ratio = sharpness_ema / self.target_sharpness
        sigma = self.base_sigma * ratio / (1 + step) ** self.gamma
        for p in model.parameters():
            if p.grad is not None:
                p.grad.add_(torch.randn_like(p.grad), alpha=sigma)


class FullSAMOptimizer:
    """Full SAM (Foret et al. 2020) with corrected sharpness tracking."""

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
    def _ascend(self, eff_rho: float):
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

    def step_with_sam(self, loss_fn: LossFn, base_loss: float | None = None) -> tuple[float, float]:
        eff_rho = self._effective_rho()
        self._ascend(eff_rho)
        self.base.zero_grad(set_to_none=True)
        perturbed_loss = loss_fn()
        perturbed_loss.backward()
        perturbed_val = perturbed_loss.item()
        self._descend()
        self.base.step()

        if base_loss is not None and not (math.isnan(base_loss) or math.isinf(base_loss)):
            sharpness = abs(perturbed_val - base_loss) / max(eff_rho, 1e-12)
            if not (math.isnan(sharpness) or math.isinf(sharpness)):
                self.state.raw = sharpness
                self.state.ema = self.ema_beta * self.state.ema + (1 - self.ema_beta) * sharpness
        self.state.step += 1
        return perturbed_val, self.state.raw

    def update_base_lrs(self):
        self._base_lrs = [g["lr"] for g in self.param_groups]


class LookSAMOptimizer:
    """LookSAM: reuse the SAM ascent direction for K steps."""

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

    def step(self, loss_fn: LossFn | None = None):
        do_sam = (self.state.step % self.sam_every == 0) and loss_fn is not None
        if do_sam:
            self._full_sam_step(loss_fn)
        else:
            self._projected_step()
        self.state.step += 1

    @torch.no_grad()
    def _full_sam_step(self, loss_fn: LossFn):
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
