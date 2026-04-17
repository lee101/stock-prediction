"""Ensemble-Orthogonal PPO (EO-PPO) auxiliary loss.

Given a frozen v6 ensemble (N loaded trading policies) and a specialist
policy under training, returns a scalar loss term that pushes the
specialist's action distribution AWAY from the averaged ensemble
distribution at states where the ensemble is uncertain.

See ``docs/ensemble_orthogonal_ppo.md`` for the design rationale.

Minimal integration:

    from src.ensemble_orthogonal import EnsembleOrthogonalLoss

    # At train init (once):
    orth = EnsembleOrthogonalLoss.load(
        checkpoints=[...],          # list of .pt paths
        num_symbols=32,
        features_per_sym=...,
        device=device,
    )

    # In minibatch loop, after computing specialist log_probs:
    kl_term = orth.loss(specialist_logits=logits, obs=obs_minibatch,
                        beta=args.ensemble_kl_beta)
    total_loss = ppo_loss + kl_term

The loss is returned negative-of-weighted-KL so that standard
``total_loss.backward()`` minimisation increases KL from the ensemble.

Correctness invariants (covered by tests):
  1. ``beta == 0`` => loss is identically zero.
  2. Grad flows through specialist logits but NOT through ensemble.
  3. Per-sample KL is clamped to [0, kl_clamp] before weighting.
  4. Gate weight w(s) is bounded in [w_min, 1] when gate == 'entropy'.
  5. Ensemble forward runs under ``torch.no_grad()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


DEFAULT_W_MIN = 0.1
DEFAULT_KL_CLAMP = 5.0


@dataclass
class OrthogonalTermDiagnostics:
    """Per-call numbers, useful for tensorboard / json logging."""

    beta: float
    kl_mean: float
    kl_max: float
    w_mean: float
    weighted_kl: float


def compute_ensemble_log_probs(
    ensemble_models: Sequence[torch.nn.Module],
    obs: torch.Tensor,
    mode: str = "softmax_avg",
) -> torch.Tensor:
    """Return log-probs of the averaged ensemble on ``obs``.

    ``mode='softmax_avg'`` matches production ensemble inference
    (see ``scripts/screened32_realism_gate._build_ensemble_policy_fn``).
    """
    if not ensemble_models:
        raise ValueError("ensemble_models must be non-empty")
    if mode not in ("softmax_avg", "logit_avg"):
        raise ValueError(f"mode must be 'softmax_avg' or 'logit_avg', got {mode!r}")

    with torch.no_grad():
        if mode == "softmax_avg":
            probs_sum = None
            for m in ensemble_models:
                logits, _ = m(obs)
                p = F.softmax(logits.float(), dim=-1)
                probs_sum = p if probs_sum is None else probs_sum + p
            assert probs_sum is not None
            probs = probs_sum / float(len(ensemble_models))
            log_probs = torch.log(probs.clamp_min(1e-12))
        else:  # logit_avg
            logit_sum = None
            for m in ensemble_models:
                logits, _ = m(obs)
                logit_sum = logits.float() if logit_sum is None else logit_sum + logits.float()
            assert logit_sum is not None
            mean_logits = logit_sum / float(len(ensemble_models))
            log_probs = F.log_softmax(mean_logits, dim=-1)
    return log_probs.detach()


def kl_specialist_vs_ensemble(
    specialist_logits: torch.Tensor,
    ensemble_log_probs: torch.Tensor,
    kl_clamp: float = DEFAULT_KL_CLAMP,
) -> torch.Tensor:
    """Per-sample KL(specialist || ensemble) clamped to [0, kl_clamp]."""
    log_p_s = F.log_softmax(specialist_logits.float(), dim=-1)
    p_s = log_p_s.exp()
    kl = (p_s * (log_p_s - ensemble_log_probs)).sum(dim=-1)
    return kl.clamp(0.0, kl_clamp)


def ensemble_entropy_gate(
    ensemble_log_probs: torch.Tensor,
    w_min: float = DEFAULT_W_MIN,
) -> torch.Tensor:
    """Gate weight per sample in [w_min, 1], higher when ensemble is unsure.

    Normalised by the per-batch max entropy (scale-free).
    """
    p = ensemble_log_probs.exp()
    h = -(p * ensemble_log_probs).sum(dim=-1)
    h_max = h.max().clamp_min(1e-8)
    w = (h / h_max).clamp(min=w_min, max=1.0)
    return w.detach()


def orthogonal_loss(
    specialist_logits: torch.Tensor,
    ensemble_log_probs: torch.Tensor,
    beta: float,
    gate: str = "entropy",
    w_min: float = DEFAULT_W_MIN,
    kl_clamp: float = DEFAULT_KL_CLAMP,
) -> tuple[torch.Tensor, OrthogonalTermDiagnostics]:
    """Scalar loss term that MAXIMISES weighted KL divergence from ensemble.

    Returns ``(loss, diagnostics)``.  ``loss`` is already scaled by beta
    and negated; add it directly to the PPO total loss.

    When ``beta == 0`` the returned loss is a zero tensor that still
    tracks the specialist-logit grad graph so composition with PPO loss
    is a no-op.
    """
    if specialist_logits.shape != ensemble_log_probs.shape:
        raise ValueError(
            f"shape mismatch: specialist={tuple(specialist_logits.shape)} "
            f"ensemble={tuple(ensemble_log_probs.shape)}"
        )
    if gate not in ("entropy", "off"):
        raise ValueError(f"gate must be 'entropy' or 'off', got {gate!r}")
    if beta == 0.0:
        zero = (specialist_logits.float() * 0.0).sum()
        return zero, OrthogonalTermDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0)

    kl = kl_specialist_vs_ensemble(
        specialist_logits=specialist_logits,
        ensemble_log_probs=ensemble_log_probs,
        kl_clamp=kl_clamp,
    )
    if gate == "entropy":
        w = ensemble_entropy_gate(ensemble_log_probs, w_min=w_min)
    else:
        w = torch.ones_like(kl)

    weighted_kl = (w * kl).mean()
    loss = -float(beta) * weighted_kl

    with torch.no_grad():
        diag = OrthogonalTermDiagnostics(
            beta=float(beta),
            kl_mean=float(kl.mean().item()),
            kl_max=float(kl.max().item()),
            w_mean=float(w.mean().item()),
            weighted_kl=float(weighted_kl.item()),
        )
    return loss, diag


def beta_schedule(
    step: int,
    total_steps: int,
    peak_beta: float,
    warmup_frac: float = 0.3,
    start_frac: float = 0.0,
) -> float:
    """Linear ramp from 0 to peak_beta with optional vanilla-first phase.

    Phases (as fraction of total_steps):
      [0,           start_frac]           : beta = 0  (pure PPO)
      [start_frac,  start_frac+warmup_frac]: linear ramp 0 -> peak_beta
      [...        , 1.0]                   : beta = peak_beta

    With start_frac=0, warmup_frac=0.3 (default) this reproduces the original
    ramp.  With start_frac=0.4, warmup_frac=0.3 the policy trains vanilla for
    the first 40% of updates, then ramps to peak over 40%-70%, then holds.
    """
    if peak_beta == 0.0 or step <= 0 or total_steps <= 0:
        return 0.0
    frac = max(0.0, min(1.0, step / total_steps))
    if frac < start_frac:
        return 0.0
    if warmup_frac <= 0.0:
        return float(peak_beta)
    ramp = min(1.0, (frac - start_frac) / float(warmup_frac))
    return float(peak_beta) * float(ramp)


class EnsembleOrthogonalLoss:
    """Convenience wrapper: holds frozen ensemble and exposes .loss()."""

    def __init__(
        self,
        ensemble_models: Sequence[torch.nn.Module],
        mode: str = "softmax_avg",
        w_min: float = DEFAULT_W_MIN,
        kl_clamp: float = DEFAULT_KL_CLAMP,
    ):
        if not ensemble_models:
            raise ValueError("ensemble_models must be non-empty")
        self._models = list(ensemble_models)
        for m in self._models:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        self.mode = mode
        self.w_min = float(w_min)
        self.kl_clamp = float(kl_clamp)

    @property
    def n_members(self) -> int:
        return len(self._models)

    @classmethod
    def load(
        cls,
        checkpoints: Iterable[str | Path],
        num_symbols: int,
        features_per_sym: int,
        device: torch.device | str,
        mode: str = "softmax_avg",
        w_min: float = DEFAULT_W_MIN,
        kl_clamp: float = DEFAULT_KL_CLAMP,
    ) -> "EnsembleOrthogonalLoss":
        """Load N checkpoints via the standard load_policy helper."""
        from pufferlib_market.evaluate_holdout import load_policy  # local import

        models: list[torch.nn.Module] = []
        for cp in checkpoints:
            lp = load_policy(
                str(cp),
                int(num_symbols),
                features_per_sym=int(features_per_sym),
                device=device,
            )
            models.append(lp.policy)
        return cls(models, mode=mode, w_min=w_min, kl_clamp=kl_clamp)

    def ensemble_log_probs(self, obs: torch.Tensor) -> torch.Tensor:
        return compute_ensemble_log_probs(self._models, obs, mode=self.mode)

    def loss(
        self,
        specialist_logits: torch.Tensor,
        obs: torch.Tensor,
        beta: float,
        gate: str = "entropy",
    ) -> tuple[torch.Tensor, OrthogonalTermDiagnostics]:
        ens_log_probs = self.ensemble_log_probs(obs)
        return orthogonal_loss(
            specialist_logits=specialist_logits,
            ensemble_log_probs=ens_log_probs,
            beta=beta,
            gate=gate,
            w_min=self.w_min,
            kl_clamp=self.kl_clamp,
        )
