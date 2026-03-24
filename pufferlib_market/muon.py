"""
Muon optimizer — MomentUm Orthogonalized by Newton-schulz.

Ported from cutellm/parameter-golf/train_gpt.py (single-GPU, no DDP).

For 2D+ weight matrices: applies Newton-Schulz orthogonalization to the
momentum buffer before applying the SGD update.
For 1D parameters (biases, layer-norm scales): falls back to AdamW.

NorMuon variant (norm_update=True): after Newton-Schulz, scales each
update so its Frobenius norm equals the parameter's Frobenius norm,
keeping parameter norms stable throughout training.  From the modded-nanogpt
speedrun (https://github.com/KellerJordan/modded-nanogpt).

Background: https://kellerjordan.github.io/posts/muon/
"""

from __future__ import annotations

import torch
import torch.optim as optim
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Newton-Schulz iteration to compute the approximate polar factor of G.

    Normalises G and repeatedly applies:
        A = X @ X.T
        X <- a*X + (b*A + c*A@A) @ X
    with coefficients (a=3.4445, b=-4.7750, c=2.0315) tuned for fast convergence.
    5 iterations flatten the singular-value spectrum sufficiently for gradient use.

    Args:
        G: 2D gradient tensor (rows x cols).  Transposed internally when rows > cols.
        steps: number of Newton-Schulz iterations (default 5).
        eps: small constant for numerical stability in normalisation.

    Returns:
        Orthogonalised update tensor with the same shape and dtype as G.
    """
    assert G.ndim >= 2, f"zeropower_via_newtonschulz5 requires ndim>=2, got {G.ndim}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz.

    Applies Newton-Schulz orthogonalization to the Nesterov momentum buffer
    of each 2D+ parameter before taking an SGD step.

    Designed for weight-matrix parameters.  Pass 1D params (biases, norm
    scales) via the adamw_params kwarg so they receive a proper AdamW update.

    Args:
        params: iterable of 2D+ parameters (weight matrices).
        lr: learning rate for orthogonalised update (default 0.02).
        momentum: SGD momentum (default 0.95).
        nesterov: use Nesterov momentum (default True).
        ns_steps: Newton-Schulz iterations (default 5).
        norm_update: NorMuon — scale update so its Frobenius norm equals the
            parameter's Frobenius norm (×lr).  Keeps param norms stable.
        adamw_params: iterable of 1D parameters for AdamW fallback.
        adamw_lr: learning rate for the AdamW fallback (default 3e-4).
        adamw_betas: AdamW betas (default (0.9, 0.999)).
        adamw_wd: AdamW weight decay (default 0.0).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        norm_update: bool = False,
        adamw_params=None,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_wd: float = 0.0,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        norm_update=norm_update)
        super().__init__(params, defaults)

        self._adamw: optim.AdamW | None = None
        if adamw_params is not None:
            adamw_list = list(adamw_params)
            if adamw_list:
                self._adamw = optim.AdamW(
                    adamw_list,
                    lr=adamw_lr,
                    betas=adamw_betas,
                    weight_decay=adamw_wd,
                    eps=1e-8,
                )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            norm_update = group["norm_update"]

            for p in params:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(g)
                g_eff = g.add(buf, alpha=momentum) if nesterov else buf

                if g_eff.ndim >= 2:
                    update = zeropower_via_newtonschulz5(g_eff, steps=ns_steps)
                    # Scale correction: preserves RMS magnitude across non-square shapes
                    update = update * max(1, update.size(0) / update.size(1)) ** 0.5
                    if norm_update:
                        # NorMuon: match update Frobenius norm to parameter norm so
                        # that param norms stay stable over training.
                        param_norm = p.norm(dtype=torch.float32)
                        update_norm = update.norm(dtype=torch.float32)
                        update = update * (param_norm / (update_norm + 1e-8)).to(update.dtype)
                else:
                    update = g_eff

                p.add_(update, alpha=-lr)

        if self._adamw is not None:
            self._adamw.step()

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        if self._adamw is not None:
            self._adamw.zero_grad(set_to_none=set_to_none)


def make_muon_optimizer(
    policy: torch.nn.Module,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    adamw_lr: float = 3e-4,
    adamw_wd: float = 0.0,
    ns_steps: int = 5,
    norm_update: bool = False,
) -> Muon:
    """
    Convenience factory: splits policy parameters into 2D (Muon) and 1D (AdamW).

    Args:
        policy: the policy nn.Module.
        muon_lr: learning rate for Muon (weight matrices).
        muon_momentum: SGD momentum for Muon.
        adamw_lr: learning rate for AdamW fallback (biases, norm params).
        adamw_wd: weight decay for AdamW fallback.
        ns_steps: Newton-Schulz iterations.

    Returns:
        Configured Muon optimizer.
    """
    matrix_params = [p for p in policy.parameters() if p.ndim >= 2]
    scalar_params = [p for p in policy.parameters() if p.ndim < 2]
    return Muon(
        matrix_params,
        lr=muon_lr,
        momentum=muon_momentum,
        ns_steps=ns_steps,
        norm_update=norm_update,
        adamw_params=scalar_params,
        adamw_lr=adamw_lr,
        adamw_wd=adamw_wd,
    )


def make_normuon_optimizer(
    policy: torch.nn.Module,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    adamw_lr: float = 3e-4,
    adamw_wd: float = 0.0,
    ns_steps: int = 5,
) -> Muon:
    """NorMuon variant: Muon with parameter-norm-matched updates.

    Convenience wrapper for make_muon_optimizer(norm_update=True).
    """
    return make_muon_optimizer(
        policy,
        muon_lr=muon_lr,
        muon_momentum=muon_momentum,
        adamw_lr=adamw_lr,
        adamw_wd=adamw_wd,
        ns_steps=ns_steps,
        norm_update=True,
    )
