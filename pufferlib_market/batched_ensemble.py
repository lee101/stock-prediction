"""Batched ensemble forward for TradingPolicy.

Stacks N identical-architecture TradingPolicy members' weights into a
single `torch.bmm`-based forward so an ensemble inference at batch=1
costs **one** GPU launch instead of N.

Contract: produces the same per-member logits as calling each policy's
`.forward(obs)` in a loop, within fp32 epsilon. Tests in
`tests/test_batched_ensemble.py` assert exact argmax parity against the
serial path on 1000 random observations across the current v7 12-model
prod ensemble.

Only handles the fixed TradingPolicy architecture (3-layer ReLU encoder
+ optional post-encoder LayerNorm + 2-layer actor/critic heads). Policies
with different hidden sizes or `per_sym_norm=True` fall back silently —
the `can_batch(policies)` helper tells the caller whether to use this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from pufferlib_market.evaluate_holdout import TradingPolicy


def can_batch(policies: Sequence[TradingPolicy]) -> bool:
    """True iff every policy has identical layer shapes, no per_sym_norm,
    and either all have `_use_encoder_norm` flipped on or all off."""
    if not policies:
        return False
    ref = policies[0]
    if getattr(ref, "_per_sym_norm", False):
        return False
    ref_en_norm = bool(getattr(ref, "_use_encoder_norm", False))
    ref_shapes = _shape_tuple(ref)
    for p in policies:
        if getattr(p, "_per_sym_norm", False):
            return False
        if bool(getattr(p, "_use_encoder_norm", False)) != ref_en_norm:
            return False
        if _shape_tuple(p) != ref_shapes:
            return False
    return True


def _shape_tuple(p: TradingPolicy) -> tuple:
    enc = p.encoder
    # Expect the stock shape: Linear, act, Linear, act, Linear, act — 3 linears.
    lins = [m for m in enc if isinstance(m, torch.nn.Linear)]
    acts = [m for m in enc if not isinstance(m, torch.nn.Linear)]
    actor_lins = [m for m in p.actor if isinstance(m, torch.nn.Linear)]
    return (
        tuple(l.weight.shape for l in lins),
        type(acts[0]).__name__ if acts else None,
        tuple(l.weight.shape for l in actor_lins),
    )


@dataclass
class StackedEnsemble:
    """Weight tensors stacked along dim=0 (N = ensemble size).

    All tensors live on the same device. Use `.forward(obs)` to get
    per-member logits of shape [N, B, num_actions].
    """
    n_members: int
    device: torch.device
    # encoder: 3 linears
    enc_w: list[torch.Tensor]   # each [N, out, in]
    enc_b: list[torch.Tensor]   # each [N, out]
    # encoder_norm (optional)
    use_en_norm: bool
    en_norm_w: torch.Tensor | None   # [N, H]
    en_norm_b: torch.Tensor | None   # [N, H]
    # actor: 2 linears
    actor_w: list[torch.Tensor]
    actor_b: list[torch.Tensor]
    activation: str

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: [B, F] → logits: [N, B, A]. Matches per-member forward."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.size(0)
        N = self.n_members
        # Broadcast obs to [N, B, F] then run as batched bmm.
        x = obs.unsqueeze(0).expand(N, B, -1).contiguous()
        # encoder linear 1: [N,B,F] @ [N,F,H] + [N,1,H]
        for li in range(3):
            w = self.enc_w[li]              # [N, out, in]
            b = self.enc_b[li]              # [N, out]
            x = torch.bmm(x, w.transpose(1, 2)) + b.unsqueeze(1)
            x = _apply_act(x, self.activation)
        if self.use_en_norm:
            # Per-member LayerNorm over last dim; weights are [N, H]
            x = F.layer_norm(x, (x.size(-1),))
            x = x * self.en_norm_w.unsqueeze(1) + self.en_norm_b.unsqueeze(1)
        # actor linear 1 + act + linear 2
        w0 = self.actor_w[0]; b0 = self.actor_b[0]
        w1 = self.actor_w[1]; b1 = self.actor_b[1]
        h = torch.bmm(x, w0.transpose(1, 2)) + b0.unsqueeze(1)
        h = _apply_act(h, self.activation)
        logits = torch.bmm(h, w1.transpose(1, 2)) + b1.unsqueeze(1)
        return logits

    @classmethod
    def from_policies(cls, policies: Sequence[TradingPolicy], device: torch.device) -> "StackedEnsemble":
        assert can_batch(policies), "policies are not stack-compatible (per_sym_norm or mismatched shapes)"
        N = len(policies)
        enc_lins = [[m for m in p.encoder if isinstance(m, torch.nn.Linear)] for p in policies]
        enc_acts = [m for m in policies[0].encoder if not isinstance(m, torch.nn.Linear)]
        activation = type(enc_acts[0]).__name__.lower() if enc_acts else "relu"
        enc_w = [torch.stack([pl[li].weight for pl in enc_lins], dim=0).to(device).contiguous()
                 for li in range(3)]
        enc_b = [torch.stack([pl[li].bias for pl in enc_lins], dim=0).to(device).contiguous()
                 for li in range(3)]
        use_en_norm = bool(getattr(policies[0], "_use_encoder_norm", False))
        en_norm_w = en_norm_b = None
        if use_en_norm:
            en_norm_w = torch.stack([p.encoder_norm.weight for p in policies], dim=0).to(device).contiguous()
            en_norm_b = torch.stack([p.encoder_norm.bias for p in policies], dim=0).to(device).contiguous()
        actor_lins = [[m for m in p.actor if isinstance(m, torch.nn.Linear)] for p in policies]
        actor_w = [torch.stack([pl[li].weight for pl in actor_lins], dim=0).to(device).contiguous()
                   for li in range(2)]
        actor_b = [torch.stack([pl[li].bias for pl in actor_lins], dim=0).to(device).contiguous()
                   for li in range(2)]
        return cls(
            n_members=N, device=device,
            enc_w=enc_w, enc_b=enc_b,
            use_en_norm=use_en_norm, en_norm_w=en_norm_w, en_norm_b=en_norm_b,
            actor_w=actor_w, actor_b=actor_b, activation=activation,
        )


def _apply_act(x: torch.Tensor, name: str) -> torch.Tensor:
    name = name.lower()
    if name in ("relu",):
        return torch.relu(x)
    if name in ("gelu",):
        return F.gelu(x)
    if name in ("tanh",):
        return torch.tanh(x)
    if name in ("silu", "swish"):
        return F.silu(x)
    if name in ("leakyrelu", "leaky_relu"):
        return F.leaky_relu(x)
    raise ValueError(f"unsupported activation {name!r}")
