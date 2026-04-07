"""Two-timescale Layer A / Layer B actor-critic policy.

Layer A (slow): target inventory fraction, target leverage, risk budget.
Layer B (fast): delta_bid_bps, delta_ask_bps, size_frac — consumed by
`gpu_trading_env`'s 4-wide action format `(p_bid, p_ask, q_bid, q_ask)`.

Shared encoder is a 3-layer MLP (F_obs -> 512 -> 512 -> 256) with GELU. All
hidden linears may be NVFP4Linear, plain nn.Linear (bf16/fp32) depending on
the `precision` cfg flag. Heads (policy, value, cost-values) stay in standard
precision per the NeMo NVFP4 recipe.

Gaussian stochastic policy with state-independent log_std per output dim for
both Layer A (3 dims) and Layer B (3 dims) => 6 action dims total.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .linear import NVFP4Linear


LAYER_A_DIM = 3  # inventory_frac, leverage, risk_budget
LAYER_B_DIM = 3  # delta_bid_bps, delta_ask_bps, size_frac
ACTION_DIM = LAYER_A_DIM + LAYER_B_DIM


def _make_linear(in_f: int, out_f: int, precision: str, seed: int) -> nn.Module:
    """Build a linear layer respecting the precision flag.

    precision='nvfp4' -> NVFP4Linear (default per recipe)
    precision='bf16'  -> nn.Linear, caller is responsible for dtype/autocast
    precision='fp32'  -> nn.Linear in fp32
    """
    p = precision.lower()
    if p == "nvfp4":
        return NVFP4Linear(in_f, out_f, seed=seed)
    if p in ("bf16", "fp32"):
        lin = nn.Linear(in_f, out_f)
        if p == "bf16" and torch.cuda.is_available():
            lin = lin.to(torch.bfloat16)
        return lin
    raise ValueError(f"unknown precision: {precision!r} (want nvfp4|bf16|fp32)")


class SharedEncoder(nn.Module):
    """Linear(F->512) GELU Linear(512->512) GELU Linear(512->256) GELU."""

    def __init__(self, obs_dim: int, precision: str = "nvfp4", seed: int = 0):
        super().__init__()
        self.l1 = _make_linear(obs_dim, 512, precision, seed + 0)
        self.l2 = _make_linear(512, 512, precision, seed + 1)
        self.l3 = _make_linear(512, 256, precision, seed + 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Match encoder weight dtype (bf16 encoders need bf16 inputs).
        w = getattr(self.l1, "weight", None)
        if w is not None and obs.dtype != w.dtype:
            obs = obs.to(w.dtype)
        x = F.gelu(self.l1(obs))
        x = F.gelu(self.l2(x))
        x = F.gelu(self.l3(x))
        return x


def _orthogonal_head(lin: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(lin.weight, gain=gain)
    nn.init.zeros_(lin.bias)


class TwoLayerPolicy(nn.Module):
    """Shared encoder + Layer A/B policy heads + value + cost-value heads.

    Args:
        obs_dim: observation feature count.
        n_costs: number of constraint cost-value heads (default 4 for
            drawdown / liquidation / leverage-violation / turnover).
        precision: 'nvfp4'|'bf16'|'fp32' — applies to the shared encoder only.
        delta_max_bps: cap for Layer B quote offset (default 50 bps).
        seed: rng seed for NVFP4Linear init.
    """

    def __init__(
        self,
        obs_dim: int,
        n_costs: int = 4,
        precision: str = "nvfp4",
        delta_max_bps: float = 50.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_costs = int(n_costs)
        self.precision = precision
        self.delta_max_bps = float(delta_max_bps)

        self.encoder = SharedEncoder(obs_dim, precision=precision, seed=seed)
        feat_dim = 256

        # Policy heads: keep standard precision (small, sensitive).
        self.head_a = nn.Linear(feat_dim, LAYER_A_DIM)
        self.head_b = nn.Linear(feat_dim, LAYER_B_DIM)
        self.v_head = nn.Linear(feat_dim, 1)
        self.cost_v_heads = nn.ModuleList(
            [nn.Linear(feat_dim, 1) for _ in range(self.n_costs)]
        )

        # State-independent log_std, one per action dim (A then B).
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -0.5))

        self._init_heads()

    def _init_heads(self) -> None:
        _orthogonal_head(self.head_a, gain=0.01)
        _orthogonal_head(self.head_b, gain=0.01)
        _orthogonal_head(self.v_head, gain=1.0)
        for h in self.cost_v_heads:
            _orthogonal_head(h, gain=1.0)

    # ---- bounded activations ------------------------------------------------

    @staticmethod
    def _layer_a_bound(raw: torch.Tensor) -> torch.Tensor:
        """raw[...,3] -> (inv_frac in [-1,1], leverage in [0,5], risk in [0,1])."""
        inv = torch.tanh(raw[..., 0:1])
        lev = 5.0 * torch.sigmoid(raw[..., 1:2])
        risk = torch.sigmoid(raw[..., 2:3])
        return torch.cat([inv, lev, risk], dim=-1)

    def _layer_b_bound(self, raw: torch.Tensor) -> torch.Tensor:
        """raw[...,3] -> (d_bid_bps, d_ask_bps in [0,Δmax], size_frac in [0,1])."""
        dbid = self.delta_max_bps * torch.sigmoid(raw[..., 0:1])
        dask = self.delta_max_bps * torch.sigmoid(raw[..., 1:2])
        size = torch.sigmoid(raw[..., 2:3])
        return torch.cat([dbid, dask, size], dim=-1)

    # ---- forward ------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(obs)
        # Heads are in the head_a weight dtype; cast features if encoder ran
        # in a lower precision (e.g. bf16) than the heads.
        head_dtype = self.head_a.weight.dtype
        if z.dtype != head_dtype:
            z = z.to(head_dtype)
        raw_a = self.head_a(z)
        raw_b = self.head_b(z)
        layer_a = self._layer_a_bound(raw_a)
        layer_b = self._layer_b_bound(raw_b)
        value = self.v_head(z).squeeze(-1)
        cost_values = torch.stack(
            [h(z).squeeze(-1) for h in self.cost_v_heads], dim=-1
        ) if self.n_costs > 0 else torch.zeros(
            *z.shape[:-1], 0, device=z.device, dtype=z.dtype
        )
        std = torch.exp(self.log_std).expand(*obs.shape[:-1], ACTION_DIM)
        return {
            "layer_a": layer_a,
            "layer_b": layer_b,
            "raw_a": raw_a,
            "raw_b": raw_b,
            "value": value,
            "cost_values": cost_values,
            "std": std,
        }

    # ---- quote conversion ---------------------------------------------------

    def to_quote_prices(
        self,
        layer_b_out: torch.Tensor,
        ref_px: torch.Tensor,
    ) -> torch.Tensor:
        """Convert Layer B output + reference price to gpu_trading_env action.

        layer_b_out: (..., 3) = (d_bid_bps, d_ask_bps, size_frac)
        ref_px:      (..., ) mid/reference price per env.
        returns:     (..., 4) = (p_bid, p_ask, q_bid, q_ask).
        """
        if ref_px.dim() == layer_b_out.dim():
            ref_px = ref_px.squeeze(-1)
        d_bid = layer_b_out[..., 0]
        d_ask = layer_b_out[..., 1]
        size = layer_b_out[..., 2]
        bps = 1e-4
        p_bid = ref_px * (1.0 - d_bid * bps)
        p_ask = ref_px * (1.0 + d_ask * bps)
        # Split size symmetrically across bid/ask quotes.
        q_bid = 0.5 * size
        q_ask = 0.5 * size
        return torch.stack([p_bid, p_ask, q_bid, q_ask], dim=-1)

    # ---- sampling / logprob -------------------------------------------------

    @staticmethod
    def gaussian_logprob(
        mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        var = std * std
        logp = -0.5 * (
            ((action - mean) ** 2) / (var + 1e-8)
            + 2 * torch.log(std + 1e-8)
            + math.log(2 * math.pi)
        )
        return logp.sum(dim=-1)

    @staticmethod
    def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
        return (0.5 * math.log(2 * math.pi * math.e) + torch.log(std + 1e-8)).sum(dim=-1)

    def act(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.forward(obs)
        # Sample in raw (pre-bound) space so logprobs are well-defined; we
        # return both raw sample and bounded action so the trainer can decide.
        mean_raw = torch.cat([out["raw_a"], out["raw_b"]], dim=-1)
        std = out["std"]
        eps = torch.randn_like(mean_raw)
        raw_action = mean_raw + std * eps
        logp = self.gaussian_logprob(mean_raw, std, raw_action)
        layer_a = self._layer_a_bound(raw_action[..., :LAYER_A_DIM])
        layer_b = self._layer_b_bound(raw_action[..., LAYER_A_DIM:])
        out.update(
            {
                "raw_action": raw_action,
                "logp": logp,
                "sampled_layer_a": layer_a,
                "sampled_layer_b": layer_b,
            }
        )
        return out
