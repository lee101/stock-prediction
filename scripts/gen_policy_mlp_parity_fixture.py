#!/usr/bin/env python
"""Generate a parity fixture that `ctrader/binance_bot/tests/test_policy_mlp_parity.c`
uses to verify the pure-C forward pass matches the Python model exactly.

Fixture format (little-endian float32):
    u32 obs_dim
    u32 num_actions
    float32[obs_dim] obs
    float32[num_actions] expected_logits
    i32 expected_argmax

The Python reference rebuilds a TradingPolicy with the exported architecture
and runs it on a deterministic RNG observation so the fixture is reproducible.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RefPolicy(nn.Module):
    """Minimal Python twin of the C forward pass, built straight from a
    stocks12 state_dict.  This must match `policy_mlp.c:ctrdpol_forward`
    exactly in fp32."""

    def __init__(self, sd: dict, *, use_encoder_norm: bool):
        super().__init__()
        enc_ids = sorted(
            int(k.split(".")[1]) for k in sd if k.startswith("encoder.") and k.endswith(".weight")
        )
        enc_layers: list[nn.Module] = []
        prev = sd[f"encoder.{enc_ids[0]}.weight"].shape[1]
        for li in enc_ids:
            W = sd[f"encoder.{li}.weight"]
            hidden = W.shape[0]
            lin = nn.Linear(prev, hidden)
            lin.weight.data.copy_(W)
            lin.bias.data.copy_(sd[f"encoder.{li}.bias"])
            enc_layers.append(lin)
            enc_layers.append(nn.ReLU())
            prev = hidden
        self.encoder = nn.Sequential(*enc_layers)

        self.use_encoder_norm = use_encoder_norm
        if use_encoder_norm:
            self.encoder_norm = nn.LayerNorm(prev)
            self.encoder_norm.weight.data.copy_(sd["encoder_norm.weight"])
            self.encoder_norm.bias.data.copy_(sd["encoder_norm.bias"])

        actor0 = nn.Linear(prev, sd["actor.0.weight"].shape[0])
        actor0.weight.data.copy_(sd["actor.0.weight"])
        actor0.bias.data.copy_(sd["actor.0.bias"])
        actor2 = nn.Linear(sd["actor.2.weight"].shape[1], sd["actor.2.weight"].shape[0])
        actor2.weight.data.copy_(sd["actor.2.weight"])
        actor2.bias.data.copy_(sd["actor.2.bias"])
        self.actor = nn.Sequential(actor0, nn.ReLU(), actor2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        if self.use_encoder_norm:
            h = self.encoder_norm(h)
        return self.actor(h)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    use_ln = bool(ck.get("use_encoder_norm", "encoder_norm.weight" in sd))

    policy = RefPolicy(sd, use_encoder_norm=use_ln).eval().to(torch.float32)
    obs_dim = int(sd["encoder.0.weight"].shape[1])
    num_actions = int(sd["actor.2.weight"].shape[0])

    rng = np.random.default_rng(args.seed)
    obs_np = rng.standard_normal(obs_dim).astype(np.float32)
    with torch.no_grad():
        logits = policy(torch.from_numpy(obs_np).unsqueeze(0)).squeeze(0).numpy()
    logits = logits.astype(np.float32)
    argmax = int(np.argmax(logits))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<II", obs_dim, num_actions))
        f.write(obs_np.tobytes())
        f.write(logits.tobytes())
        f.write(struct.pack("<i", argmax))

    print(f"wrote {out}  obs_dim={obs_dim}  num_actions={num_actions}  argmax={argmax}")
    print(f"  logits[{argmax}]={logits[argmax]:+.6f}  min={logits.min():+.6f}  max={logits.max():+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
