#!/usr/bin/env python
"""Produce an obs parity fixture from a real MKTD file at a fixed timestep.

The fixture exercises the ctrader C obs_builder against the exact Python
inference.build_observation code on shared data so we can trust the C
backtest matches Python evaluate_holdout.

Fixture schema (little-endian):
    u32 obs_dim
    u32 S
    u32 F
    u32 t
    float32[obs_dim] expected_obs
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _read_mktd_features_and_closes(path: str, t: int):
    """Parse the MKTD binary far enough to pull `features[t]` and `closes[t]`.
    Matches the header layout used by inference_tts / pufferlib_market.train."""
    with open(path, "rb") as f:
        header = f.read(64)
    # features_per_sym defaults to 16 for older files (inference_tts logic)
    # Header layout: <4sIIIII = magic(4), version(4), S(4), T(4), F(4), price_features(4)
    magic, version, S, T, F, price_features = struct.unpack("<4sIIIII", header[:24])
    if F == 0:
        F = 16

    with open(path, "rb") as f:
        f.seek(64)
        # symbol table
        sym_bytes = f.read(S * 16)
        symbols = [sym_bytes[i * 16:(i + 1) * 16].split(b"\x00", 1)[0].decode() for i in range(S)]
        # features block [T*S*F] float32
        feat_bytes = f.read(T * S * F * 4)
        features = np.frombuffer(feat_bytes, dtype=np.float32).reshape(T, S, F)
        # price block: open, high, low, close, volume * T * S
        price_bytes = f.read(T * S * 5 * 4)
        prices = np.frombuffer(price_bytes, dtype=np.float32).reshape(T, S, 5)
    closes = prices[..., 3]
    return symbols, features, closes, int(T), int(S), int(F)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="MKTD binary file")
    p.add_argument("--t", type=int, default=30, help="timestep to snapshot")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    symbols, features, closes, T, S, F = _read_mktd_features_and_closes(args.data, args.t)
    if args.t >= T:
        print(f"t={args.t} >= T={T}", file=sys.stderr)
        return 1

    obs_dim = S * F + 5 + S
    obs = np.zeros(obs_dim, dtype=np.float32)

    # Replicate pufferlib_market/inference.py:build_observation exactly for
    # the flat case: no open position, cash=10000, hold=0, step=t.
    obs[:S * F] = features[args.t].flatten()
    base = S * F
    cash = 10000.0
    pos_val = 0.0
    max_steps = 90
    obs[base + 0] = cash / 10000.0
    obs[base + 1] = pos_val / 10000.0
    obs[base + 2] = 0.0
    obs[base + 3] = 0.0 / max_steps
    obs[base + 4] = args.t / max_steps
    # one-hot position: all zeros (flat)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<IIII", obs_dim, S, F, args.t))
        f.write(obs.tobytes())

    print(f"wrote {out}  S={S} F={F} obs_dim={obs_dim} t={args.t}  symbols={symbols[:S]}")
    print(f"  obs head: {obs[:6]}  portfolio block: {obs[S*F:S*F+5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
