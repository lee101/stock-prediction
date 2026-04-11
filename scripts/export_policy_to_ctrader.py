#!/usr/bin/env python
"""Export a pufferlib_market stocks12-style MLP policy to a flat binary that
`ctrader/binance_bot/policy_mlp.c` can mmap/read directly.

Binary format (little-endian, all floats are float32, all ints are int32):

    offset  type        field
    0       char[8]     magic = b"CTRDPOL1"
    8       u32         version = 1
    12      u32         obs_dim
    16      u32         hidden      (encoder hidden, typically 1024)
    20      u32         n_encoder_layers  (typically 3: three Linear+ReLU)
    24      u32         use_encoder_norm  (0/1; LayerNorm after encoder)
    28      u32         actor_hidden      (typically 512)
    32      u32         num_actions       (typically 1 + 2*num_symbols = 25)
    36      u32         num_symbols       (derived, for sanity checking)
    40      u32         activation        (0=relu, 1=relu_sq; only relu is
                                           supported in the C forward pass
                                           for now)
    44      u32         disable_shorts    (0/1)
    48      u32         _reserved[4]      (zeroed, 16 bytes)
    64      float32[*]  weights/biases in this exact order:
              encoder.0.weight  [hidden × obs_dim]
              encoder.0.bias    [hidden]
              encoder.2.weight  [hidden × hidden]
              encoder.2.bias    [hidden]
              encoder.4.weight  [hidden × hidden]
              encoder.4.bias    [hidden]
              encoder_norm.weight [hidden]      (only if use_encoder_norm)
              encoder_norm.bias   [hidden]
              actor.0.weight    [actor_hidden × hidden]
              actor.0.bias      [actor_hidden]
              actor.2.weight    [num_actions × actor_hidden]
              actor.2.bias      [num_actions]

Linear weights are stored in PyTorch row-major layout (`[out, in]`): so reading
`W[out*in + i]` yields the i-th element of the out-th row. The C forward pass
iterates `for o in range(out): for i in range(in): y[o] += W[o*in + i] * x[i]`.

Usage:
    python scripts/export_policy_to_ctrader.py \\
        --checkpoint pufferlib_market/checkpoints/stocks12_v5_rsi/tp05_s42/best.pt \\
        --out ctrader/models/stocks12_v5_rsi_s42.ctrdpol \\
        --num-symbols 12
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

MAGIC = b"CTRDPOL1"
VERSION = 1
HEADER_SIZE = 64  # magic(8) + version(4) + 9*u32 fields + 4*u32 reserved = 8+4+36+16 = 64

ACTIVATION_IDS = {"relu": 0, "relu_sq": 1}


def _pack_header(*, obs_dim: int, hidden: int, n_encoder_layers: int,
                 use_encoder_norm: bool, actor_hidden: int, num_actions: int,
                 num_symbols: int, activation: str, disable_shorts: bool) -> bytes:
    if activation not in ACTIVATION_IDS:
        raise ValueError(f"activation {activation!r} not supported by C forward pass")
    header = bytearray(HEADER_SIZE)
    header[0:8] = MAGIC
    struct.pack_into("<I", header, 8, VERSION)
    struct.pack_into("<I", header, 12, int(obs_dim))
    struct.pack_into("<I", header, 16, int(hidden))
    struct.pack_into("<I", header, 20, int(n_encoder_layers))
    struct.pack_into("<I", header, 24, int(bool(use_encoder_norm)))
    struct.pack_into("<I", header, 28, int(actor_hidden))
    struct.pack_into("<I", header, 32, int(num_actions))
    struct.pack_into("<I", header, 36, int(num_symbols))
    struct.pack_into("<I", header, 40, ACTIVATION_IDS[activation])
    struct.pack_into("<I", header, 44, int(bool(disable_shorts)))
    # reserved bytes 48..64 stay zero
    return bytes(header)


def _fp32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).contiguous().cpu().numpy()


def export(checkpoint_path: str, out_path: str, *, num_symbols: int) -> dict:
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

    # Derive shapes from state dict (single source of truth).
    enc0_w = sd["encoder.0.weight"]
    hidden, obs_dim = int(enc0_w.shape[0]), int(enc0_w.shape[1])
    # Count encoder.{even}.weight entries (0, 2, 4 in Linear+ReLU sequence).
    enc_layer_indices = sorted(
        int(k.split(".")[1])
        for k in sd.keys()
        if k.startswith("encoder.") and k.endswith(".weight")
    )
    n_encoder_layers = len(enc_layer_indices)
    for li in enc_layer_indices:
        assert sd[f"encoder.{li}.weight"].shape == (hidden, hidden) or li == enc_layer_indices[0], \
            f"encoder.{li}.weight shape {tuple(sd[f'encoder.{li}.weight'].shape)} != ({hidden},{hidden})"

    use_encoder_norm = "encoder_norm.weight" in sd
    if use_encoder_norm:
        assert sd["encoder_norm.weight"].shape == (hidden,)
        assert sd["encoder_norm.bias"].shape == (hidden,)

    actor_hidden, hidden_check = int(sd["actor.0.weight"].shape[0]), int(sd["actor.0.weight"].shape[1])
    assert hidden_check == hidden, f"actor.0 input {hidden_check} != encoder hidden {hidden}"
    num_actions = int(sd["actor.2.weight"].shape[0])
    assert sd["actor.2.weight"].shape[1] == actor_hidden

    activation = ck.get("activation", "relu") if isinstance(ck, dict) else "relu"
    disable_shorts = bool(ck.get("disable_shorts", False)) if isinstance(ck, dict) else False

    # num_actions = 1 + 2*S*alloc_bins*level_bins. For 1/1 bins: 1 + 2*S.
    inferred_symbols = (num_actions - 1) // 2
    if inferred_symbols != num_symbols:
        print(f"WARNING: inferred num_symbols={inferred_symbols} != provided {num_symbols}",
              file=sys.stderr)

    header = _pack_header(
        obs_dim=obs_dim,
        hidden=hidden,
        n_encoder_layers=n_encoder_layers,
        use_encoder_norm=use_encoder_norm,
        actor_hidden=actor_hidden,
        num_actions=num_actions,
        num_symbols=num_symbols,
        activation=activation,
        disable_shorts=disable_shorts,
    )

    # Weights in the exact order the C loader expects.
    tensors_in_order: list[np.ndarray] = []
    for li in enc_layer_indices:
        tensors_in_order.append(_fp32(sd[f"encoder.{li}.weight"]))  # [out, in]
        tensors_in_order.append(_fp32(sd[f"encoder.{li}.bias"]))    # [out]
    if use_encoder_norm:
        tensors_in_order.append(_fp32(sd["encoder_norm.weight"]))
        tensors_in_order.append(_fp32(sd["encoder_norm.bias"]))
    tensors_in_order.append(_fp32(sd["actor.0.weight"]))  # [actor_hidden, hidden]
    tensors_in_order.append(_fp32(sd["actor.0.bias"]))
    tensors_in_order.append(_fp32(sd["actor.2.weight"]))  # [num_actions, actor_hidden]
    tensors_in_order.append(_fp32(sd["actor.2.bias"]))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(header)
        for t in tensors_in_order:
            assert t.dtype == np.float32 and t.flags["C_CONTIGUOUS"]
            f.write(t.tobytes(order="C"))

    total_params = sum(t.size for t in tensors_in_order)
    summary = {
        "checkpoint": checkpoint_path,
        "out": str(out),
        "obs_dim": obs_dim,
        "hidden": hidden,
        "n_encoder_layers": n_encoder_layers,
        "use_encoder_norm": use_encoder_norm,
        "actor_hidden": actor_hidden,
        "num_actions": num_actions,
        "num_symbols": num_symbols,
        "activation": activation,
        "disable_shorts": disable_shorts,
        "total_params": total_params,
        "bytes_written": out.stat().st_size,
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-symbols", type=int, required=True,
                   help="number of trading symbols the policy was trained on (sanity check)")
    args = p.parse_args()
    s = export(args.checkpoint, args.out, num_symbols=args.num_symbols)
    for k, v in s.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
