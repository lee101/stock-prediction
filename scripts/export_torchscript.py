#!/usr/bin/env python3
"""Export pufferlib checkpoint to TorchScript for ctrader inference."""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pufferlib_market.train import (
    TradingPolicy,
    ResidualTradingPolicy,
    TransformerTradingPolicy,
)

ARCH_CLASSES = {
    "mlp": TradingPolicy,
    "residual": ResidualTradingPolicy,
    "transformer": TransformerTradingPolicy,
}


class PolicyLogitsOnly(nn.Module):
    """Wrapper that returns only logits (no value head) for C inference."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _value = self.policy(x)
        return logits


def infer_dims_from_state_dict(state_dict: dict, arch: str) -> dict:
    """Infer obs_size, num_actions, hidden from state_dict shapes."""
    if arch == "mlp":
        obs_size = state_dict["encoder.0.weight"].shape[1]
        hidden = state_dict["encoder.0.weight"].shape[0]
        num_actions = state_dict["actor.2.weight"].shape[0]
        return {"obs_size": obs_size, "num_actions": num_actions, "hidden": hidden}
    elif arch == "residual":
        obs_size = state_dict["input_proj.weight"].shape[1]
        hidden = state_dict["input_proj.weight"].shape[0]
        num_actions = state_dict["actor.2.weight"].shape[0]
        num_blocks = sum(1 for k in state_dict if k.startswith("blocks.") and k.endswith(".norm.weight"))
        return {"obs_size": obs_size, "num_actions": num_actions, "hidden": hidden, "num_blocks": num_blocks}
    elif arch == "transformer":
        features_per_sym = state_dict["symbol_proj.weight"].shape[1]
        hidden = state_dict["mlp.0.weight"].shape[0]
        num_actions = state_dict["actor.2.weight"].shape[0]
        embed_dim = max(hidden // 4, 32)
        attn_out_size = state_dict["mlp.0.weight"].shape[1]
        num_symbols = (attn_out_size - 5) // (embed_dim + 1)
        obs_size = num_symbols * (features_per_sym + 1) + 5
        return {
            "obs_size": obs_size,
            "num_actions": num_actions,
            "hidden": hidden,
            "features_per_sym": features_per_sym,
        }
    raise ValueError(f"Unknown arch: {arch}")


def build_model(arch: str, dims: dict) -> nn.Module:
    cls = ARCH_CLASSES[arch]
    if arch == "mlp":
        return cls(dims["obs_size"], dims["num_actions"], dims["hidden"])
    elif arch == "residual":
        return cls(dims["obs_size"], dims["num_actions"], dims["hidden"], dims.get("num_blocks", 3))
    elif arch == "transformer":
        return cls(
            dims["obs_size"],
            dims["num_actions"],
            dims["hidden"],
            features_per_sym=dims.get("features_per_sym", 16),
        )
    raise ValueError(f"Unknown arch: {arch}")


def main():
    parser = argparse.ArgumentParser(description="Export pufferlib checkpoint to TorchScript")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default="ctrader/models/policy.pt", help="Output TorchScript path")
    parser.add_argument("--include-value", action="store_true", help="Include value head (default: logits only)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        print("ERROR: checkpoint missing 'model' key")
        sys.exit(1)

    arch = ckpt.get("arch", "mlp")
    state_dict = ckpt["model"]
    dims = infer_dims_from_state_dict(state_dict, arch)

    print(f"arch={arch} obs={dims['obs_size']} actions={dims['num_actions']} hidden={dims['hidden']}")

    model = build_model(arch, dims)
    model.load_state_dict(state_dict)
    model.eval()

    if not args.include_value:
        model = PolicyLogitsOnly(model)

    example = torch.randn(1, dims["obs_size"])
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(out_path))

    # Verify round-trip
    loaded = torch.jit.load(str(out_path))
    with torch.no_grad():
        orig_out = model(example)
        loaded_out = loaded(example)
        if isinstance(orig_out, tuple):
            for i, (a, b) in enumerate(zip(orig_out, loaded_out)):
                if not torch.allclose(a, b, atol=1e-6):
                    print(f"ERROR: round-trip verification failed on output {i}")
                    sys.exit(1)
        else:
            if not torch.allclose(orig_out, loaded_out, atol=1e-6):
                print("ERROR: round-trip verification failed")
                sys.exit(1)

    meta = {
        "arch": arch,
        "obs_size": dims["obs_size"],
        "num_actions": dims["num_actions"],
        "hidden": dims["hidden"],
        "include_value": args.include_value,
        "source_checkpoint": str(ckpt_path),
        "action_allocation_bins": ckpt.get("action_allocation_bins"),
        "action_level_bins": ckpt.get("action_level_bins"),
        "action_max_offset_bps": ckpt.get("action_max_offset_bps"),
        "disable_shorts": ckpt.get("disable_shorts"),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Exported: {out_path} ({size_mb:.1f} MB)")
    print(f"Metadata: {meta_path}")
    if isinstance(loaded_out, tuple):
        print(f"Output shapes: logits={loaded_out[0].shape} value={loaded_out[1].shape}")
    else:
        print(f"Output shape: {loaded_out.shape}")


if __name__ == "__main__":
    main()
