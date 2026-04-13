#!/usr/bin/env python3
"""Export a PyTorch policy checkpoint to ONNX format for Go inference."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn

from binanceneural.config import TrainingConfig
from binanceneural.model import build_policy, policy_config_from_payload
from src.torch_load_utils import torch_load_compat


class PolicyWrapper(nn.Module):
    """Wraps the policy to output flat logits tensor for ONNX export."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, features):
        outputs = self.policy(features)
        # Stack logits into (B, T, num_outputs)
        logit_keys = ["buy_price_logits", "sell_price_logits",
                       "buy_amount_logits", "sell_amount_logits"]
        if "hold_hours_logits" in outputs:
            logit_keys.append("hold_hours_logits")
        if "allocation_logits" in outputs:
            logit_keys.append("allocation_logits")
        stacked = torch.stack([outputs[k].squeeze(-1) for k in logit_keys], dim=-1)
        return stacked


def export(ckpt_path: str, output_path: str, seq_len: int = 48):
    payload = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    cfg = payload.get("config", TrainingConfig())
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__

    # Detect input_dim from state_dict
    embed_key = next((k for k in state_dict if "embed" in k and "weight" in k), None)
    if embed_key:
        input_dim = state_dict[embed_key].shape[1]
    else:
        input_dim = cfg.get("input_dim", 32)

    from binanceneural.model import align_state_dict_input_dim
    state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)
    policy_cfg = policy_config_from_payload(cfg, input_dim=input_dim, state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    wrapper = PolicyWrapper(model)

    dummy = torch.randn(1, seq_len, input_dim)
    with torch.no_grad():
        test_out = wrapper(dummy)
    print(f"test output shape: {test_out.shape}")

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
    )
    print(f"exported to {output_path}")
    print(f"  input_dim={input_dim} seq_len={seq_len} num_outputs={test_out.shape[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="path to .pt checkpoint")
    parser.add_argument("--output", default=None, help="output .onnx path")
    parser.add_argument("--seq-len", type=int, default=48)
    args = parser.parse_args()

    output = args.output or args.checkpoint.replace(".pt", ".onnx")
    export(args.checkpoint, output, args.seq_len)
