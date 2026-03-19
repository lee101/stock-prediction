"""Shared test fixtures for CuteChronos2Model tests."""

import os
import sys

import pytest
import torch

# Ensure chronos2 source is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "chronos-forecasting", "src"))

from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model

from cutechronos.model import CuteChronos2Model


def make_test_config() -> Chronos2CoreConfig:
    """Create a Chronos2CoreConfig for testing (no dropout for determinism)."""
    return Chronos2CoreConfig(
        d_model=768,
        d_kv=64,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        initializer_factor=0.05,
        feed_forward_proj="relu",
        vocab_size=2,
        rope_theta=10000.0,
        attn_implementation="eager",
        chronos_config={
            "context_length": 512,
            "input_patch_size": 16,
            "input_patch_stride": 16,
            "output_patch_size": 16,
            "max_output_patches": 4,
            "quantiles": [
                0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
            ],
            "use_reg_token": True,
            "use_arcsinh": True,
        },
    )


def build_model_pair():
    """Build an original and cute model pair sharing the same random weights."""
    torch.manual_seed(42)
    config = make_test_config()
    original = Chronos2Model(config).eval()
    cute = CuteChronos2Model.from_original(original)
    return original, cute


def build_cute_only():
    """Build a CuteChronos2Model (discards the original)."""
    _, cute = build_model_pair()
    return cute
