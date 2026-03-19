"""Backward-compatible re-export of binanceneural.model.

Legacy GRU / SCINet helpers are preserved at the bottom for older scripts
(e.g. predict_stock.py) that still ``from model import GRU``.
"""
from __future__ import annotations

from binanceneural.model import *  # noqa: F401,F403
from binanceneural.model import (
    BinanceHourlyPolicy,
    BinanceHourlyPolicyNano,
    BinancePolicyBase,
    PolicyConfig,
    PositionalEncoding,
    align_state_dict_input_dim,
    build_policy,
    policy_config_from_payload,
)

# ---------------------------------------------------------------------------
# Legacy helpers kept for backward compatibility (predict_stock.py, etc.)
# Guarded behind try/except so the binanceneural re-exports above always work
# even when SCINet is not installed.
# ---------------------------------------------------------------------------
import torch
from torch import nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    from SCINet.models.SCINet import SCINet  # noqa: F811
except ImportError:
    SCINet = None  # type: ignore[assignment,misc]


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to("cuda")
        out, (hn) = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


def get_model(input_len, input_dim=9, hidden_dim=1, num_stacks=1, output_dim=1):
    if SCINet is None:
        raise ImportError("SCINet is required for get_model(); install it first.")
    model = SCINet(
        output_len=output_dim,
        input_len=input_len,
        input_dim=input_dim,
        hid_size=hidden_dim,
        num_stacks=1,
        num_levels=3,
        concat_len=0,
        groups=1,
        kernel=3,
        dropout=0.5,
        single_step_output_One=0,
        positionalE=True,
        modified=True,
        RIN=False,
    )
    model.to(DEVICE)
    return model
