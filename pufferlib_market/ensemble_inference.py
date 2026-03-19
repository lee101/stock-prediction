"""Ensemble inference for multiple PPO trading policies.

Combines N checkpoint policies via logit averaging, majority vote,
or softmax-weighted averaging to produce a single action per step.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pufferlib_market.evaluate_multiperiod import load_policy


ENSEMBLE_MODES = ("logit_avg", "majority_vote", "softmax_avg")


def load_policies(
    checkpoint_paths: list[str],
    num_symbols: int,
    device: torch.device,
) -> list[nn.Module]:
    """Load multiple checkpoints, returning a list of eval-mode policies."""
    policies = []
    for path in checkpoint_paths:
        policy, _, _ = load_policy(str(path), num_symbols, device=device)
        policies.append(policy)
    return policies


def _collect_logits(
    policies: list[nn.Module],
    obs_t: torch.Tensor,
) -> torch.Tensor:
    """Run obs through all policies, return stacked logits [N, 1, A]."""
    all_logits = []
    with torch.no_grad():
        for policy in policies:
            logits, _ = policy(obs_t)
            all_logits.append(logits)
    return torch.stack(all_logits, dim=0)


def _action_from_logit_avg(stacked_logits: torch.Tensor, deterministic: bool) -> int:
    """Average raw logits across policies, then pick action."""
    avg_logits = stacked_logits.mean(dim=0)
    if deterministic:
        return int(torch.argmax(avg_logits, dim=-1).item())
    return int(torch.distributions.Categorical(logits=avg_logits).sample().item())


def _action_from_softmax_avg(stacked_logits: torch.Tensor, deterministic: bool) -> int:
    """Average softmax probabilities across policies, then pick action."""
    probs = F.softmax(stacked_logits, dim=-1)
    avg_probs = probs.mean(dim=0)
    if deterministic:
        return int(torch.argmax(avg_probs, dim=-1).item())
    return int(torch.distributions.Categorical(probs=avg_probs).sample().item())


def _action_from_majority_vote(stacked_logits: torch.Tensor, deterministic: bool) -> int:
    """Each policy votes for its best action; most common wins (ties broken arbitrarily)."""
    if deterministic:
        votes = torch.argmax(stacked_logits, dim=-1).view(-1).tolist()
    else:
        votes = [
            int(torch.distributions.Categorical(logits=stacked_logits[i]).sample().item())
            for i in range(stacked_logits.shape[0])
        ]
    counter = Counter(votes)
    winner, _ = counter.most_common(1)[0]
    return winner


_MODE_DISPATCH = {
    "logit_avg": _action_from_logit_avg,
    "majority_vote": _action_from_majority_vote,
    "softmax_avg": _action_from_softmax_avg,
}


class EnsembleTrader:
    """Combines N checkpoint policies via logit averaging or majority vote."""

    def __init__(
        self,
        checkpoint_paths: list[str] | None = None,
        num_symbols: int = 5,
        device: str = "cpu",
        mode: str = "logit_avg",
        *,
        policies: list[nn.Module] | None = None,
    ):
        if mode not in _MODE_DISPATCH:
            raise ValueError(f"mode must be one of {tuple(_MODE_DISPATCH)}, got {mode!r}")
        self.mode = mode
        self.device = torch.device(device)
        self._action_fn = _MODE_DISPATCH[mode]

        if policies is not None:
            self.policies = list(policies)
        elif checkpoint_paths:
            self.policies = load_policies(checkpoint_paths, num_symbols, self.device)
        else:
            raise ValueError("Must provide checkpoint_paths or policies")

    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Run obs through all policies, combine via the chosen mode."""
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device=self.device).view(1, -1)
        stacked = _collect_logits(self.policies, obs_t)
        return self._action_fn(stacked, deterministic)

    def get_policy_fn(self, deterministic: bool = True):
        """Return a policy_fn closure compatible with simulate_daily_policy."""
        def _policy_fn(obs: np.ndarray) -> int:
            return self.get_action(obs, deterministic=deterministic)
        return _policy_fn
