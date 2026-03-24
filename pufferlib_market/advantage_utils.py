from __future__ import annotations

import torch


def normalize_advantages(
    advantages: torch.Tensor,
    *,
    rewards: torch.Tensor | None = None,
    mode: str = "global",
    group_relative_size: int = 8,
    group_relative_mix: float = 0.0,
    group_relative_clip: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize advantages for PPO or GSPO-like rollout ranking.

    `group_relative` is intentionally a conservative approximation:
    1. normalize each env trajectory across time,
    2. compute a rollout-level score from summed rewards,
    3. rescale each env trajectory by its z-scored group rank,
    4. normalize the full batch again before PPO updates.
    """
    mode = str(mode or "global").strip().lower()

    if mode == "global":
        # No clone needed: arithmetic ops produce new tensors without mutating input.
        return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)

    # Non-global modes operate on a private copy to avoid mutating the caller's tensor.
    adv = advantages.clone()

    if adv.ndim != 2:
        raise ValueError(f"normalize_advantages expects [T, N] tensor, got shape={tuple(adv.shape)}")

    centered = adv - adv.mean(dim=0, keepdim=True)
    per_env = centered / (adv.std(dim=0, keepdim=True, unbiased=False) + eps)

    if mode == "per_env":
        return per_env

    if mode != "group_relative":
        raise ValueError(f"Unknown advantage normalization mode: {mode}")
    if rewards is None:
        raise ValueError("group_relative normalization requires rewards.")

    scores = rewards.sum(dim=0)
    num_envs = int(per_env.shape[1])
    group_size = max(2, min(int(group_relative_size), num_envs))
    mix = max(0.0, float(group_relative_mix))
    rel_clip = max(0.0, float(group_relative_clip))

    if mix <= 0.0:
        # No group rescaling needed — skip permutation, cloning, and the loop.
        return (per_env - per_env.mean()) / (per_env.std(unbiased=False) + eps)

    perm = torch.randperm(num_envs, device=per_env.device)
    inv_perm = torch.argsort(perm)
    permuted_adv = per_env[:, perm]
    permuted_scores = scores[perm]

    adjusted = permuted_adv.clone()
    for start in range(0, num_envs, group_size):
        idx = slice(start, min(start + group_size, num_envs))
        group_scores = permuted_scores[idx]
        denom = group_scores.std(unbiased=False) + eps
        rel = (group_scores - group_scores.mean()) / denom
        if rel_clip > 0.0:
            rel = torch.clamp(rel, -rel_clip, rel_clip)
        weights = torch.clamp(1.0 + mix * rel, min=0.1)
        adjusted[:, idx] = adjusted[:, idx] * weights.unsqueeze(0)

    adjusted = adjusted[:, inv_perm]
    return (adjusted - adjusted.mean()) / (adjusted.std(unbiased=False) + eps)
