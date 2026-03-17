from __future__ import annotations

import torch

from pufferlib_market.advantage_utils import normalize_advantages


def test_normalize_advantages_global_zero_centers_batch() -> None:
    advantages = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    normalized = normalize_advantages(advantages, mode="global")

    assert torch.isclose(normalized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(normalized.std(unbiased=False), torch.tensor(1.0), atol=1e-6)


def test_normalize_advantages_per_env_zero_centers_each_column() -> None:
    advantages = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=torch.float32,
    )

    normalized = normalize_advantages(advantages, mode="per_env")

    assert torch.allclose(normalized.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(normalized.std(dim=0, unbiased=False), torch.ones(2), atol=1e-6)


def test_normalize_advantages_group_relative_weights_better_rollouts_more() -> None:
    advantages = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ],
        dtype=torch.float32,
    )
    rewards = torch.tensor(
        [
            [3.0, 1.0, -1.0, -3.0],
            [3.0, 1.0, -1.0, -3.0],
            [3.0, 1.0, -1.0, -3.0],
        ],
        dtype=torch.float32,
    )

    normalized = normalize_advantages(
        advantages,
        rewards=rewards,
        mode="group_relative",
        group_relative_size=4,
        group_relative_mix=0.5,
        group_relative_clip=2.0,
    )
    amplitudes = normalized.abs().mean(dim=0)

    assert amplitudes[0] > amplitudes[1] > amplitudes[2] > amplitudes[3]


def test_normalize_advantages_group_relative_requires_rewards() -> None:
    advantages = torch.ones((2, 2), dtype=torch.float32)

    try:
        normalize_advantages(advantages, mode="group_relative")
    except ValueError as exc:
        assert "requires rewards" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError")
