"""GPU-parallel GAE via Triton. Falls back to CPU if Triton unavailable.

Convention: dones[t] == 1 means step t is terminal. The bootstrap from
values[t+1] and GAE carry are zeroed when dones[t] == 1, matching the
inline GAE loop in train.py.
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _gae_kernel(
        rewards_ptr, values_ptr, dones_ptr,
        next_value_ptr,
        advantages_ptr, returns_ptr,
        N,
        T: tl.constexpr,
        gamma: tl.constexpr,
        gae_lambda: tl.constexpr,
    ):
        env_id = tl.program_id(0)
        if env_id >= N:
            return

        nv = tl.load(next_value_ptr + env_id)
        last_gae = tl.zeros([], dtype=tl.float32)

        for t_rev in range(T):
            t = T - 1 - t_rev
            off = t * N + env_id
            r = tl.load(rewards_ptr + off)
            v = tl.load(values_ptr + off)
            not_done = 1.0 - tl.load(dones_ptr + off)

            if t == T - 1:
                next_val = nv
            else:
                next_val = tl.load(values_ptr + (t + 1) * N + env_id)

            delta = r + gamma * next_val * not_done - v
            last_gae = delta + gamma * gae_lambda * not_done * last_gae

            tl.store(advantages_ptr + off, last_gae)
            tl.store(returns_ptr + off, last_gae + v)


def compute_gae_gpu(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    """GPU-parallel GAE. Returns (advantages, returns).

    Args:
        rewards: [T, N] float32
        values: [T, N] float32
        dones: [T, N] float32 (1.0 = terminal at step t)
        next_value: [N] float32 (bootstrap value after last step)
        gamma: discount factor
        gae_lambda: GAE lambda
    """
    if not HAS_TRITON or not rewards.is_cuda:
        return _compute_gae_cpu(rewards, values, dones, next_value, gamma, gae_lambda)

    T, N = rewards.shape
    advantages = torch.empty_like(rewards)
    returns = torch.empty_like(rewards)

    _gae_kernel[(N,)](
        rewards, values, dones,
        next_value,
        advantages, returns,
        N, T=T, gamma=gamma, gae_lambda=gae_lambda,
    )
    return advantages, returns


def _compute_gae_cpu(rewards, values, dones, next_value, gamma, gae_lambda):
    """Reference CPU implementation matching train.py inline loop."""
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * not_done - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns
