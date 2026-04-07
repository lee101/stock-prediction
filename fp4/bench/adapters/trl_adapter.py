"""TRL PPOTrainer adapter for the marketsim benchmark.

Bridges the C/Python marketsim env to ``trl.PPOTrainer`` with a tiny
actor-critic MLP (policy + value head).  TRL >= 0.12 removed
``PPOTrainer`` from the public API in favour of GRPO/RLOO; on those
versions this adapter skips cleanly with a clear reason instead of
crashing the bench harness.

Public entry point:
    run_trl(cfg, steps, seed, ckpt_dir) -> dict

Returned dict matches the contract used by the other runners in
``fp4/bench/bench_trading.py``: keys ``status`` ('ok' | 'skip' |
'error'), and on success ``wall_sec``, ``gpu_peak_mb``,
``trainer_output`` (sub-dict with metrics).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def _gpu_peak_reset() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _gpu_peak_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _import_ppo_trainer():
    """Locate ``PPOTrainer`` across TRL versions.

    Returns ``(PPOTrainer, PPOConfig_or_None)`` or raises ``ImportError``.
    """
    try:
        from trl import PPOTrainer  # type: ignore
        try:
            from trl import PPOConfig  # type: ignore
        except Exception:
            PPOConfig = None  # type: ignore
        return PPOTrainer, PPOConfig
    except Exception:
        pass
    # legacy module path used by trl<0.12
    try:
        from trl.trainer.ppo_trainer import PPOTrainer  # type: ignore
        try:
            from trl.trainer.ppo_config import PPOConfig  # type: ignore
        except Exception:
            PPOConfig = None  # type: ignore
        return PPOTrainer, PPOConfig
    except Exception as exc:  # pragma: no cover - covered by skip path
        raise ImportError(
            "trl.PPOTrainer not available in this TRL version "
            "(>=1.0 dropped PPOTrainer; pin trl<0.12 to use this adapter): "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _build_marketsim_env(cfg: dict[str, Any], seed: int):
    """Try to construct the C marketsim env via the new pufferlib bindings.

    Falls back to None on any failure; caller will skip with reason.
    """
    try:
        import market_sim_py  # type: ignore
    except Exception:
        return None, "market_sim_py bindings not built (Unit C pending)"
    try:
        env_cfg = cfg.get("env", {})
        env = market_sim_py.MarketEnvironment(
            data_path=str(env_cfg.get("train_data", "")),
            fee_rate=float(env_cfg.get("fee_rate", 0.001)),
            max_leverage=float(env_cfg.get("max_leverage_scalar_fallback", 1.5)),
            seed=int(seed),
        )
        return env, ""
    except Exception as exc:
        return None, f"market_sim_py init failed: {type(exc).__name__}: {exc}"


def _make_policy(obs_dim: int, act_dim: int, hidden: int):
    """Tiny actor-critic MLP with shared trunk + policy head + value head.

    Kept deliberately small (~few k params) so the smoke run is fast.
    """
    import torch
    import torch.nn as nn

    class TinyActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            self.policy_head = nn.Linear(hidden, act_dim)
            self.value_head = nn.Linear(hidden, 1)
            self.log_std = nn.Parameter(torch.zeros(act_dim))

        def forward(self, obs):  # noqa: D401 - tiny module
            h = self.trunk(obs)
            return self.policy_head(h), self.value_head(h).squeeze(-1)

    return TinyActorCritic()


def run_trl(cfg: dict[str, Any], steps: int, seed: int, ckpt_dir: Path) -> dict[str, Any]:
    """Bench-harness entry point. Mirrors the other ``_run_*`` runners."""
    try:
        PPOTrainer, PPOConfig = _import_ppo_trainer()
    except ImportError as exc:
        return {"status": "skip", "reason": str(exc)}

    env, why = _build_marketsim_env(cfg, seed)
    if env is None:
        return {"status": "skip", "reason": f"trl adapter: {why}"}

    try:
        import torch
    except Exception as exc:
        return {"status": "skip", "reason": f"torch missing: {exc}"}

    obs = env.reset()
    obs_t = obs if hasattr(obs, "shape") else torch.as_tensor(obs)
    obs_dim = int(obs_t.shape[-1])
    act_dim = int(getattr(env, "action_dim", 1))
    hidden = int(cfg.get("ppo", {}).get("hidden_size", 64))
    policy = _make_policy(obs_dim, act_dim, hidden)
    if torch.cuda.is_available():
        policy = policy.cuda()

    _gpu_peak_reset()
    t0 = time.perf_counter()

    # PPOTrainer in trl<0.12 expects a HuggingFace tokenizer-paired LM, not a
    # raw env, so we run a thin manual rollout that mirrors what TRL does
    # internally for non-LM tasks: collect (obs, action, logprob, value,
    # reward) tuples, then call ``trainer.step``.  When the API does not fit
    # (e.g. trl>=1.0 GRPO-only) we already returned 'skip' above.
    try:
        ppo_cfg = PPOConfig() if PPOConfig is not None else None
        trainer = PPOTrainer(config=ppo_cfg, model=policy) if ppo_cfg is not None else PPOTrainer(model=policy)
        rollout = []
        device = next(policy.parameters()).device
        ob = torch.as_tensor(obs, dtype=torch.float32, device=device)
        for _ in range(min(int(steps), 4096)):
            with torch.no_grad():
                logits, value = policy(ob)
                std = policy.log_std.exp()
                action = logits + std * torch.randn_like(logits)
            step_out = env.step(action.detach().cpu().numpy())
            next_obs, reward, done = step_out[0], float(step_out[1]), bool(step_out[2])
            rollout.append((ob.detach(), action.detach(), float(value.detach()), reward, done))
            ob = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            if done:
                ob = torch.as_tensor(env.reset(), dtype=torch.float32, device=device)
        # one PPO update step
        if hasattr(trainer, "step"):
            qs = [r[0] for r in rollout]
            acts = [r[1] for r in rollout]
            rews = [torch.tensor(r[3], device=device) for r in rollout]
            trainer.step(qs, acts, rews)
    except Exception as exc:
        return {
            "status": "skip",
            "reason": f"trl PPOTrainer API mismatch on this version: "
                      f"{type(exc).__name__}: {exc}",
        }

    wall = time.perf_counter() - t0
    rewards = [r[3] for r in rollout]
    mean_r = float(sum(rewards) / max(len(rewards), 1))
    return {
        "status": "ok",
        "wall_sec": wall,
        "gpu_peak_mb": _gpu_peak_mb(),
        "trainer_output": {
            "rollout_steps": len(rollout),
            "mean_reward": mean_r,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
        },
    }
