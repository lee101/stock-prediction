"""HF Trainer adapter for the pufferlib C marketsim.

This adapter performs an *offline* PPO update on rollouts collected from the C
trading environment.  The rollout is run once with the current policy, then a
few PPO optimisation epochs are run via ``transformers.Trainer`` using a
custom ``compute_loss`` that implements the PPO clipped surrogate objective
plus a value-function loss and an entropy bonus.

It is intentionally minimal — the goal is an apples-to-apples *adapter* slot
in the bench harness so we can compare HF Trainer's optimisation loop against
pufferlib_market's hand-rolled PPO and the fp4 NVFP4 trainer on the same
environment.  No CUDA graphs, no heroic SPS optimisation here.

Cleanly skips with a clear reason when:
  * ``transformers`` is not importable, or
  * the pufferlib_market C binding / shared market data cannot be loaded, or
  * the rollout produces zero usable samples.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def _skip(reason: str) -> Dict[str, Any]:
    return {"status": "skip", "reason": reason}


def _read_num_symbols(data_path: Path) -> int:
    """Read num_symbols from the .bin header (matches bench_env._write_mktd_bin)."""
    import struct

    with open(data_path, "rb") as f:
        head = f.read(16)
    if len(head) < 8:
        raise ValueError(f"data file too small to contain header: {data_path}")
    # Header layout in pufferlib_market exporters: <i magic, i num_timesteps,
    # i num_symbols, i features_per_sym, ...> -- num_symbols is the 3rd int.
    fields = struct.unpack("<iiii", head)
    num_symbols = int(fields[2])
    if num_symbols <= 0 or num_symbols > 1024:
        raise ValueError(f"implausible num_symbols={num_symbols} in {data_path}")
    return num_symbols


def run(cfg: Dict[str, Any], steps: int, seed: int, ckpt_dir: Path) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import IterableDataset
    except Exception as exc:  # pragma: no cover - torch is a hard dep elsewhere
        return _skip(f"torch missing: {type(exc).__name__}: {exc}")

    try:
        import transformers
        from transformers import Trainer, TrainingArguments
    except Exception as exc:
        return _skip(f"transformers import failed: {type(exc).__name__}: {exc}")

    try:
        import pufferlib_market.binding as binding  # type: ignore
    except Exception as exc:
        return _skip(f"pufferlib_market.binding import failed: {type(exc).__name__}: {exc}")

    env_cfg = cfg.get("env", {})
    ppo_cfg = cfg.get("ppo", {})
    train_data = REPO / env_cfg.get("train_data", "")
    if not train_data.exists():
        return _skip(f"train data missing: {train_data}")

    try:
        num_symbols = _read_num_symbols(train_data)
    except Exception as exc:
        return _skip(f"could not read num_symbols header: {exc}")

    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = 1 + 2 * num_symbols  # flat + long(S) + short(S), no extra bins

    num_envs = max(2, int(ppo_cfg.get("num_envs", 8)))
    rollout_len = max(8, int(ppo_cfg.get("rollout_len", 64)))
    target_total = max(num_envs * rollout_len, int(steps))
    rollout_iters = max(1, target_total // (num_envs * rollout_len))
    samples_per_iter = num_envs * rollout_len
    total_samples = rollout_iters * samples_per_iter

    hidden = max(32, min(256, int(ppo_cfg.get("hidden_size", 128))))
    lr = float(ppo_cfg.get("lr", 3.0e-4))
    clip_eps = float(ppo_cfg.get("clip_eps", 0.2))
    ent_coef = float(ppo_cfg.get("ent_coef", 0.02))
    vf_coef = 0.5
    gamma = 0.99
    gae_lambda = 0.95
    ppo_epochs = max(1, int(ppo_cfg.get("ppo_epochs", 2)))
    minibatch_size = max(32, int(ppo_cfg.get("minibatch_size", 256)))

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class ActorCritic(nn.Module):
        def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            self.policy_head = nn.Linear(hidden_dim, act_dim)
            self.value_head = nn.Linear(hidden_dim, 1)

        def forward(self, obs: "torch.Tensor"):
            h = self.body(obs)
            return self.policy_head(h), self.value_head(h).squeeze(-1)

    policy = ActorCritic(obs_size, num_actions, hidden).to(device)

    # Load shared market data once.
    try:
        binding.shared(data_path=str(train_data.resolve()))
    except Exception as exc:
        return _skip(f"binding.shared() failed: {type(exc).__name__}: {exc}")

    # Allocate vector-env buffers (numpy, owned by Python).
    obs_buf = np.zeros((num_envs, obs_size), dtype=np.float32)
    act_buf = np.zeros((num_envs,), dtype=np.int32)
    rew_buf = np.zeros((num_envs,), dtype=np.float32)
    term_buf = np.zeros((num_envs,), dtype=np.uint8)
    trunc_buf = np.zeros((num_envs,), dtype=np.uint8)

    try:
        vec_handle = binding.vec_init(
            obs_buf,
            act_buf,
            rew_buf,
            term_buf,
            trunc_buf,
            num_envs,
            int(seed),
            max_steps=720,
            fee_rate=float(env_cfg.get("fee_rate", 0.001)),
            max_leverage=float(env_cfg.get("max_leverage_scalar_fallback", 1.5)),
            periods_per_year=8760.0,
            reward_scale=10.0,
            reward_clip=5.0,
            cash_penalty=0.01,
        )
        binding.vec_reset(vec_handle, int(seed))
    except Exception as exc:
        return _skip(f"vec_init/vec_reset failed: {type(exc).__name__}: {exc}")

    # ---- Rollout ---------------------------------------------------------
    # Storage on host; we move to device for SGD.
    T = rollout_iters * rollout_len
    obs_store = np.zeros((T, num_envs, obs_size), dtype=np.float32)
    act_store = np.zeros((T, num_envs), dtype=np.int64)
    logp_store = np.zeros((T, num_envs), dtype=np.float32)
    val_store = np.zeros((T, num_envs), dtype=np.float32)
    rew_store = np.zeros((T, num_envs), dtype=np.float32)
    done_store = np.zeros((T, num_envs), dtype=np.float32)

    policy.eval()
    with torch.no_grad():
        for t in range(T):
            obs_store[t] = obs_buf
            obs_t = torch.from_numpy(obs_buf).to(device)
            logits, vals = policy(obs_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample()
            logp = dist.log_prob(actions)
            act_store[t] = actions.cpu().numpy()
            logp_store[t] = logp.cpu().numpy()
            val_store[t] = vals.cpu().numpy()

            act_buf[:] = act_store[t].astype(np.int32)
            try:
                binding.vec_step(vec_handle)
            except Exception as exc:
                try:
                    binding.vec_close(vec_handle)
                except Exception:
                    pass
                return _skip(f"vec_step failed at t={t}: {exc}")
            rew_store[t] = rew_buf
            done_store[t] = (term_buf | trunc_buf).astype(np.float32)

        # Bootstrap value for GAE.
        last_vals = policy(torch.from_numpy(obs_buf).to(device))[1].cpu().numpy()

    try:
        binding.vec_close(vec_handle)
    except Exception:
        pass

    # ---- GAE -------------------------------------------------------------
    adv = np.zeros_like(rew_store)
    last_gae = np.zeros((num_envs,), dtype=np.float32)
    for t in range(T - 1, -1, -1):
        next_val = last_vals if t == T - 1 else val_store[t + 1]
        next_nonterm = 1.0 - done_store[t]
        delta = rew_store[t] + gamma * next_val * next_nonterm - val_store[t]
        last_gae = delta + gamma * gae_lambda * next_nonterm * last_gae
        adv[t] = last_gae
    returns = adv + val_store

    flat_obs = obs_store.reshape(-1, obs_size)
    flat_act = act_store.reshape(-1)
    flat_logp = logp_store.reshape(-1)
    flat_adv = adv.reshape(-1)
    flat_ret = returns.reshape(-1)
    n_samples = flat_obs.shape[0]
    if n_samples == 0:
        return _skip("rollout produced zero samples")

    # Normalise advantages.
    if flat_adv.std() > 1e-8:
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

    # Repeat the dataset ppo_epochs times so a single Trainer.train() pass
    # performs `ppo_epochs` PPO update epochs over the rollout.
    class RolloutDataset(IterableDataset):
        def __init__(self, n_epochs: int) -> None:
            self.n_epochs = n_epochs

        def __iter__(self):
            rng = np.random.default_rng(seed)
            for _ in range(self.n_epochs):
                order = rng.permutation(n_samples)
                for i in order:
                    yield {
                        "obs": flat_obs[i],
                        "action": int(flat_act[i]),
                        "old_logp": float(flat_logp[i]),
                        "advantage": float(flat_adv[i]),
                        "ret": float(flat_ret[i]),
                    }

        def __len__(self) -> int:
            return self.n_epochs * n_samples

    def collate(batch):
        return {
            "obs": torch.tensor(np.stack([b["obs"] for b in batch]), dtype=torch.float32),
            "action": torch.tensor([b["action"] for b in batch], dtype=torch.long),
            "old_logp": torch.tensor([b["old_logp"] for b in batch], dtype=torch.float32),
            "advantage": torch.tensor([b["advantage"] for b in batch], dtype=torch.float32),
            "ret": torch.tensor([b["ret"] for b in batch], dtype=torch.float32),
        }

    class PPOTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            obs = inputs["obs"].to(model.policy_head.weight.device)
            action = inputs["action"].to(obs.device)
            old_logp = inputs["old_logp"].to(obs.device)
            advantage = inputs["advantage"].to(obs.device)
            ret = inputs["ret"].to(obs.device)

            logits, value = model(obs)
            log_probs = F.log_softmax(logits, dim=-1)
            new_logp = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
            entropy = -(log_probs * log_probs.exp()).sum(dim=-1).mean()

            ratio = torch.exp(new_logp - old_logp)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(value, ret)
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            if return_outputs:
                return loss, {"policy_loss": policy_loss.detach(), "value_loss": value_loss.detach()}
            return loss

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=minibatch_size,
        learning_rate=lr,
        max_steps=max(1, math.ceil(ppo_epochs * n_samples / minibatch_size)),
        logging_steps=max(1, math.ceil(n_samples / minibatch_size / 4)),
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=int(seed),
        fp16=False,
        bf16=False,
    )

    dataset = RolloutDataset(n_epochs=1)  # max_steps already governs total updates

    trainer = PPOTrainer(
        model=policy,
        args=args,
        train_dataset=dataset,
        data_collator=collate,
    )
    try:
        train_out = trainer.train()
    except Exception as exc:
        return {"status": "error", "reason": f"trainer.train() failed: {type(exc).__name__}: {exc}"}

    # Save a minimal checkpoint so the bench harness can find a *.pt file.
    try:
        torch.save({"model": policy.state_dict(),
                    "obs_dim": obs_size,
                    "action_dim": num_actions,
                    "hidden": hidden},
                   ckpt_dir / "final.pt")
    except Exception:
        pass

    # Sortino on the rollout rewards (per-env episode return -> ratio).
    ep_ret = rew_store.sum(axis=0)  # (num_envs,)
    mean_ret = float(ep_ret.mean())
    downside = ep_ret[ep_ret < 0.0]
    downside_std = float(downside.std()) if downside.size > 0 else 0.0
    sortino = mean_ret / downside_std if downside_std > 1e-9 else float(mean_ret)
    if not math.isfinite(sortino):
        sortino = 0.0

    metrics = {
        "status": "ok",
        "trainer": "hf_trainer",
        "samples": int(n_samples),
        "rollout_iters": int(rollout_iters),
        "num_envs": int(num_envs),
        "rollout_len": int(rollout_len),
        "ppo_epochs": int(ppo_epochs),
        "minibatch_size": int(minibatch_size),
        "obs_dim": int(obs_size),
        "action_dim": int(num_actions),
        "hidden": int(hidden),
        "mean_episode_return": mean_ret,
        "sortino": sortino,
        "train_loss": float(getattr(train_out, "training_loss", 0.0) or 0.0),
        "global_step": int(getattr(train_out, "global_step", 0) or 0),
        "transformers_version": transformers.__version__,
    }
    return metrics
