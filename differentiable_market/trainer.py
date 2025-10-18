from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import torch
from torch.distributions import Dirichlet
from torch.nn.utils import clip_grad_norm_

from .config import DataConfig, EnvironmentConfig, EvaluationConfig, TrainingConfig
from .data import load_aligned_ohlc, log_data_preview, split_train_eval
from .env import DifferentiableMarketEnv, smooth_abs
from .features import ohlc_to_features
from .losses import dirichlet_kl
from .policy import DirichletGRUPolicy
from .optim import MuonConfig, build_muon_optimizer
from .utils import append_jsonl, ensure_dir, resolve_device, resolve_dtype, set_seed


@dataclass(slots=True)
class TrainingState:
    step: int = 0
    best_eval_loss: float = math.inf
    best_step: int = -1


class DifferentiableMarketTrainer:
    def __init__(
        self,
        data_cfg: DataConfig,
        env_cfg: EnvironmentConfig,
        train_cfg: TrainingConfig,
        eval_cfg: EvaluationConfig | None = None,
    ):
        self.data_cfg = data_cfg
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg or EvaluationConfig()

        set_seed(train_cfg.seed)
        self.device = resolve_device(train_cfg.device)
        self.dtype = resolve_dtype(train_cfg.dtype, self.device)
        self.autocast_enabled = self.device.type == "cuda" and train_cfg.bf16_autocast

        # Load data
        ohlc_all, symbols, index = load_aligned_ohlc(data_cfg)
        self.symbols = symbols
        self.index = index

        train_tensor, eval_tensor = split_train_eval(ohlc_all)
        self.train_features, self.train_returns = ohlc_to_features(train_tensor)
        self.eval_features, self.eval_returns = ohlc_to_features(eval_tensor)

        if self.train_features.shape[0] <= train_cfg.lookback:
            raise ValueError("Training data shorter than lookback window")
        if self.eval_features.shape[0] <= train_cfg.lookback // 2:
            raise ValueError("Evaluation data insufficient for validation")

        self.asset_count = self.train_features.shape[1]
        self.feature_dim = self.train_features.shape[2]

        self.env = DifferentiableMarketEnv(env_cfg)

        self.policy = DirichletGRUPolicy(
            n_assets=self.asset_count,
            feature_dim=self.feature_dim,
            gradient_checkpointing=train_cfg.gradient_checkpointing,
        ).to(self.device)

        self.ref_policy = DirichletGRUPolicy(
            n_assets=self.asset_count,
            feature_dim=self.feature_dim,
            gradient_checkpointing=False,
        ).to(self.device)
        self.ref_policy.load_state_dict(self.policy.state_dict())
        for param in self.ref_policy.parameters():
            param.requires_grad_(False)

        self.optimizer = self._make_optimizer()

        self.state = TrainingState()
        self.run_dir = self._prepare_run_dir()
        self.ckpt_dir = ensure_dir(self.run_dir / "checkpoints")
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._write_config_snapshot(log_data_preview(ohlc_all, symbols, index))

        self._train_step_impl = self._build_train_step()
        if train_cfg.use_compile and hasattr(torch, "compile"):
            self._train_step = torch.compile(self._train_step_impl, mode=train_cfg.torch_compile_mode)
        else:
            self._train_step = self._train_step_impl

    def fit(self) -> TrainingState:
        for step in range(self.train_cfg.epochs):
            train_stats = self._train_step()
            self.state.step = step + 1
            train_payload = {"phase": "train", "step": step}
            train_payload.update(train_stats)
            append_jsonl(self.metrics_path, train_payload)
            if (
                self.train_cfg.eval_interval > 0
                and (step % self.train_cfg.eval_interval == 0 or step == self.train_cfg.epochs - 1)
            ):
                eval_stats = self.evaluate()
                eval_payload = {"phase": "eval", "step": step}
                eval_payload.update(eval_stats)
                append_jsonl(self.metrics_path, eval_payload)
                eval_loss = -eval_stats["eval_objective"]
                self._update_checkpoints(eval_loss, step, eval_stats)
            if step % 50 == 0:
                print(
                    f"[step {step}] loss={train_stats['loss']:.4f} "
                    f"reward_mean={train_stats['reward_mean']:.4f} kl={train_stats['kl']:.4f}"
                )
        return self.state

    def evaluate(self) -> Dict[str, float]:
        self.policy.eval()
        features = self.eval_features.unsqueeze(0).to(self.device, dtype=self.dtype)
        returns = self.eval_returns.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            alpha = self.policy(features).float().squeeze(0)
            weights = alpha / alpha.sum(dim=-1, keepdim=True)

        w_prev = torch.full(
            (self.asset_count,),
            1.0 / self.asset_count,
            device=self.device,
            dtype=torch.float32,
        )
        rewards = []
        turnovers = []
        steps = weights.shape[0]
        for t in range(steps):
            w_t = weights[t].to(torch.float32)
            r_next = returns[t]
            reward = self.env.step(w_t, r_next, w_prev)
            rewards.append(reward)
            turnovers.append(smooth_abs(w_t - w_prev, self.env_cfg.smooth_abs_eps).sum())
            w_prev = w_t
        reward_tensor = torch.stack(rewards)
        turnover_tensor = torch.stack(turnovers)

        objective = self.env.aggregate_rewards(reward_tensor)
        mean_reward = reward_tensor.mean()
        std_reward = reward_tensor.std(unbiased=False).clamp_min(1e-8)
        sharpe = mean_reward / std_reward

        metrics = {
            "eval_objective": float(objective.item()),
            "eval_mean_reward": float(mean_reward.item()),
            "eval_std_reward": float(std_reward.item()),
            "eval_turnover": float(turnover_tensor.mean().item()),
            "eval_sharpe": float(sharpe.item()),
            "eval_steps": int(steps),
        }
        return metrics

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _prepare_run_dir(self) -> Path:
        base = ensure_dir(self.train_cfg.save_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return ensure_dir(base / timestamp)

    def _write_config_snapshot(self, data_preview: Dict[str, object]) -> None:
        config_payload = {
            "data": self._serialize_config(self.data_cfg),
            "env": self._serialize_config(self.env_cfg),
            "train": self._serialize_config(self.train_cfg),
            "eval": self._serialize_config(self.eval_cfg),
            "preview": data_preview,
            "symbols": self.symbols,
        }
        config_path = self.run_dir / "config.json"
        config_path.write_text(json.dumps(config_payload, indent=2))

    def _serialize_config(self, cfg) -> Dict[str, object]:
        raw = asdict(cfg)
        for key, value in raw.items():
            if isinstance(value, Path):
                raw[key] = str(value)
        return raw

    def _make_optimizer(self):
        params = list(self.policy.named_parameters())
        muon_params = []
        aux_params = []
        other_params = []
        for name, param in params:
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and ("gru" in name or "head" in name):
                muon_params.append(param)
            elif "gru" in name:
                aux_params.append(param)
            else:
                other_params.append(param)

        if self.train_cfg.use_muon:
            muon_opt = build_muon_optimizer(
                muon_params,
                aux_params + other_params,
                MuonConfig(
                    lr_muon=self.train_cfg.lr_muon,
                    lr_adamw=self.train_cfg.lr_adamw,
                    weight_decay=self.train_cfg.weight_decay,
                    betas=(0.9, 0.95),
                    momentum=0.95,
                    ns_steps=5,
                ),
            )
            if muon_opt is not None:
                return muon_opt
            else:
                print("Muon backend unavailable; falling back to AdamW.")

        return torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.train_cfg.lr_adamw,
            betas=(0.9, 0.95),
            weight_decay=self.train_cfg.weight_decay,
        )

    def _sample_windows(self) -> tuple[torch.Tensor, torch.Tensor]:
        L = self.train_cfg.lookback
        B = self.train_cfg.batch_windows
        max_start = self.train_features.shape[0] - L
        if max_start <= 1:
            raise ValueError("Training window length exceeds dataset")
        start_indices = torch.randint(0, max_start, (B,))

        x_windows = []
        r_windows = []
        for start in start_indices.tolist():
            x = self.train_features[start : start + L]
            r = self.train_returns[start : start + L]
            x_windows.append(x.unsqueeze(0))
            r_windows.append(r.unsqueeze(0))
        x_batch = torch.cat(x_windows, dim=0).contiguous()
        r_batch = torch.cat(r_windows, dim=0).contiguous()
        return x_batch, r_batch

    def _rollout_group(
        self,
        alpha: torch.Tensor,
        returns: torch.Tensor,
        w0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K = self.train_cfg.rollout_groups
        B, T, A = alpha.shape
        rewards = []
        log_probs = []
        entropies = []

        for _ in range(K):
            dist = Dirichlet(alpha)
            w_seq = dist.rsample()
            logp = dist.log_prob(w_seq).sum(dim=1)  # [B]
            entropy = dist.entropy().mean(dim=1)  # [B]

            w_prev = w0
            step_rewards = []
            for t in range(T):
                w_t = w_seq[:, t, :].to(torch.float32)
                r_next = returns[:, t, :]
                reward = self.env.step(w_t, r_next, w_prev)
                step_rewards.append(reward)
                w_prev = w_seq[:, t, :]
            episode_reward = torch.stack(step_rewards, dim=1).sum(dim=1)
            rewards.append(episode_reward)
            log_probs.append(logp)
            entropies.append(entropy)

        return (
            torch.stack(rewards, dim=1),
            torch.stack(log_probs, dim=1),
            torch.stack(entropies, dim=1),
        )

    def _build_train_step(self):
        def train_step():
            self.policy.train()
            self.optimizer.zero_grad(set_to_none=True)

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            x_batch_cpu, r_batch_cpu = self._sample_windows()
            total_windows = x_batch_cpu.shape[0]
            micro = self.train_cfg.microbatch_windows or total_windows
            micro = max(1, min(micro, total_windows))
            accum_steps = math.ceil(total_windows / micro)

            loss_total = 0.0
            policy_total = 0.0
            entropy_total = 0.0
            kl_total = 0.0
            reward_sum = 0.0
            reward_sq_sum = 0.0
            reward_count = 0
            chunks = 0

            for start in range(0, total_windows, micro):
                end = start + micro
                x_micro = x_batch_cpu[start:end].to(self.device, dtype=self.dtype, non_blocking=True)
                r_micro = r_batch_cpu[start:end].to(self.device, dtype=torch.float32, non_blocking=True)
                Bm = x_micro.shape[0]
                w0 = torch.full((Bm, self.asset_count), 1.0 / self.asset_count, device=self.device, dtype=torch.float32)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.autocast_enabled,
                ):
                    alpha = self.policy(x_micro).float()
                    rewards, logp, entropy = self._rollout_group(alpha, r_micro, w0)
                    baseline = rewards.mean(dim=1, keepdim=True)
                    advantages = rewards - baseline
                    advantages = advantages / (advantages.std(dim=1, keepdim=True) + 1e-6)

                    policy_loss = -(advantages.detach() * logp).mean()
                    entropy_scalar = entropy.mean()
                    entropy_bonus = -self.train_cfg.entropy_coef * entropy_scalar

                    with torch.no_grad():
                        alpha_ref = self.ref_policy(x_micro).float()
                    kl = dirichlet_kl(alpha, alpha_ref).mean()
                    kl_term = self.train_cfg.kl_coef * kl

                    loss_unscaled = policy_loss + entropy_bonus + kl_term

                (loss_unscaled / accum_steps).backward()

                loss_total += loss_unscaled.detach().item()
                policy_total += policy_loss.detach().item()
                entropy_total += entropy_scalar.detach().item()
                kl_total += kl.detach().item()

                rewards_cpu = rewards.detach().cpu()
                reward_sum += rewards_cpu.sum().item()
                reward_sq_sum += rewards_cpu.pow(2).sum().item()
                reward_count += rewards_cpu.numel()
                chunks += 1

            clip_grad_norm_(self.policy.parameters(), self.train_cfg.grad_clip)
            self.optimizer.step()

            with torch.no_grad():
                ema = 0.95
                for ref_param, pol_param in zip(self.ref_policy.parameters(), self.policy.parameters()):
                    ref_param.data.lerp_(pol_param.data, 1 - ema)

            peak_mem_gb = 0.0
            if self.device.type == "cuda":
                peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                torch.cuda.reset_peak_memory_stats(self.device)

            reward_mean = reward_sum / max(reward_count, 1)
            reward_var = max(reward_sq_sum / max(reward_count, 1) - reward_mean ** 2, 0.0)
            reward_std = reward_var ** 0.5

            avg = lambda total: total / max(chunks, 1)

            return {
                "loss": avg(loss_total),
                "policy": avg(policy_total),
                "entropy": avg(entropy_total),
                "kl": avg(kl_total),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "peak_mem_gb": peak_mem_gb,
                "microbatch": micro,
                "windows": total_windows,
            }

        return train_step

    def _update_checkpoints(self, eval_loss: float, step: int, eval_stats: Dict[str, float]) -> None:
        latest_path = self.ckpt_dir / "latest.pt"
        best_path = self.ckpt_dir / "best.pt"
        payload = {
            "step": step,
            "eval_loss": eval_loss,
            "policy_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": {
                "data": self._serialize_config(self.data_cfg),
                "env": self._serialize_config(self.env_cfg),
                "train": self._serialize_config(self.train_cfg),
                "eval": self._serialize_config(self.eval_cfg),
            },
            "symbols": self.symbols,
            "metrics": eval_stats,
        }
        torch.save(payload, latest_path)
        if eval_loss < self.state.best_eval_loss:
            torch.save(payload, best_path)
            self.state.best_eval_loss = eval_loss
            self.state.best_step = step
            print(f"[step {step}] new best eval loss {eval_loss:.4f}")
