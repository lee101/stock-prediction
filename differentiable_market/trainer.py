from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
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
from .differentiable_utils import (
    TradeMemoryState,
    augment_market_features,
    risk_budget_mismatch,
    soft_drawdown,
    trade_memory_update,
)
from wandboard import WandBoardLogger


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
        train_len = train_tensor.shape[0]
        eval_len = eval_tensor.shape[0]
        self.train_index = index[:train_len]
        self.eval_index = index[train_len : train_len + eval_len]
        self.eval_periods_per_year = self._estimate_periods_per_year(self.eval_index)
        add_cash = self.train_cfg.include_cash or self.data_cfg.include_cash
        self.train_features, self.train_returns = self._build_features(train_tensor, add_cash=add_cash, phase="train")
        self.eval_features, self.eval_returns = self._build_features(eval_tensor, add_cash=add_cash, phase="eval")

        if self.train_features.shape[0] <= train_cfg.lookback:
            raise ValueError("Training data shorter than lookback window")
        if self.eval_features.shape[0] <= train_cfg.lookback // 2:
            raise ValueError("Evaluation data insufficient for validation")

        self.asset_count = self.train_features.shape[1]
        self.feature_dim = self.train_features.shape[2]

        self.env = DifferentiableMarketEnv(env_cfg)
        self.asset_names: List[str] = list(self.symbols)
        if add_cash and (not self.asset_names or self.asset_names[-1] != "CASH"):
            self.asset_names = self.asset_names + ["CASH"]
        if self.asset_names:
            self.env.set_asset_universe(self.asset_names)
        self.env.reset()

        if self.train_cfg.risk_budget_target:
            if len(self.train_cfg.risk_budget_target) != self.asset_count:
                raise ValueError(
                    f"risk_budget_target length {len(self.train_cfg.risk_budget_target)} "
                    f"does not match asset_count {self.asset_count}"
                )
            self.risk_budget_target = torch.tensor(
                self.train_cfg.risk_budget_target,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self.risk_budget_target = None

        self.trade_memory_state: TradeMemoryState | None = None

        self.policy = DirichletGRUPolicy(
            n_assets=self.asset_count,
            feature_dim=self.feature_dim,
            gradient_checkpointing=train_cfg.gradient_checkpointing,
            enable_shorting=train_cfg.enable_shorting,
            max_intraday_leverage=train_cfg.max_intraday_leverage,
            max_overnight_leverage=train_cfg.max_overnight_leverage,
        ).to(self.device)

        self.ref_policy = DirichletGRUPolicy(
            n_assets=self.asset_count,
            feature_dim=self.feature_dim,
            gradient_checkpointing=False,
            enable_shorting=train_cfg.enable_shorting,
            max_intraday_leverage=train_cfg.max_intraday_leverage,
            max_overnight_leverage=train_cfg.max_overnight_leverage,
        ).to(self.device)
        self.ref_policy.load_state_dict(self.policy.state_dict())
        for param in self.ref_policy.parameters():
            param.requires_grad_(False)

        self.init_checkpoint: Path | None = None
        self._init_eval_loss: float | None = None
        if train_cfg.init_checkpoint is not None:
            ckpt_path = Path(train_cfg.init_checkpoint)
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint.get("policy_state")
            if state_dict is None:
                raise ValueError(f"Checkpoint {ckpt_path} missing 'policy_state'")
            current_state = self.policy.state_dict()
            incompatible_keys = [
                key
                for key, tensor in state_dict.items()
                if key in current_state and tensor.shape != current_state[key].shape
            ]
            for key in incompatible_keys:
                state_dict.pop(key, None)
            missing, unexpected = self.policy.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                allowed_mismatch = {"head.weight", "head.bias", "alpha_bias"}
                filtered_missing = [name for name in missing if name not in allowed_mismatch]
                filtered_unexpected = [name for name in unexpected if name not in allowed_mismatch]
                if filtered_missing or filtered_unexpected:
                    raise ValueError(
                        f"Checkpoint {ckpt_path} incompatible with policy. "
                        f"Missing keys: {filtered_missing or 'None'}, unexpected: {filtered_unexpected or 'None'}"
                    )
                else:
                    print(
                        f"Loaded checkpoint {ckpt_path} with partial head initialisation "
                        f"(enable_shorting={self.train_cfg.enable_shorting})."
                    )
            self.ref_policy.load_state_dict(self.policy.state_dict())
            eval_loss = checkpoint.get("eval_loss")
            if isinstance(eval_loss, (float, int)):
                self._init_eval_loss = float(eval_loss)
            self.init_checkpoint = ckpt_path
            print(f"Loaded policy weights from {ckpt_path}")

        self.optimizer = self._make_optimizer()

        self.state = TrainingState()
        if self._init_eval_loss is not None:
            self.state.best_eval_loss = min(self.state.best_eval_loss, self._init_eval_loss)
        self.run_dir = self._prepare_run_dir()
        self.ckpt_dir = ensure_dir(self.run_dir / "checkpoints")
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._write_config_snapshot(log_data_preview(ohlc_all, symbols, index))
        self.metrics_logger = self._init_metrics_logger()
        self.best_k = max(1, int(self.train_cfg.best_k_checkpoints))
        self._topk_records: List[Dict[str, Any]] = []
        self.topk_index_path = self.run_dir / "topk_checkpoints.json"

        self._augmented_losses = (
            self.train_cfg.soft_drawdown_lambda > 0.0
            or self.train_cfg.risk_budget_lambda > 0.0
            or self.train_cfg.trade_memory_lambda > 0.0
        )

        self._train_step_impl = self._build_train_step()
        self._train_step = self._train_step_impl
        if train_cfg.use_compile and hasattr(torch, "compile"):
            try:
                self._train_step = torch.compile(self._train_step_impl, mode=train_cfg.torch_compile_mode)
            except RuntimeError as exc:
                reason = "augmented losses" if self._augmented_losses else "torch runtime"
                print(f"torch.compile fallback ({reason}): {exc}")
                self._train_step = self._train_step_impl

    def _build_features(
        self,
        ohlc_tensor: torch.Tensor,
        add_cash: bool,
        phase: Literal["train", "eval"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct feature and return tensors for the requested phase."""
        del phase  # Default implementation does not distinguish between phases.
        features, forward_returns = ohlc_to_features(ohlc_tensor, add_cash=add_cash)
        features = augment_market_features(
            features,
            forward_returns,
            use_taylor=self.train_cfg.use_taylor_features,
            taylor_order=self.train_cfg.taylor_order,
            taylor_scale=self.train_cfg.taylor_scale,
            use_wavelet=self.train_cfg.use_wavelet_features,
            wavelet_levels=self.train_cfg.wavelet_levels,
            padding_mode=self.train_cfg.wavelet_padding_mode,
        ).contiguous()
        return features, forward_returns.contiguous()

    def fit(self) -> TrainingState:
        try:
            for step in range(self.train_cfg.epochs):
                train_stats = self._train_step()
                self.state.step = step + 1
                train_payload = {"phase": "train", "step": step}
                train_payload.update(train_stats)
                append_jsonl(self.metrics_path, train_payload)
                self._log_metrics("train", self.state.step, train_stats, commit=False)
                if (
                    self.train_cfg.eval_interval > 0
                    and (step % self.train_cfg.eval_interval == 0 or step == self.train_cfg.epochs - 1)
                ):
                    eval_stats = self.evaluate()
                    eval_payload = {"phase": "eval", "step": step}
                    eval_payload.update(eval_stats)
                    append_jsonl(self.metrics_path, eval_payload)
                    self._log_metrics("eval", self.state.step, eval_stats, commit=True)
                    eval_loss = -eval_stats["eval_objective"]
                    self._update_checkpoints(eval_loss, step, eval_stats)
                if step % 50 == 0:
                    print(
                        f"[step {step}] loss={train_stats['loss']:.4f} "
                        f"reward_mean={train_stats['reward_mean']:.4f} kl={train_stats['kl']:.4f}"
                    )
        finally:
            self._finalize_logging()
        return self.state

    def evaluate(self) -> Dict[str, float]:
        self.policy.eval()
        features = self.eval_features.unsqueeze(0).to(self.device, dtype=self.dtype)
        returns = self.eval_returns.to(self.device, dtype=torch.float32)

        self.env.reset()

        with torch.no_grad():
            alpha = self.policy(features).float()
            weights_seq, overnight_seq = self.policy.decode_concentration(alpha)

        weights = weights_seq.squeeze(0)
        overnight_weights = overnight_seq.squeeze(0)

        if self.train_cfg.enable_shorting:
            w_prev = torch.zeros(
                (self.asset_count,),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            w_prev = torch.full(
                (self.asset_count,),
                1.0 / self.asset_count,
                device=self.device,
                dtype=torch.float32,
            )
        rewards = []
        gross_returns = []
        turnovers = []
        gross_leverages = []
        overnight_leverages = []
        steps = weights.shape[0]
        for t in range(steps):
            w_t = weights[t].to(torch.float32)
            r_next = returns[t]
            gross = torch.dot(w_t, r_next)
            reward = self.env.step(w_t, r_next, w_prev)
            rewards.append(reward)
            gross_returns.append(gross)
            turnovers.append(smooth_abs(w_t - w_prev, self.env_cfg.smooth_abs_eps).sum())
            gross_leverages.append(w_t.abs().sum())
            overnight_leverages.append(overnight_weights[t].abs().sum())
            w_prev = overnight_weights[t].to(torch.float32)
        if steps == 0:
            metrics = {
                "eval_objective": 0.0,
                "eval_mean_reward": 0.0,
                "eval_std_reward": 0.0,
                "eval_turnover": 0.0,
                "eval_sharpe": 0.0,
                "eval_steps": 0,
                "eval_total_return": 0.0,
                "eval_annual_return": 0.0,
                "eval_total_return_gross": 0.0,
                "eval_annual_return_gross": 0.0,
                "eval_max_drawdown": 0.0,
                "eval_final_wealth": 1.0,
                "eval_final_wealth_gross": 1.0,
                "eval_periods_per_year": float(self.eval_periods_per_year),
                "eval_trading_pnl": 0.0,
                "eval_gross_leverage_mean": 0.0,
                "eval_gross_leverage_max": 0.0,
                "eval_overnight_leverage_max": 0.0,
            }
            self.policy.train()
            return metrics

        reward_tensor = torch.stack(rewards)
        gross_tensor = torch.stack(gross_returns)
        turnover_tensor = torch.stack(turnovers)
        gross_leverage_tensor = torch.stack(gross_leverages)
        overnight_leverage_tensor = torch.stack(overnight_leverages)

        objective = self.env.aggregate_rewards(reward_tensor)
        mean_reward = reward_tensor.mean()
        std_reward = reward_tensor.std(unbiased=False).clamp_min(1e-8)
        sharpe = mean_reward / std_reward

        total_log_net = reward_tensor.sum().item()
        total_log_gross = gross_tensor.sum().item()
        total_return_net = float(math.expm1(total_log_net))
        total_return_gross = float(math.expm1(total_log_gross))
        mean_log_net = mean_reward.item()
        mean_log_gross = gross_tensor.mean().item()
        annual_return_net = self._annualise_from_log(mean_log_net, self.eval_periods_per_year)
        annual_return_gross = self._annualise_from_log(mean_log_gross, self.eval_periods_per_year)

        net_cumulative = reward_tensor.cumsum(dim=0)
        gross_cumulative = gross_tensor.cumsum(dim=0)
        wealth_net = torch.exp(net_cumulative)
        wealth_gross = torch.exp(gross_cumulative)
        running_max, _ = torch.cummax(wealth_net, dim=0)
        drawdowns = (running_max - wealth_net) / running_max.clamp_min(1e-12)
        max_drawdown = float(drawdowns.max().item())

        metrics = {
            "eval_objective": float(objective.item()),
            "eval_mean_reward": float(mean_reward.item()),
            "eval_std_reward": float(std_reward.item()),
            "eval_turnover": float(turnover_tensor.mean().item()),
            "eval_sharpe": float(sharpe.item()),
            "eval_steps": int(steps),
            "eval_total_return": total_return_net,
            "eval_total_return_gross": total_return_gross,
            "eval_annual_return": annual_return_net,
            "eval_annual_return_gross": annual_return_gross,
            "eval_max_drawdown": max_drawdown,
            "eval_final_wealth": float(wealth_net[-1].item()),
            "eval_final_wealth_gross": float(wealth_gross[-1].item()),
            "eval_periods_per_year": float(self.eval_periods_per_year),
            "eval_trading_pnl": total_return_net,
            "eval_gross_leverage_mean": float(gross_leverage_tensor.mean().item()),
            "eval_gross_leverage_max": float(gross_leverage_tensor.max().item()),
            "eval_overnight_leverage_max": float(overnight_leverage_tensor.max().item()),
        }
        self.policy.train()
        return metrics

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _prepare_run_dir(self) -> Path:
        base = ensure_dir(self.train_cfg.save_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return ensure_dir(base / timestamp)

    def _estimate_periods_per_year(self, index: Sequence[pd.Timestamp]) -> float:
        if isinstance(index, pd.DatetimeIndex):
            datetimes = index
        else:
            datetimes = pd.DatetimeIndex(index)
        if len(datetimes) < 2:
            return 252.0
        values = datetimes.asi8.astype(np.float64)
        diffs = np.diff(values)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return 252.0
        avg_ns = float(diffs.mean())
        if not math.isfinite(avg_ns) or avg_ns <= 0.0:
            return 252.0
        seconds_per_period = avg_ns / 1e9
        if seconds_per_period <= 0.0:
            return 252.0
        seconds_per_year = 365.25 * 24 * 3600
        return float(seconds_per_year / seconds_per_period)

    @staticmethod
    def _annualise_from_log(mean_log_return: float, periods_per_year: float) -> float:
        if not math.isfinite(mean_log_return) or not math.isfinite(periods_per_year) or periods_per_year <= 0.0:
            return float("nan")
        return float(math.expm1(mean_log_return * periods_per_year))

    def _remove_topk_step(self, step: int) -> None:
        for idx, record in enumerate(list(self._topk_records)):
            if int(record.get("step", -1)) == int(step):
                path_str = record.get("path")
                if isinstance(path_str, str):
                    path = Path(path_str)
                    if not path.is_absolute():
                        path = self.run_dir / path
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                self._topk_records.pop(idx)
                break

    def _update_topk(self, eval_loss: float, step: int, payload: Dict[str, Any]) -> None:
        if self.best_k <= 0:
            return
        if self._topk_records and len(self._topk_records) >= self.best_k:
            worst_loss = float(self._topk_records[-1]["loss"])
            if eval_loss >= worst_loss:
                return
        self._remove_topk_step(step)
        ckpt_name = f"best_step{step:06d}_loss{eval_loss:.6f}.pt"
        ckpt_path = self.ckpt_dir / ckpt_name
        torch.save(payload, ckpt_path)
        try:
            relative_path = ckpt_path.relative_to(self.run_dir)
            path_str = str(relative_path)
        except ValueError:
            path_str = str(ckpt_path)
        record = {
            "loss": float(eval_loss),
            "step": int(step),
            "path": path_str,
        }
        self._topk_records.append(record)
        self._topk_records.sort(key=lambda item: float(item["loss"]))
        while len(self._topk_records) > self.best_k:
            removed = self._topk_records.pop(-1)
            path_str = removed.get("path")
            if isinstance(path_str, str):
                path = Path(path_str)
                if not path.is_absolute():
                    path = self.run_dir / path
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        for rank, rec in enumerate(self._topk_records, start=1):
            rec["rank"] = rank
        try:
            self.topk_index_path.write_text(json.dumps(self._topk_records, indent=2))
        except Exception as exc:
            print(f"Failed to update top-k checkpoint index: {exc}")

    def _init_metrics_logger(self) -> Optional[WandBoardLogger]:
        enable_tb = self.train_cfg.tensorboard_root is not None
        enable_wandb = self.train_cfg.use_wandb
        if not (enable_tb or enable_wandb):
            return None
        log_dir = self.train_cfg.tensorboard_root
        tb_subdir = self.train_cfg.tensorboard_subdir
        if not tb_subdir:
            tb_subdir = str(Path("differentiable_market") / self.run_dir.name)
        run_name = self.train_cfg.wandb_run_name or f"differentiable_market_{self.run_dir.name}"
        config_payload = getattr(self, "_config_snapshot", None)
        try:
            logger = WandBoardLogger(
                run_name=run_name,
                project=self.train_cfg.wandb_project,
                entity=self.train_cfg.wandb_entity,
                tags=self.train_cfg.wandb_tags if self.train_cfg.wandb_tags else None,
                group=self.train_cfg.wandb_group,
                notes=self.train_cfg.wandb_notes,
                mode=self.train_cfg.wandb_mode,
                enable_wandb=enable_wandb,
                log_dir=log_dir,
                tensorboard_subdir=tb_subdir,
                config=config_payload,
                settings=self.train_cfg.wandb_settings or None,
                log_metrics=self.train_cfg.wandb_log_metrics,
                metric_log_level=self.train_cfg.wandb_metric_log_level,
            )
        except Exception as exc:
            print(f"[differentiable_market] Failed to initialise WandBoardLogger: {exc}")
            return None
        return logger

    def _log_metrics(self, phase: str, step: int, stats: Dict[str, object], *, commit: bool) -> None:
        logger = getattr(self, "metrics_logger", None)
        if logger is None:
            return
        payload: Dict[str, object] = {}
        for key, value in stats.items():
            metric_name = key
            prefix = f"{phase}_"
            if metric_name.startswith(prefix):
                metric_name = metric_name[len(prefix) :]
            name = f"{phase}/{metric_name}"
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    payload[name] = value.item()
                continue
            payload[name] = value
        if payload:
            logger.log(payload, step=step, commit=commit)

    def _finalize_logging(self) -> None:
        logger = getattr(self, "metrics_logger", None)
        if logger is None:
            return
        if self._topk_records:
            topk_metrics = {
                f"run/topk_loss_{int(rec.get('rank', idx + 1))}": float(rec["loss"])
                for idx, rec in enumerate(self._topk_records)
            }
            logger.log(topk_metrics, step=self.state.step, commit=False)
        summary: Dict[str, object] = {"run/epochs_completed": self.state.step}
        if math.isfinite(self.state.best_eval_loss):
            summary["run/best_eval_loss"] = self.state.best_eval_loss
        if self.state.best_step >= 0:
            summary["run/best_eval_step"] = self.state.best_step
        if summary:
            logger.log(summary, step=self.state.step, commit=True)
        logger.flush()
        logger.finish()
        self.metrics_logger = None

    def close(self) -> None:
        self._finalize_logging()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    def _write_config_snapshot(self, data_preview: Dict[str, object]) -> None:
        config_payload = {
            "data": self._serialize_config(self.data_cfg),
            "env": self._serialize_config(self.env_cfg),
            "train": self._serialize_config(self.train_cfg),
            "eval": self._serialize_config(self.eval_cfg),
            "preview": data_preview,
            "symbols": self.symbols,
        }
        self._config_snapshot = config_payload
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        K = self.train_cfg.rollout_groups
        B, T, A = alpha.shape
        rewards = []
        log_probs = []
        entropies = []
        reward_traces = []
        weight_traces = []

        self.env.reset()

        for _ in range(K):
            dist = Dirichlet(alpha)
            alloc_seq = dist.rsample()
            logp = dist.log_prob(alloc_seq).sum(dim=1)  # [B]
            entropy = dist.entropy().mean(dim=1)  # [B]

            intraday_seq, overnight_seq = self.policy.allocations_to_weights(alloc_seq)
            w_prev = w0
            step_rewards = []
            for t in range(T):
                w_t = intraday_seq[:, t, :].to(torch.float32)
                r_next = returns[:, t, :]
                reward = self.env.step(w_t, r_next, w_prev)
                step_rewards.append(reward)
                w_prev = overnight_seq[:, t, :].to(torch.float32)
            reward_seq = torch.stack(step_rewards, dim=1)
            rewards.append(reward_seq.sum(dim=1))
            log_probs.append(logp)
            entropies.append(entropy)
            reward_traces.append(reward_seq)
            weight_traces.append(intraday_seq)

        return (
            torch.stack(rewards, dim=1),
            torch.stack(log_probs, dim=1),
            torch.stack(entropies, dim=1),
            torch.stack(reward_traces, dim=0),
            torch.stack(weight_traces, dim=0),
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
            drawdown_total = 0.0
            risk_total = 0.0
            trade_total = 0.0
            reward_sum = 0.0
            reward_sq_sum = 0.0
            reward_count = 0
            chunks = 0

            for start in range(0, total_windows, micro):
                end = start + micro
                x_micro = x_batch_cpu[start:end].to(self.device, dtype=self.dtype, non_blocking=True)
                r_micro = r_batch_cpu[start:end].to(self.device, dtype=torch.float32, non_blocking=True)
                Bm = x_micro.shape[0]
                if self.train_cfg.enable_shorting:
                    w0 = torch.zeros((Bm, self.asset_count), device=self.device, dtype=torch.float32)
                else:
                    w0 = torch.full(
                        (Bm, self.asset_count),
                        1.0 / self.asset_count,
                        device=self.device,
                        dtype=torch.float32,
                    )

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.autocast_enabled,
                ):
                    alpha = self.policy(x_micro).float()
                    rewards, logp, entropy, reward_traces, weight_traces = self._rollout_group(alpha, r_micro, w0)
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

                    if self.train_cfg.soft_drawdown_lambda > 0.0:
                        reward_seq_mean = reward_traces.mean(dim=0)  # [B, T]
                        _, drawdown = soft_drawdown(reward_seq_mean)
                        drawdown_penalty = drawdown.max(dim=-1).values.mean()
                        loss_unscaled = loss_unscaled + self.train_cfg.soft_drawdown_lambda * drawdown_penalty
                    else:
                        drawdown_penalty = torch.zeros((), device=self.device, dtype=torch.float32)

                    if self.train_cfg.risk_budget_lambda > 0.0 and self.risk_budget_target is not None:
                        ret_flat = r_micro.reshape(-1, self.asset_count)
                        if ret_flat.shape[0] > 1:
                            ret_centered = ret_flat - ret_flat.mean(dim=0, keepdim=True)
                            cov = (ret_centered.T @ ret_centered) / (ret_flat.shape[0] - 1)
                        else:
                            cov = torch.eye(self.asset_count, device=self.device, dtype=torch.float32)
                        weight_avg = weight_traces.mean(dim=0).mean(dim=1)
                        risk_penalty = risk_budget_mismatch(weight_avg, cov, self.risk_budget_target)
                        loss_unscaled = loss_unscaled + self.train_cfg.risk_budget_lambda * risk_penalty
                    else:
                        risk_penalty = torch.zeros((), device=self.device, dtype=torch.float32)

                    if self.train_cfg.trade_memory_lambda > 0.0:
                        pnl_vector = rewards.mean(dim=0)
                        tm_state, regret_signal, _ = trade_memory_update(
                            self.trade_memory_state,
                            pnl_vector,
                            ema_decay=self.train_cfg.trade_memory_ema_decay,
                        )
                        trade_penalty = regret_signal.mean()
                        loss_unscaled = loss_unscaled + self.train_cfg.trade_memory_lambda * trade_penalty
                        self.trade_memory_state = TradeMemoryState(
                            ema_pnl=tm_state.ema_pnl.detach().clone(),
                            cumulative_pnl=tm_state.cumulative_pnl.detach().clone(),
                            steps=tm_state.steps.detach().clone(),
                        )
                    else:
                        trade_penalty = torch.zeros((), device=self.device, dtype=torch.float32)

                (loss_unscaled / accum_steps).backward()

                loss_total += loss_unscaled.detach().item()
                policy_total += policy_loss.detach().item()
                entropy_total += entropy_scalar.detach().item()
                kl_total += kl.detach().item()
                drawdown_total += drawdown_penalty.detach().item()
                risk_total += risk_penalty.detach().item()
                trade_total += trade_penalty.detach().item()

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
                "drawdown_penalty": avg(drawdown_total),
                "risk_penalty": avg(risk_total),
                "trade_penalty": avg(trade_total),
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
        self._update_topk(eval_loss, step, payload)
