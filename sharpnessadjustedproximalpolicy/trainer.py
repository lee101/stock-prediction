"""Sharpness-Adjusted Proximal trainer.

Extends BinanceHourlyTrainer with sharpness-aware optimization.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from binanceneural.config import TrainingConfig
from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer, MultiSymbolDataModule
from binanceneural.model import BinancePolicyBase, PolicyConfig, build_policy
from binanceneural.trainer import BinanceHourlyTrainer, TrainingArtifacts, TrainingHistoryEntry
from differentiable_loss_utils import (
    HOURLY_PERIODS_PER_YEAR,
    compute_loss_by_type,
    simulate_hourly_trades,
    simulate_hourly_trades_binary,
)
from src.checkpoint_manager import TopKCheckpointManager
from src.serialization_utils import serialize_for_checkpoint

from .config import SAPConfig
from .sam_optimizer import (
    FullSAMOptimizer,
    GradientNoiseInjector,
    LookSAMOptimizer,
    SWAWrapper,
    SharpnessAdjustedOptimizer,
    SharpnessState,
)

try:
    from trainingefficiency.triton_sim_kernel import simulate_hourly_trades_triton
    HAS_TRITON_SIM = True
except ImportError:
    simulate_hourly_trades_triton = None
    HAS_TRITON_SIM = False

try:
    from trainingefficiency.fast_differentiable_sim import simulate_hourly_trades_fast
except ImportError:
    simulate_hourly_trades_fast = None

logger = logging.getLogger(__name__)


def _spectral_reg(model, weight: float) -> torch.Tensor:
    """Spectral regularization: penalize largest singular value of 2D weight matrices."""
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        if p.ndim == 2 and p.requires_grad:
            # Power iteration approximation for speed
            u = torch.randn(p.shape[0], device=p.device)
            u = u / u.norm()
            v = p.T @ u
            v = v / v.norm()
            sigma = (u @ p @ v).abs()
            reg = reg + sigma
    return weight * reg


@dataclass
class SAPHistoryEntry:
    epoch: int
    train_loss: float
    train_score: float
    train_sortino: float
    train_return: float
    val_loss: float
    val_score: float
    val_sortino: float
    val_return: float
    sharpness_raw: float = 0.0
    sharpness_ema: float = 0.0
    lr_scale: float = 1.0
    wall_time_s: float = 0.0
    # Continuous (marketsim-style) val metrics — populated when continuous_val_eval=True
    ms_sortino: float = 0.0
    ms_return: float = 0.0
    ms_num_trades: int = 0


class SAPTrainer:
    """Trainer with sharpness-adjusted proximal optimization."""

    def __init__(
        self,
        training_config: TrainingConfig,
        sap_config: SAPConfig,
        data_module: BinanceHourlyDataModule | MultiSymbolDataModule,
    ):
        self.tc = training_config
        self.sc = sap_config
        self.data = data_module
        self.device = torch.device(self.tc.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        run_name = self.tc.run_name or f"sap_{time.strftime('%Y%m%d_%H%M%S')}"
        self.tc.run_name = run_name
        self.checkpoint_dir = Path(self.tc.checkpoint_root) / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._total_train_steps = 0
        self._warmup_base_lrs: list[float] = []

    def _seed_everything(self, seed: int) -> None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def train(self) -> tuple[TrainingArtifacts, list[SAPHistoryEntry]]:
        self._seed_everything(self.tc.seed)
        # The differentiable Sortino helper is compile-wrapped independently of model compilation.
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
            torch._dynamo.config.suppress_errors = True
        if self.tc.use_tf32 and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        policy_cfg = PolicyConfig(
            input_dim=len(self.data.feature_columns),
            hidden_dim=self.tc.transformer_dim,
            dropout=self.tc.transformer_dropout,
            price_offset_pct=self.tc.price_offset_pct,
            min_price_gap_pct=self.tc.min_price_gap_pct,
            trade_amount_scale=self.tc.trade_amount_scale,
            num_heads=self.tc.transformer_heads,
            num_layers=self.tc.transformer_layers,
            max_len=max(self.tc.sequence_length, 32),
            model_arch=self.tc.model_arch,
            num_kv_heads=self.tc.num_kv_heads,
            mlp_ratio=self.tc.mlp_ratio,
            logits_softcap=self.tc.logits_softcap,
            use_midpoint_offsets=True,
            num_outputs=self.tc.num_outputs,
            max_hold_hours=self.tc.max_hold_hours,
            use_flex_attention=self.tc.use_flex_attention,
            moe_num_experts=getattr(self.tc, "moe_num_experts", 0),
        )
        model = build_policy(policy_cfg).to(self.device)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model params: {param_count:,}", flush=True)

        if self.tc.use_compile and hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

        base_optimizer = self._build_base_optimizer(model)
        sam_optimizer = self._wrap_with_sam(base_optimizer)
        self._warmup_base_lrs = [g["lr"] for g in base_optimizer.param_groups]

        # SWA
        swa = None
        if self.sc.use_swa:
            swa = SWAWrapper(model, swa_start_frac=self.sc.swa_start_frac)

        # Gradient noise
        grad_noise = None
        if self.sc.use_grad_noise:
            target_sharp = self.sc.target_sharpness
            if hasattr(sam_optimizer, 'target_sharpness'):
                target_sharp = sam_optimizer.target_sharpness
            grad_noise = GradientNoiseInjector(
                base_sigma=self.sc.grad_noise_sigma,
                gamma=self.sc.grad_noise_gamma,
                target_sharpness=max(target_sharp, 1e-8),
            )

        train_loader, val_loader = self._build_loaders()
        self._total_train_steps = max(1, len(train_loader) * self.tc.epochs)

        history: list[SAPHistoryEntry] = []
        best_score = float("-inf")
        best_checkpoint: Path | None = None
        ckpt_mgr = TopKCheckpointManager(self.checkpoint_dir, max_keep=10, mode="max")
        epochs_without_improvement = 0

        global_step = 0
        extras = []
        if swa: extras.append("swa")
        if grad_noise: extras.append("gradnoise")
        if self.sc.use_adaptive_feature_noise: extras.append("afn")
        extra_str = f" +{','.join(extras)}" if extras else ""
        print(f"Training: {len(train_loader)} train, {len(val_loader)} val batches, mode={self.sc.sam_mode}{extra_str}", flush=True)

        for epoch in range(1, self.tc.epochs + 1):
            t0 = time.time()

            train_metrics, global_step = self._train_epoch(
                model, train_loader, sam_optimizer, base_optimizer, global_step, epoch,
                grad_noise=grad_noise,
            )

            # SWA: update weight average after each epoch
            if swa:
                swa.update(epoch, self.tc.epochs)

            val_metrics = self._val_epoch(model, val_loader, epoch)

            # Optionally run full-period continuous simulation as the real val metric
            ms_metrics: dict[str, float] = {}
            use_cont = getattr(self.sc, "continuous_val_eval", False)
            if use_cont:
                ms_metrics = self._run_continuous_val(model)

            wall_time = time.time() - t0
            sharp_state = self._get_sharpness_state(sam_optimizer)

            entry = SAPHistoryEntry(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_score=train_metrics["score"],
                train_sortino=train_metrics["sortino"],
                train_return=train_metrics["return"],
                val_loss=val_metrics["loss"],
                val_score=val_metrics["score"],
                val_sortino=val_metrics["sortino"],
                val_return=val_metrics["return"],
                sharpness_raw=sharp_state.raw,
                sharpness_ema=sharp_state.ema,
                lr_scale=sharp_state.lr_scale,
                wall_time_s=wall_time,
                ms_sortino=ms_metrics.get("sortino", 0.0),
                ms_return=ms_metrics.get("return", 0.0),
                ms_num_trades=ms_metrics.get("num_trades", 0),
            )
            history.append(entry)

            # Checkpoint selection: use continuous val sortino when available, else batch score
            ckpt_score = ms_metrics.get("sortino", val_metrics["score"]) if use_cont and ms_metrics else val_metrics["score"]

            ckpt_path = self._save_checkpoint(model, epoch, val_metrics, sharp_state)
            ckpt_mgr.register(ckpt_path, ckpt_score, epoch=epoch)
            if ckpt_score > best_score:
                best_score = ckpt_score
                best_checkpoint = ckpt_path
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            ms_str = f" MS Sort={ms_metrics['sortino']:.3f} Ret={ms_metrics['return']:.4f} T={ms_metrics.get('num_trades',0)}" if ms_metrics else ""
            print(
                f"Ep {epoch}/{self.tc.epochs} | "
                f"T {train_metrics['sortino']:.3f}/{train_metrics['return']:.4f} | "
                f"V {val_metrics['sortino']:.3f}/{val_metrics['return']:.4f} | "
                f"Sharp {sharp_state.ema:.3f} Scale {sharp_state.lr_scale:.2f} |"
                f"{ms_str} {wall_time:.1f}s",
                flush=True,
            )

            if (
                epoch >= self.sc.early_stop_min_epochs
                and self.sc.early_stop_patience > 0
                and epochs_without_improvement >= self.sc.early_stop_patience
            ):
                print(f"Early stop at epoch {epoch} (no improvement for {self.sc.early_stop_patience} epochs)")
                break

        # SWA: evaluate with averaged weights
        if swa and swa.n_averaged > 1:
            print(f"SWA: applying averaged weights ({swa.n_averaged} snapshots)", flush=True)
            swa.apply_swa_weights()
            swa_metrics = self._val_epoch(model, val_loader, epoch=0)
            swa_entry = SAPHistoryEntry(
                epoch=self.tc.epochs + 1,
                train_loss=0, train_score=0, train_sortino=0, train_return=0,
                val_loss=swa_metrics["loss"], val_score=swa_metrics["score"],
                val_sortino=swa_metrics["sortino"], val_return=swa_metrics["return"],
                sharpness_raw=0, sharpness_ema=0, lr_scale=1, wall_time_s=0,
            )
            history.append(swa_entry)
            ckpt_path = self._save_checkpoint(model, self.tc.epochs + 1, swa_metrics, SharpnessState())
            ckpt_mgr.register(ckpt_path, swa_metrics["score"], epoch=self.tc.epochs + 1)
            if swa_metrics["score"] > best_score:
                best_score = swa_metrics["score"]
                best_checkpoint = ckpt_path
            print(f"SWA val: Sort={swa_metrics['sortino']:.3f} Ret={swa_metrics['return']:.4f}", flush=True)

        self._save_history(history)

        artifacts = TrainingArtifacts(
            state_dict=model.state_dict(),
            normalizer=self.data.normalizer,
            history=[TrainingHistoryEntry(
                epoch=h.epoch, train_loss=h.train_loss, train_score=h.train_score,
                train_sortino=h.train_sortino, train_return=h.train_return,
                val_loss=h.val_loss, val_score=h.val_score,
                val_sortino=h.ms_sortino if h.ms_sortino != 0.0 else h.val_sortino,
                val_return=h.ms_return if h.ms_return != 0.0 else h.val_return,
            ) for h in history],
            feature_columns=list(self.data.feature_columns),
            config=self.tc,
            checkpoint_paths=list(self.checkpoint_dir.glob("*.pt")),
            best_checkpoint=best_checkpoint,
        )
        return artifacts, history

    def _train_epoch(
        self,
        model: BinancePolicyBase,
        loader,
        sam_opt,
        base_opt,
        global_step: int,
        epoch: int,
        grad_noise: GradientNoiseInjector | None = None,
    ) -> tuple[dict[str, float], int]:
        model.train()
        dev = self.device
        nb = dev.type == "cuda"
        loss_sum = score_sum = sortino_sum = return_sum = 0.0
        steps = 0

        sim_fn = self._pick_sim_fn()
        base_lag = int(self.tc.decision_lag_bars)
        lag_str = self.tc.decision_lag_range.strip()
        lag_list = [int(x) for x in lag_str.split(",") if x.strip()] if lag_str else [base_lag]
        scale = float(self.tc.trade_amount_scale)
        fill_buf = self.tc.fill_buffer_pct
        if self.tc.fill_buffer_warmup_epochs > 0 and epoch <= self.tc.fill_buffer_warmup_epochs:
            fill_buf = fill_buf * epoch / self.tc.fill_buffer_warmup_epochs

        sharp_state = self._get_sharpness_state(sam_opt)

        for batch_idx, batch in enumerate(loader):
            features = batch["features"].to(dev, non_blocking=nb)
            # static feature noise
            if self.tc.feature_noise_std > 0:
                features = features + self.tc.feature_noise_std * torch.randn_like(features)
            # adaptive feature noise: scale by sharpness
            if self.sc.use_adaptive_feature_noise and sharp_state.ema > 0:
                ratio = min(sharp_state.ema / max(sharp_state.raw, 1e-8), 3.0) if sharp_state.raw > 0 else 1.0
                afn_std = min(self.sc.adaptive_fn_base * max(ratio, 0.1), self.sc.adaptive_fn_max)
                features = features + afn_std * torch.randn_like(features)
            highs = batch["high"].to(dev, non_blocking=nb)
            lows = batch["low"].to(dev, non_blocking=nb)
            closes = batch["close"].to(dev, non_blocking=nb)
            opens = batch["open"].to(dev, non_blocking=nb) if "open" in batch else None
            ref_close = batch["reference_close"].to(dev, non_blocking=nb)
            ch_high = batch["chronos_high"].to(dev, non_blocking=nb)
            ch_low = batch["chronos_low"].to(dev, non_blocking=nb)

            def forward_and_loss():
                outputs = model(features)
                actions = model.decode_actions(outputs, reference_close=ref_close, chronos_high=ch_high, chronos_low=ch_low)
                ti = actions["trade_amount"] / scale
                bi = actions["buy_amount"] / scale
                si = actions["sell_amount"] / scale
                sim_kwargs = dict(
                    highs=highs, lows=lows, closes=closes, opens=opens,
                    buy_prices=actions["buy_price"], sell_prices=actions["sell_price"],
                    trade_intensity=ti, buy_trade_intensity=bi, sell_trade_intensity=si,
                    maker_fee=batch.get("maker_fee", self.tc.maker_fee),
                    initial_cash=self.tc.initial_cash,
                    can_short=self.sc.can_short,
                    can_long=batch.get("can_long", True),
                    max_leverage=self.tc.max_leverage,
                    market_order_entry=self.tc.market_order_entry,
                    fill_buffer_pct=fill_buf,
                    margin_annual_rate=float(self.tc.margin_annual_rate),
                )
                ppy = float(self.tc.periods_per_year or HOURLY_PERIODS_PER_YEAR)
                if len(lag_list) > 1:
                    losses = []
                    for lag_i in lag_list:
                        sim_i = sim_fn(**sim_kwargs, temperature=float(self.tc.fill_temperature), decision_lag_bars=lag_i)
                        lo_i, _, _, _ = compute_loss_by_type(
                            sim_i.returns.float(), self.tc.loss_type,
                            target_sign=self.tc.sortino_target_sign, periods_per_year=ppy,
                            return_weight=self.tc.return_weight, smoothness_penalty=self.tc.smoothness_penalty,
                            dd_penalty=self.tc.dd_penalty,
                        )
                        losses.append(lo_i)
                    main_loss = sum(losses) / len(losses)
                else:
                    sim = sim_fn(**sim_kwargs, temperature=float(self.tc.fill_temperature), decision_lag_bars=base_lag)
                    main_loss, _, _, _ = compute_loss_by_type(
                        sim.returns.float(), self.tc.loss_type,
                        target_sign=self.tc.sortino_target_sign, periods_per_year=ppy,
                        return_weight=self.tc.return_weight, smoothness_penalty=self.tc.smoothness_penalty,
                        dd_penalty=self.tc.dd_penalty,
                    )

                total_loss = main_loss

                # Multi-period loss: compute loss on sub-windows for horizon diversity
                if self.sc.multi_period_windows and self.sc.multi_period_weight > 0:
                    T_full = features.shape[1]
                    sub_losses = []
                    for win_len in self.sc.multi_period_windows:
                        if win_len >= T_full:
                            continue
                        # Compute on last win_len bars of the sequence
                        sub_sim = sim_fn(
                            highs=highs[:, -win_len:], lows=lows[:, -win_len:],
                            closes=closes[:, -win_len:],
                            opens=opens[:, -win_len:] if opens is not None else None,
                            buy_prices=actions["buy_price"][:, -win_len:],
                            sell_prices=actions["sell_price"][:, -win_len:],
                            trade_intensity=ti[:, -win_len:],
                            buy_trade_intensity=bi[:, -win_len:],
                            sell_trade_intensity=si[:, -win_len:],
                            maker_fee=batch.get("maker_fee", self.tc.maker_fee),
                            initial_cash=self.tc.initial_cash,
                            can_short=batch.get("can_short", False),
                            can_long=batch.get("can_long", True),
                            max_leverage=self.tc.max_leverage,
                            market_order_entry=self.tc.market_order_entry,
                            fill_buffer_pct=fill_buf,
                            margin_annual_rate=float(self.tc.margin_annual_rate),
                            temperature=float(self.tc.fill_temperature),
                            decision_lag_bars=base_lag,
                        )
                        sub_lo, _, _, _ = compute_loss_by_type(
                            sub_sim.returns.float(), self.tc.loss_type,
                            target_sign=self.tc.sortino_target_sign, periods_per_year=ppy,
                            return_weight=self.tc.return_weight, smoothness_penalty=self.tc.smoothness_penalty,
                            dd_penalty=self.tc.dd_penalty,
                        )
                        sub_losses.append(sub_lo)
                    if sub_losses:
                        sub_avg = sum(sub_losses) / len(sub_losses)
                        total_loss = (1 - self.sc.multi_period_weight) * main_loss + self.sc.multi_period_weight * sub_avg

                # Spectral regularization
                if self.sc.spectral_reg_weight > 0:
                    total_loss = total_loss + _spectral_reg(model, self.sc.spectral_reg_weight)

                return total_loss

            # -- Main training step --
            if isinstance(sam_opt, FullSAMOptimizer):
                sam_opt.zero_grad(set_to_none=True)
                loss = forward_and_loss()
                loss.backward()
                if self.tc.grad_clip:
                    clip_grad_norm_(model.parameters(), self.tc.grad_clip)
                if grad_noise:
                    grad_noise.inject(model, global_step, sharp_state.ema)
                self._apply_lr_schedule(base_opt, global_step)
                sam_opt.update_base_lrs()
                _, _ = sam_opt.step_with_sam(forward_and_loss)
            elif isinstance(sam_opt, LookSAMOptimizer):
                sam_opt.zero_grad(set_to_none=True)
                loss = forward_and_loss()
                loss.backward()
                if self.tc.grad_clip:
                    clip_grad_norm_(model.parameters(), self.tc.grad_clip)
                if grad_noise:
                    grad_noise.inject(model, global_step, sharp_state.ema)
                self._apply_lr_schedule(base_opt, global_step)
                sam_opt.step(loss_fn=forward_and_loss)
            elif isinstance(sam_opt, SharpnessAdjustedOptimizer):
                sam_opt.zero_grad(set_to_none=True)
                loss = forward_and_loss()
                loss.backward()
                if self.tc.grad_clip:
                    clip_grad_norm_(model.parameters(), self.tc.grad_clip)
                if grad_noise:
                    grad_noise.inject(model, global_step, sharp_state.ema)

                # Apply LR schedule before step
                self._apply_lr_schedule(base_opt, global_step)
                sam_opt.update_base_lrs()

                sam_opt.step()

                if sam_opt.should_probe():
                    # probe_sharpness already wraps the call in inference_mode internally
                    sam_opt.probe_sharpness(forward_and_loss, loss.item())
            else:
                # Baseline: no SAM
                base_opt.zero_grad(set_to_none=True)
                loss = forward_and_loss()
                loss.backward()
                if self.tc.grad_clip:
                    clip_grad_norm_(model.parameters(), self.tc.grad_clip)
                if grad_noise:
                    grad_noise.inject(model, global_step, sharp_state.ema)
                self._apply_lr_schedule(base_opt, global_step)
                base_opt.step()

            # Compute full metrics for logging (detached, no dropout)
            model.eval()
            with torch.no_grad():
                outputs = model(features)
                actions = model.decode_actions(outputs, reference_close=ref_close, chronos_high=ch_high, chronos_low=ch_low)
                sim_kwargs_log = dict(
                    highs=highs, lows=lows, closes=closes, opens=opens,
                    buy_prices=actions["buy_price"], sell_prices=actions["sell_price"],
                    trade_intensity=actions["trade_amount"] / scale,
                    buy_trade_intensity=actions["buy_amount"] / scale,
                    sell_trade_intensity=actions["sell_amount"] / scale,
                    maker_fee=batch.get("maker_fee", self.tc.maker_fee),
                    initial_cash=self.tc.initial_cash,
                    can_short=self.sc.can_short,
                    can_long=batch.get("can_long", True),
                    max_leverage=self.tc.max_leverage,
                    market_order_entry=self.tc.market_order_entry,
                    fill_buffer_pct=fill_buf,
                    margin_annual_rate=float(self.tc.margin_annual_rate),
                )
                sim_log = sim_fn(**sim_kwargs_log, temperature=float(self.tc.fill_temperature), decision_lag_bars=base_lag)
                ppy = float(self.tc.periods_per_year or HOURLY_PERIODS_PER_YEAR)
                _, sc, so, ar = compute_loss_by_type(
                    sim_log.returns.float(), self.tc.loss_type,
                    target_sign=self.tc.sortino_target_sign, periods_per_year=ppy,
                    return_weight=self.tc.return_weight, smoothness_penalty=self.tc.smoothness_penalty,
                    dd_penalty=self.tc.dd_penalty,
                )
                loss_sum += loss.detach().mean().item()
                score_sum += sc.detach().mean().item()
                sortino_sum += so.detach().mean().item()
                return_sum += ar.detach().mean().item()
            model.train()

            steps += 1
            global_step += 1

        n = max(steps, 1)
        return {
            "loss": loss_sum / n,
            "score": score_sum / n,
            "sortino": sortino_sum / n,
            "return": return_sum / n,
        }, global_step

    def _val_epoch(self, model: BinancePolicyBase, loader, epoch: int) -> dict[str, float]:
        model.eval()
        dev = self.device
        nb = dev.type == "cuda"
        loss_sum = score_sum = sortino_sum = return_sum = 0.0
        steps = 0
        scale = float(self.tc.trade_amount_scale)
        lag = int(self.tc.decision_lag_bars)
        lag_str = self.tc.decision_lag_range.strip()
        lag_list = [int(x) for x in lag_str.split(",") if x.strip()] if lag_str else [lag]
        val_lag = max(lag_list)

        with torch.inference_mode():
            for batch in loader:
                features = batch["features"].to(dev, non_blocking=nb)
                highs = batch["high"].to(dev, non_blocking=nb)
                lows = batch["low"].to(dev, non_blocking=nb)
                closes = batch["close"].to(dev, non_blocking=nb)
                opens = batch["open"].to(dev, non_blocking=nb) if "open" in batch else None
                ref_close = batch["reference_close"].to(dev, non_blocking=nb)
                ch_high = batch["chronos_high"].to(dev, non_blocking=nb)
                ch_low = batch["chronos_low"].to(dev, non_blocking=nb)

                outputs = model(features)
                actions = model.decode_actions(outputs, reference_close=ref_close, chronos_high=ch_high, chronos_low=ch_low)

                sim_kwargs = dict(
                    highs=highs, lows=lows, closes=closes, opens=opens,
                    buy_prices=actions["buy_price"], sell_prices=actions["sell_price"],
                    trade_intensity=actions["trade_amount"] / scale,
                    buy_trade_intensity=actions["buy_amount"] / scale,
                    sell_trade_intensity=actions["sell_amount"] / scale,
                    maker_fee=batch.get("maker_fee", self.tc.maker_fee),
                    initial_cash=self.tc.initial_cash,
                    can_short=self.sc.can_short,
                    can_long=batch.get("can_long", True),
                    max_leverage=self.tc.max_leverage,
                    market_order_entry=self.tc.market_order_entry,
                    fill_buffer_pct=self.tc.fill_buffer_pct,
                    margin_annual_rate=float(self.tc.margin_annual_rate),
                )

                if bool(self.tc.validation_use_binary_fills):
                    sim = simulate_hourly_trades_binary(**sim_kwargs, decision_lag_bars=val_lag)
                else:
                    sim_fn = self._pick_sim_fn()
                    sim = sim_fn(**sim_kwargs, temperature=float(self.tc.fill_temperature), decision_lag_bars=val_lag)

                ppy = float(self.tc.periods_per_year or HOURLY_PERIODS_PER_YEAR)
                lo, sc, so, ar = compute_loss_by_type(
                    sim.returns.float(), self.tc.loss_type,
                    target_sign=self.tc.sortino_target_sign, periods_per_year=ppy,
                    return_weight=self.tc.return_weight, smoothness_penalty=self.tc.smoothness_penalty,
                    dd_penalty=self.tc.dd_penalty,
                )
                loss_sum += lo.detach().mean().item()
                score_sum += sc.detach().mean().item()
                sortino_sum += so.detach().mean().item()
                return_sum += ar.detach().mean().item()
                steps += 1

        n = max(steps, 1)
        return {
            "loss": loss_sum / n,
            "score": score_sum / n,
            "sortino": sortino_sum / n,
            "return": return_sum / n,
        }

    def _pick_sim_fn(self):
        if HAS_TRITON_SIM and self.device.type == "cuda":
            return simulate_hourly_trades_triton
        if simulate_hourly_trades_fast is not None and self.tc.use_vectorized_sim:
            return simulate_hourly_trades_fast
        return simulate_hourly_trades

    def _build_base_optimizer(self, model) -> torch.optim.Optimizer:
        fused = self.device.type == "cuda"
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=self.tc.learning_rate,
            weight_decay=self.tc.weight_decay,
            fused=fused,
        )

    def _wrap_with_sam(self, base_opt):
        mode = self.sc.sam_mode.lower()
        if mode == "periodic":
            return SharpnessAdjustedOptimizer(
                base_opt,
                rho=self.sc.rho,
                probe_every=self.sc.probe_every,
                ema_beta=self.sc.ema_beta,
                target_sharpness=self.sc.target_sharpness,
                min_scale=self.sc.min_lr_scale,
                max_scale=self.sc.max_lr_scale,
                scale_mode=self.sc.scale_mode,
            )
        elif mode == "full_sam":
            return FullSAMOptimizer(
                base_opt,
                rho=self.sc.rho,
                adaptive=self.sc.adaptive_sam,
                ema_beta=self.sc.ema_beta,
                target_sharpness=self.sc.target_sharpness,
                rho_min=self.sc.rho_min,
                rho_max=self.sc.rho_max,
            )
        elif mode == "looksam":
            return LookSAMOptimizer(
                base_opt,
                rho=self.sc.rho,
                sam_every=self.sc.looksam_every,
                alpha=self.sc.looksam_alpha,
            )
        else:
            return None  # baseline, use base_opt directly

    def _apply_lr_schedule(self, optimizer, global_step):
        import math
        if self._total_train_steps <= 0:
            return
        sched = (self.tc.lr_schedule or "none").lower()
        if sched == "cosine":
            warmup = self.tc.warmup_steps
            min_r = self.tc.lr_min_ratio
            if global_step < warmup:
                frac = (global_step + 1) / warmup
                for group, base_lr in zip(optimizer.param_groups, self._warmup_base_lrs):
                    group["lr"] = base_lr * frac
            else:
                decay = (global_step - warmup) / max(1, self._total_train_steps - warmup)
                decay = min(decay, 1.0)
                mult = min_r + (1 - min_r) * 0.5 * (1 + math.cos(math.pi * decay))
                for group, base_lr in zip(optimizer.param_groups, self._warmup_base_lrs):
                    group["lr"] = base_lr * mult
        elif sched == "none" and self.tc.warmup_steps and global_step < self.tc.warmup_steps:
            frac = (global_step + 1) / self.tc.warmup_steps
            for group, base_lr in zip(optimizer.param_groups, self._warmup_base_lrs):
                group["lr"] = base_lr * frac

    def _get_sharpness_state(self, sam_opt) -> SharpnessState:
        if hasattr(sam_opt, "state"):
            return sam_opt.state
        return SharpnessState()

    def _save_checkpoint(self, model, epoch, metrics, sharp_state) -> Path:
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        payload = {
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "epoch": epoch,
            "config": serialize_for_checkpoint(self.tc),
            "sap_config": asdict(self.sc),
            "normalizer": self.data.normalizer.to_dict(),
            "feature_columns": list(self.data.feature_columns),
            "sharpness": {
                "raw": sharp_state.raw,
                "ema": sharp_state.ema,
                "lr_scale": sharp_state.lr_scale,
            },
        }
        torch.save(payload, path)
        return path

    def _save_history(self, history: list[SAPHistoryEntry]):
        path = self.checkpoint_dir / "sap_history.json"
        data = [
            {
                "epoch": h.epoch,
                "train_loss": h.train_loss, "train_score": h.train_score,
                "train_sortino": h.train_sortino, "train_return": h.train_return,
                "val_loss": h.val_loss, "val_score": h.val_score,
                "val_sortino": h.val_sortino, "val_return": h.val_return,
                "ms_sortino": h.ms_sortino, "ms_return": h.ms_return,
                "ms_num_trades": h.ms_num_trades,
                "sharpness_raw": h.sharpness_raw, "sharpness_ema": h.sharpness_ema,
                "lr_scale": h.lr_scale, "wall_time_s": h.wall_time_s,
            }
            for h in history
        ]
        path.write_text(json.dumps(data, indent=2))

    def _run_continuous_val(self, model) -> dict[str, float]:
        """Full-period continuous simulation on val dataset (like marketsim).

        Uses rolling window inference (seq_len-bar windows) then runs binary-fill
        simulation over the entire val sequence in one shot.  ~5-15s per epoch.
        Returns sortino, return (total), num_trades.  Empty dict on failure.
        """
        if not hasattr(self.data, "val_dataset"):
            return {}
        vds = self.data.val_dataset
        T = len(vds.frame)
        if T < self.tc.sequence_length + 5:
            return {}

        model.eval()
        dev = self.device
        seq_len = self.tc.sequence_length
        scale = float(self.tc.trade_amount_scale)

        try:
            with torch.inference_mode():
                feat_all = torch.from_numpy(vds.features).float().to(dev)  # [T, F]
                ref_close_all = torch.from_numpy(vds.reference_close).float().to(dev)
                ch_high_all = torch.from_numpy(vds.chronos_high).float().to(dev)
                ch_low_all = torch.from_numpy(vds.chronos_low).float().to(dev)
                opens_all = torch.from_numpy(vds.opens).float().to(dev)
                highs_all = torch.from_numpy(vds.highs).float().to(dev)
                lows_all = torch.from_numpy(vds.lows).float().to(dev)
                closes_all = torch.from_numpy(vds.closes).float().to(dev)

                buy_prices_list: list[torch.Tensor] = []
                sell_prices_list: list[torch.Tensor] = []
                trade_int_list: list[torch.Tensor] = []
                buy_int_list: list[torch.Tensor] = []
                sell_int_list: list[torch.Tensor] = []

                # Batch the rolling windows for efficiency
                infer_bs = 128
                for b_start in range(0, T, infer_bs):
                    b_end = min(b_start + infer_bs, T)
                    batch_f, batch_rc, batch_chh, batch_chl = [], [], [], []
                    for t in range(b_start, b_end):
                        ws = max(0, t - seq_len + 1)
                        win = feat_all[ws:t + 1]
                        if win.shape[0] < seq_len:
                            win = torch.cat([win[:1].expand(seq_len - win.shape[0], -1), win], dim=0)
                        batch_f.append(win)
                        batch_rc.append(ref_close_all[t])
                        if ch_high_all.dim() == 1:
                            batch_chh.append(ch_high_all[t].unsqueeze(0))
                            batch_chl.append(ch_low_all[t].unsqueeze(0))
                        else:
                            batch_chh.append(ch_high_all[t])
                            batch_chl.append(ch_low_all[t])

                    feats_b = torch.stack(batch_f, dim=0)  # [B, seq_len, F]
                    rc_b = torch.stack(batch_rc, dim=0).unsqueeze(-1)  # [B, 1]
                    chh_b = torch.stack(batch_chh, dim=0)  # [B, ...]
                    chl_b = torch.stack(batch_chl, dim=0)

                    outputs = model(feats_b)
                    actions = model.decode_actions(outputs, reference_close=rc_b, chronos_high=chh_b, chronos_low=chl_b)

                    # Take only last-step action
                    buy_prices_list.append(actions["buy_price"][:, -1:])
                    sell_prices_list.append(actions["sell_price"][:, -1:])
                    trade_int_list.append(actions["trade_amount"][:, -1:] / scale)
                    buy_int_list.append(actions["buy_amount"][:, -1:] / scale)
                    sell_int_list.append(actions["sell_amount"][:, -1:] / scale)

                # Assemble full-T arrays [1, T] — each element is [B, 1] so cat → [T, 1] → reshape
                def _cat_to_1T(lst):
                    return torch.cat(lst, dim=0).view(1, T)  # [T, 1] → [1, T]

                buy_prices = _cat_to_1T(buy_prices_list)
                sell_prices = _cat_to_1T(sell_prices_list)
                trade_int = _cat_to_1T(trade_int_list)
                buy_int = _cat_to_1T(buy_int_list)
                sell_int = _cat_to_1T(sell_int_list)

                # Re-shape price arrays to [1, T]
                h = highs_all.unsqueeze(0)   # [1, T]
                l = lows_all.unsqueeze(0)
                c = closes_all.unsqueeze(0)
                o = opens_all.unsqueeze(0)

                sim = simulate_hourly_trades_binary(
                    highs=h, lows=l, closes=c, opens=o,
                    buy_prices=buy_prices, sell_prices=sell_prices,
                    trade_intensity=trade_int, buy_trade_intensity=buy_int,
                    sell_trade_intensity=sell_int,
                    maker_fee=self.tc.maker_fee,
                    initial_cash=self.tc.initial_cash,
                    can_short=self.sc.can_short, can_long=True,
                    max_leverage=self.tc.max_leverage,
                    fill_buffer_pct=self.tc.fill_buffer_pct,
                    margin_annual_rate=float(self.tc.margin_annual_rate),
                    decision_lag_bars=int(self.tc.decision_lag_bars),
                )

            returns = sim.returns.float().cpu().squeeze(0)
            pv = sim.portfolio_values.float().cpu().squeeze(0)
            mean_r = returns.mean().item()
            downside = returns.clamp(max=0.0)
            down_std = (downside.square().mean() + 1e-10).sqrt().item()
            ann_factor = 8760.0 / max(T, 1)
            ms_sort = (mean_r / max(down_std, 1e-10)) * (ann_factor ** 0.5)
            ms_ret = (pv[-1] / pv[0] - 1).item()

            # Count trades from inventory_path
            num_trades = 0
            inv_path = getattr(sim, "inventory_path", None)
            if inv_path is not None:
                inv = inv_path.float().cpu().squeeze(0)
                was_flat = True
                for i in range(inv.shape[0]):
                    is_flat = abs(inv[i].item()) < 1e-8
                    if was_flat and not is_flat:
                        num_trades += 1
                    was_flat = is_flat

            return {"sortino": ms_sort, "return": ms_ret, "num_trades": num_trades}

        except Exception as exc:
            logger.warning(f"continuous_val failed: {exc}")
            return {}

    def _build_loaders(self):
        if self.device.type == "cuda" and hasattr(self.data, "gpu_cached_dataloader"):
            try:
                train_ld = self.data.gpu_cached_dataloader("train", self.tc.batch_size, self.device, shuffle=True)
                val_ld = self.data.gpu_cached_dataloader("val", self.tc.batch_size, self.device, shuffle=False)
                return train_ld, val_ld
            except RuntimeError:
                torch.cuda.empty_cache()
        train_ld = self.data.train_dataloader(self.tc.batch_size, self.tc.num_workers)
        val_ld = self.data.val_dataloader(self.tc.batch_size, self.tc.num_workers)
        return train_ld, val_ld
