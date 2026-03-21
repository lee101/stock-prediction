from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
from differentiable_loss_utils import (
    HOURLY_PERIODS_PER_YEAR,
    compute_loss_by_type,
    simulate_hourly_trades,
    simulate_hourly_trades_binary,
)
try:
    from trainingefficiency.fast_differentiable_sim import (
        simulate_hourly_trades_fast,
        simulate_hourly_trades_compiled,
    )
except ImportError:
    simulate_hourly_trades_fast = None
    simulate_hourly_trades_compiled = None
from torch.nn.utils import clip_grad_norm_  # type: ignore
from traininglib.optim_factory import MultiOptim

from src.checkpoint_manager import TopKCheckpointManager
from src.serialization_utils import serialize_for_checkpoint
from src.torch_load_utils import torch_load_compat

from .config import TrainingConfig
from .data import BinanceHourlyDataModule, FeatureNormalizer, MultiSymbolDataModule
from .model import BinancePolicyBase, PolicyConfig, build_policy


try:
    import wandb as _wandb
except ImportError:
    _wandb = None  # type: ignore[assignment]

try:
    from torch.optim._muon import Muon
    MUON_AVAILABLE = True
except Exception:
    try:
        from pytorch_optimizer import Muon
        MUON_AVAILABLE = True
    except Exception:
        Muon = None  # type: ignore
        MUON_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistoryEntry:
    epoch: int
    train_loss: float
    train_score: float
    train_sortino: float
    train_return: float
    val_loss: float | None = None
    val_score: float | None = None
    val_sortino: float | None = None
    val_return: float | None = None


@dataclass
class TrainingArtifacts:
    state_dict: dict[str, torch.Tensor]
    normalizer: FeatureNormalizer
    history: list[TrainingHistoryEntry] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    config: TrainingConfig | None = None
    checkpoint_paths: list[Path] = field(default_factory=list)
    best_checkpoint: Path | None = None


class BinanceHourlyTrainer:
    def __init__(self, config: TrainingConfig, data_module: BinanceHourlyDataModule | MultiSymbolDataModule) -> None:
        self.config = config
        self.data = data_module
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        run_name = self.config.run_name or time.strftime("binanceneural_%Y%m%d_%H%M%S")
        self.config.run_name = run_name
        self.checkpoint_dir = self.config.checkpoint_root / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._warmup_base_lrs: list[float] = []
        self._amp_context = nullcontext()
        self._scaler = None
        self._total_train_steps = 0
        self._weight_decay_groups: list[tuple[dict, float]] = []

    def _get_fill_buffer(self, epoch: int) -> float:
        buf = self.config.fill_buffer_pct
        warmup = self.config.fill_buffer_warmup_epochs
        if warmup > 0 and epoch <= warmup:
            buf = buf * epoch / warmup
        return buf

    def train(self) -> TrainingArtifacts:
        torch.manual_seed(self.config.seed)
        if float(self.config.fill_temperature) <= 0.0:
            raise ValueError(
                f"fill_temperature must be > 0 for differentiable fills (got {self.config.fill_temperature})."
            )
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
            torch._dynamo.config.suppress_errors = True
        if self.config.use_tf32:
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            try:  # pragma: no cover - best-effort perf toggle
                from src.torch_backend import configure_tf32_backends

                configure_tf32_backends(torch, logger=logger)
            except Exception:
                pass
        if self.config.use_flash_attention and self.device.type == "cuda":
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)

        wandb_run = self._init_wandb()

        policy_cfg = PolicyConfig(
            input_dim=len(self.data.feature_columns),
            hidden_dim=self.config.transformer_dim,
            dropout=self.config.transformer_dropout,
            price_offset_pct=self.config.price_offset_pct,
            min_price_gap_pct=self.config.min_price_gap_pct,
            trade_amount_scale=self.config.trade_amount_scale,
            num_heads=self.config.transformer_heads,
            num_layers=self.config.transformer_layers,
            max_len=max(self.config.sequence_length, 32),
            model_arch=self.config.model_arch,
            num_kv_heads=self.config.num_kv_heads,
            mlp_ratio=self.config.mlp_ratio,
            logits_softcap=self.config.logits_softcap,
            rope_base=self.config.rope_base,
            use_qk_norm=self.config.use_qk_norm,
            use_causal_attention=self.config.use_causal_attention,
            rms_norm_eps=self.config.rms_norm_eps,
            attention_window=self.config.attention_window,
            use_residual_scalars=self.config.use_residual_scalars,
            residual_scale_init=self.config.residual_scale_init,
            skip_scale_init=self.config.skip_scale_init,
            use_value_embedding=self.config.use_value_embedding,
            value_embedding_every=self.config.value_embedding_every,
            value_embedding_scale=self.config.value_embedding_scale,
            use_midpoint_offsets=True,
            num_outputs=self.config.num_outputs,
            max_hold_hours=self.config.max_hold_hours,
            num_memory_tokens=self.config.num_memory_tokens,
            dilated_strides=self.config.dilated_strides,
        )
        model = build_policy(policy_cfg).to(self.device)

        if self.config.preload_checkpoint_path:
            preload_path = Path(self.config.preload_checkpoint_path)
            if preload_path.exists():
                logger.info("Preloading weights from %s", preload_path)
                checkpoint = torch_load_compat(preload_path, map_location="cpu", weights_only=False)
                state_dict = checkpoint.get("state_dict", checkpoint)
                # Strip _orig_mod. prefix from torch.compile'd checkpoints
                if any(k.startswith("_orig_mod.") for k in state_dict):
                    state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
                # Handle input dim mismatch between pretrain and finetune
                for key in list(state_dict.keys()):
                    if state_dict[key].shape != model.state_dict().get(key, state_dict[key]).shape:
                        logger.warning(
                            "Shape mismatch for %s: %s vs %s, skipping",
                            key,
                            state_dict[key].shape,
                            model.state_dict()[key].shape,
                        )
                        del state_dict[key]
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning("Missing keys during preload: %s", missing)
                if unexpected:
                    logger.warning("Unexpected keys during preload: %s", unexpected)
            else:
                logger.warning("Preload checkpoint not found: %s", preload_path)

        if self.config.use_compile and hasattr(torch, "compile"):
            model = torch.compile(model, mode="max-autotune", fullgraph=False)

        optimizer = self._build_optimizer(model)
        self._warmup_base_lrs = [group.get("lr", self.config.learning_rate) for group in optimizer.param_groups]
        self._weight_decay_groups = [
            (group, float(group.get("weight_decay", 0.0))) for group in self._iter_param_groups(optimizer)
        ]
        self._amp_context, self._scaler = self._build_amp()

        if self.device.type == "cuda" and hasattr(self.data, "gpu_cached_dataloader"):
            train_loader = self.data.gpu_cached_dataloader("train", self.config.batch_size, self.device, shuffle=True)
            val_loader = self.data.gpu_cached_dataloader("val", self.config.batch_size, self.device, shuffle=False)
            logger.info("Using GPU-cached dataloaders")
        else:
            train_loader = self.data.train_dataloader(self.config.batch_size, self.config.num_workers)
            val_loader = self.data.val_dataloader(self.config.batch_size, self.config.num_workers)
        self._total_train_steps = max(1, len(train_loader) * max(1, self.config.epochs))
        if self.config.dry_train_steps:
            self._total_train_steps = min(
                self._total_train_steps,
                int(self.config.dry_train_steps) * max(1, self.config.epochs),
            )

        history: list[TrainingHistoryEntry] = []
        best_score = float("-inf")
        best_checkpoint: Path | None = None
        ckpt_mgr = TopKCheckpointManager(self.checkpoint_dir, max_keep=10, mode="max")

        global_step = 0
        for epoch in range(1, self.config.epochs + 1):
            train_metrics, global_step = self._run_epoch(
                model,
                train_loader,
                optimizer,
                train=True,
                global_step=global_step,
                current_epoch=epoch,
            )
            val_metrics, _ = self._run_epoch(
                model,
                val_loader,
                optimizer=None,
                train=False,
                global_step=global_step,
                current_epoch=epoch,
            )

            entry = TrainingHistoryEntry(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_score=train_metrics["score"],
                train_sortino=train_metrics["sortino"],
                train_return=train_metrics["return"],
                val_loss=val_metrics["loss"],
                val_score=val_metrics["score"],
                val_sortino=val_metrics["sortino"],
                val_return=val_metrics["return"],
            )
            history.append(entry)

            ckpt_path = self._save_checkpoint(model, epoch, val_metrics)
            ckpt_mgr.register(ckpt_path, val_metrics["score"], epoch=epoch)
            if val_metrics["score"] > best_score:
                best_score = val_metrics["score"]
                best_checkpoint = ckpt_path

            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Score: {train_metrics['score']:.4f} "
                f"Sortino: {train_metrics['sortino']:.4f} Return: {train_metrics['return']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Score: {val_metrics['score']:.4f} "
                f"Sortino: {val_metrics['sortino']:.4f} Return: {val_metrics['return']:.4f}"
            )

            if wandb_run is not None:
                lr_now = optimizer.param_groups[0]["lr"] if hasattr(optimizer, "param_groups") else self.config.learning_rate
                wandb_run.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/sortino": train_metrics["sortino"],
                    "train/return": train_metrics["return"],
                    "train/score": train_metrics["score"],
                    "val/loss": val_metrics["loss"],
                    "val/sortino": val_metrics["sortino"],
                    "val/return": val_metrics["return"],
                    "val/score": val_metrics["score"],
                    "learning_rate": lr_now,
                }, step=epoch)

        if wandb_run is not None:
            wandb_run.finish()

        return TrainingArtifacts(
            state_dict=model.state_dict(),
            normalizer=self.data.normalizer,
            history=history,
            feature_columns=list(self.data.feature_columns),
            config=self.config,
            checkpoint_paths=list(self.checkpoint_dir.glob("*.pt")),
            best_checkpoint=best_checkpoint,
        )

    def _init_wandb(self):
        if _wandb is None or not self.config.wandb_project:
            return None
        try:
            cfg_dict = serialize_for_checkpoint(self.config)
            run = _wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.run_name,
                config=cfg_dict,
                reinit=True,
            )
            return run
        except Exception as e:
            logger.warning("wandb init failed: %s", e)
            return None

    def _run_epoch(
        self,
        model: BinancePolicyBase,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer | None,
        *,
        train: bool,
        global_step: int,
        current_epoch: int = 1,
    ) -> tuple[dict[str, float], int]:
        model.train(train)
        total_loss = 0.0
        total_score = 0.0
        total_sortino = 0.0
        total_return = 0.0
        steps = 0

        # Torch compile may wrap forward in CUDAGraphs; mark the start of each step to
        # avoid "overwritten output" errors when internal tensors are reused across runs.
        mark_step_begin = None
        if bool(self.config.use_compile):
            compiler = getattr(torch, "compiler", None)
            mark_step_begin = getattr(compiler, "cudagraph_mark_step_begin", None) if compiler is not None else None

        nb = self.device.type == "cuda"

        for batch in loader:
            if mark_step_begin is not None:
                mark_step_begin()
            features = batch["features"].to(self.device, non_blocking=nb)
            if train and self.config.feature_noise_std > 0:
                features = features + self.config.feature_noise_std * torch.randn_like(features)
            opens = batch["open"].to(self.device, non_blocking=nb) if "open" in batch else None
            highs = batch["high"].to(self.device, non_blocking=nb)
            lows = batch["low"].to(self.device, non_blocking=nb)
            closes = batch["close"].to(self.device, non_blocking=nb)
            reference_close = batch["reference_close"].to(self.device, non_blocking=nb)
            chronos_high = batch["chronos_high"].to(self.device, non_blocking=nb)
            chronos_low = batch["chronos_low"].to(self.device, non_blocking=nb)

            split_amp = bool(self.config.split_amp) and self._amp_context is not None
            use_vsim = bool(self.config.use_vectorized_sim) and simulate_hourly_trades_fast is not None

            with self._amp_context:
                outputs = model(features)
                actions = model.decode_actions(
                    outputs,
                    reference_close=reference_close,
                    chronos_high=chronos_high,
                    chronos_low=chronos_low,
                )

            sim_context = nullcontext() if split_amp else self._amp_context

            with sim_context:
                scale = float(self.config.trade_amount_scale)
                if split_amp:
                    trade_intensity = actions["trade_amount"].float() / scale
                    buy_intensity = actions["buy_amount"].float() / scale
                    sell_intensity = actions["sell_amount"].float() / scale
                    sim_highs = highs.float()
                    sim_lows = lows.float()
                    sim_closes = closes.float()
                    sim_opens = opens.float() if opens is not None else None
                    sim_buy = actions["buy_price"].float()
                    sim_sell = actions["sell_price"].float()
                else:
                    trade_intensity = actions["trade_amount"] / scale
                    buy_intensity = actions["buy_amount"] / scale
                    sell_intensity = actions["sell_amount"] / scale
                    sim_highs = highs
                    sim_lows = lows
                    sim_closes = closes
                    sim_opens = opens
                    sim_buy = actions["buy_price"]
                    sim_sell = actions["sell_price"]

                sim_kwargs = {
                    "highs": sim_highs,
                    "lows": sim_lows,
                    "closes": sim_closes,
                    "opens": sim_opens,
                    "buy_prices": sim_buy,
                    "sell_prices": sim_sell,
                    "trade_intensity": trade_intensity,
                    "buy_trade_intensity": buy_intensity,
                    "sell_trade_intensity": sell_intensity,
                    "maker_fee": batch.get("maker_fee", self.config.maker_fee),
                    "initial_cash": self.config.initial_cash,
                    "can_short": batch.get("can_short", False),
                    "can_long": batch.get("can_long", True),
                    "max_leverage": self.config.max_leverage,
                    "market_order_entry": self.config.market_order_entry,
                    "fill_buffer_pct": self._get_fill_buffer(current_epoch),
                    "margin_annual_rate": float(self.config.margin_annual_rate),
                }
                base_lag = int(self.config.decision_lag_bars)
                lag_range_str = self.config.decision_lag_range.strip()
                lag_list = [int(x) for x in lag_range_str.split(",") if x.strip()] if lag_range_str else [base_lag]

                sim_fn = simulate_hourly_trades_fast if use_vsim else simulate_hourly_trades

                if train and len(lag_list) > 1:
                    all_losses = []
                    all_scores = []
                    all_sortinos = []
                    all_returns_val = []
                    for lag_i in lag_list:
                        sim_i = sim_fn(
                            **sim_kwargs,
                            temperature=float(self.config.fill_temperature),
                            decision_lag_bars=lag_i,
                        )
                        ret_i = sim_i.returns.float()
                        ppy = batch.get("periods_per_year", None)
                        if ppy is None:
                            ppy = float(self.config.periods_per_year or HOURLY_PERIODS_PER_YEAR)
                        lo_i, sc_i, so_i, ar_i = compute_loss_by_type(
                            ret_i,
                            self.config.loss_type,
                            target_sign=self.config.sortino_target_sign,
                            periods_per_year=ppy,
                            return_weight=self.config.return_weight,
                            smoothness_penalty=self.config.smoothness_penalty,
                            dd_penalty=self.config.dd_penalty,
                            multiwindow_fractions=self.config.multiwindow_fractions,
                            multiwindow_aggregation=self.config.multiwindow_aggregation,
                        )
                        all_losses.append(lo_i)
                        all_scores.append(sc_i.detach())
                        all_sortinos.append(so_i.detach())
                        all_returns_val.append(ar_i.detach())
                    loss = sum(all_losses) / len(all_losses)
                    score = sum(all_scores) / len(all_scores)
                    sortino = sum(all_sortinos) / len(all_sortinos)
                    annual_return = sum(all_returns_val) / len(all_returns_val)
                else:
                    val_lag = max(lag_list) if not train else base_lag
                    if (not train) and bool(self.config.validation_use_binary_fills):
                        sim = simulate_hourly_trades_binary(**sim_kwargs, decision_lag_bars=val_lag)
                    else:
                        sim = sim_fn(
                            **sim_kwargs,
                            temperature=float(self.config.fill_temperature),
                            decision_lag_bars=val_lag,
                        )

            if not (train and len(lag_list) > 1):
                returns = sim.returns.float()
                periods_per_year = batch.get("periods_per_year", None)
                if periods_per_year is None:
                    periods_per_year = float(self.config.periods_per_year or HOURLY_PERIODS_PER_YEAR)
                loss, score, sortino, annual_return = compute_loss_by_type(
                    returns,
                    self.config.loss_type,
                    target_sign=self.config.sortino_target_sign,
                    periods_per_year=periods_per_year,
                    return_weight=self.config.return_weight,
                    smoothness_penalty=self.config.smoothness_penalty,
                    dd_penalty=self.config.dd_penalty,
                    multiwindow_fractions=self.config.multiwindow_fractions,
                    multiwindow_aggregation=self.config.multiwindow_aggregation,
                )

            if train and self.config.spread_penalty > 0:
                bp = actions["buy_price"]
                sp = actions["sell_price"]
                tgt = self.config.spread_target
                buy_gap = (bp - lows[..., : bp.shape[-1]]) / closes[..., : bp.shape[-1]].clamp(min=1e-4)
                sell_gap = (highs[..., : sp.shape[-1]] - sp) / closes[..., : sp.shape[-1]].clamp(min=1e-4)
                buy_pen = torch.relu(tgt - buy_gap).mean()
                sell_pen = torch.relu(tgt - sell_gap).mean()
                loss = loss + self.config.spread_penalty * (buy_pen + sell_pen)

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                    if self.config.grad_clip:
                        self._scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    self._apply_schedules(optimizer, global_step)
                    if self.config.warmup_steps and global_step < self.config.warmup_steps:
                        warmup_frac = float(global_step + 1) / float(self.config.warmup_steps)
                        for group, base_lr in zip(optimizer.param_groups, self._warmup_base_lrs):
                            group["lr"] = float(base_lr) * warmup_frac
                    self._scaler.step(optimizer)
                    self._scaler.update()
                    self._apply_cautious_weight_decay(model, self.config.muon_lr)
                else:
                    loss.backward()
                    if self.config.grad_clip:
                        clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    self._apply_schedules(optimizer, global_step)
                    if self.config.warmup_steps and global_step < self.config.warmup_steps:
                        warmup_frac = float(global_step + 1) / float(self.config.warmup_steps)
                        for group, base_lr in zip(optimizer.param_groups, self._warmup_base_lrs):
                            group["lr"] = float(base_lr) * warmup_frac
                    optimizer.step()
                    self._apply_cautious_weight_decay(model, self.config.muon_lr)
                global_step += 1

            total_loss += float(loss.detach().mean().item())
            total_score += float(score.detach().mean().item())
            total_sortino += float(sortino.detach().mean().item())
            total_return += float(annual_return.detach().mean().item())
            steps += 1

            if self.config.dry_train_steps and steps >= self.config.dry_train_steps:
                break

        if steps == 0:
            raise RuntimeError("No batches available for training/validation")
        metrics = {
            "loss": total_loss / steps,
            "score": total_score / steps,
            "sortino": total_sortino / steps,
            "return": total_return / steps,
        }
        return metrics, global_step

    def _build_amp(self):
        if not self.config.use_amp or self.device.type != "cuda":
            return nullcontext(), None
        dtype_name = str(self.config.amp_dtype or "bfloat16").lower()
        if dtype_name in {"float16", "fp16"}:
            dtype = torch.float16
            use_scaler = True
        else:
            dtype = torch.bfloat16
            use_scaler = False

        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            amp_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
        else:  # pragma: no cover - legacy fallback
            amp_ctx = torch.cuda.amp.autocast(dtype=dtype)

        scaler = None
        if use_scaler:
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                try:
                    scaler = torch.amp.GradScaler(device_type="cuda")
                except TypeError:  # pragma: no cover - older signature
                    scaler = torch.amp.GradScaler()
            else:  # pragma: no cover - legacy fallback
                scaler = torch.cuda.amp.GradScaler()
        return amp_ctx, scaler

    def _iter_param_groups(self, optimizer: torch.optim.Optimizer) -> list[dict]:
        if isinstance(optimizer, MultiOptim):
            groups: list[dict] = []
            for opt in optimizer.optimizers:
                groups.extend(opt.param_groups)
            return groups
        return list(optimizer.param_groups)

    def _apply_schedules(self, optimizer: torch.optim.Optimizer, global_step: int) -> None:
        if self._total_train_steps <= 0:
            return
        import math

        progress = min(max(global_step / float(self._total_train_steps), 0.0), 1.0)

        lr_sched = (self.config.lr_schedule or "none").lower()
        if lr_sched == "cosine":
            warmup = self.config.warmup_steps
            min_r = self.config.lr_min_ratio
            if global_step >= warmup:
                decay_progress = (global_step - warmup) / max(1, self._total_train_steps - warmup)
                decay_progress = min(decay_progress, 1.0)
                lr_mult = min_r + (1.0 - min_r) * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
                for group, base_lr in zip(self._iter_param_groups(optimizer), self._warmup_base_lrs):
                    group["lr"] = float(base_lr * lr_mult)
        elif lr_sched == "linear_warmdown":
            warmup = self.config.warmup_steps
            wd_ratio = self.config.lr_warmdown_ratio
            min_r = self.config.lr_min_ratio
            wd_start = int(self._total_train_steps * (1.0 - wd_ratio))
            if global_step >= warmup and global_step >= wd_start:
                wd_progress = (global_step - wd_start) / max(1, self._total_train_steps - wd_start)
                wd_progress = min(wd_progress, 1.0)
                lr_mult = 1.0 - (1.0 - min_r) * wd_progress
                for group, base_lr in zip(self._iter_param_groups(optimizer), self._warmup_base_lrs):
                    group["lr"] = float(base_lr * lr_mult)

        schedule = (self.config.weight_decay_schedule or "none").lower()
        if schedule in {"linear", "linear_to_zero", "linear_decay"}:
            wd_end = float(self.config.weight_decay_end)
            for group, base_wd in self._weight_decay_groups:
                if base_wd <= 0:
                    continue
                group["weight_decay"] = float(base_wd + (wd_end - base_wd) * progress)

        if self.config.muon_momentum_start is not None and self.config.muon_momentum_warmup_steps > 0:
            start = float(self.config.muon_momentum_start)
            end = float(self.config.muon_momentum)
            frac = min(global_step / float(self.config.muon_momentum_warmup_steps), 1.0)
            momentum = start + (end - start) * frac
            for group in self._iter_param_groups(optimizer):
                if "momentum" in group:
                    group["momentum"] = momentum

        if self.config.cooldown_fraction > 0 and self._total_train_steps > 0:
            cooldown_start = int(self._total_train_steps * (1.0 - self.config.cooldown_fraction))
            if global_step >= cooldown_start:
                cooldown_steps = max(1, self._total_train_steps - cooldown_start)
                frac = min((global_step - cooldown_start) / float(cooldown_steps), 1.0)
                target = float(self.config.muon_momentum_end)
                peak = float(self.config.muon_momentum)
                for group in self._iter_param_groups(optimizer):
                    if "momentum" in group:
                        group["momentum"] = peak + (target - peak) * frac

    def _save_checkpoint(self, model: BinancePolicyBase, epoch: int, metrics: dict[str, float]) -> Path:
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        payload = {
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "epoch": epoch,
            "config": serialize_for_checkpoint(self.config),
            "normalizer": self.data.normalizer.to_dict(),
            "feature_columns": list(self.data.feature_columns),
        }
        torch.save(payload, path)
        return path

    def _build_optimizer(self, model: BinancePolicyBase) -> torch.optim.Optimizer:
        name = (self.config.optimizer_name or "adamw").lower()
        if name in {"muon", "muon_mix", "dual"}:
            if not MUON_AVAILABLE:
                logger.warning("Muon not available; falling back to AdamW.")
                return self._build_adamw(model)
            muon_params, adam_groups = self._split_muon_params(model, self.config)
            optimizers: list[torch.optim.Optimizer] = []
            if adam_groups:
                fused = self.device.type == "cuda"
                optimizers.append(torch.optim.AdamW(adam_groups, lr=self.config.learning_rate, fused=fused))
            if muon_params:
                muon_wd = 0.0 if self.config.cautious_weight_decay else self.config.weight_decay
                optimizers.append(
                    Muon(  # type: ignore[call-arg]
                        muon_params,
                        lr=self.config.muon_lr,
                        momentum=self.config.muon_momentum,
                        nesterov=self.config.muon_nesterov,
                        ns_steps=self.config.muon_ns_steps,
                        weight_decay=muon_wd,
                        adjust_lr_fn="original",
                    )
                )
            if not optimizers:
                logger.warning("No trainable params; defaulting to AdamW.")
                return self._build_adamw(model)
            logger.info("Muon: %d matrix params, %d adam groups", len(muon_params), len(adam_groups))
            if len(optimizers) == 1:
                return optimizers[0]
            return MultiOptim(optimizers)

        return self._build_adamw(model)

    def _build_adamw(self, model: BinancePolicyBase) -> torch.optim.Optimizer:
        fused = self.device.type == "cuda"
        _, adam_groups = self._split_muon_params(model, self.config, muon=False)
        if not adam_groups:
            return torch.optim.AdamW(
                model.parameters(), lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay, fused=fused,
            )
        return torch.optim.AdamW(adam_groups, lr=self.config.learning_rate, fused=fused)

    @staticmethod
    def _split_muon_params(
        model: BinancePolicyBase, config: "TrainingConfig", *, muon: bool = True
    ) -> tuple[list[torch.Tensor], list[dict]]:
        muon_params: list[torch.Tensor] = []
        embed_params: list[torch.Tensor] = []
        head_params: list[torch.Tensor] = []
        matrix_params: list[torch.Tensor] = []
        other_params: list[torch.Tensor] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "embed" in name:
                embed_params.append(param)
            elif "head" in name:
                head_params.append(param)
            elif param.ndim == 2:
                if muon:
                    muon_params.append(param)
                else:
                    matrix_params.append(param)
            else:
                other_params.append(param)
        adam_groups: list[dict] = []
        embed_wd = config.embed_weight_decay if config.embed_weight_decay is not None else config.weight_decay
        head_wd = config.head_weight_decay if config.head_weight_decay is not None else 0.0
        if matrix_params:
            adam_groups.append({"params": matrix_params, "lr": config.learning_rate, "weight_decay": config.weight_decay})
        if embed_params:
            adam_groups.append({"params": embed_params, "lr": config.learning_rate * config.embed_lr_mult, "weight_decay": embed_wd})
        if head_params:
            adam_groups.append({"params": head_params, "lr": config.learning_rate * config.head_lr_mult, "weight_decay": head_wd})
        if other_params:
            adam_groups.append({"params": other_params, "lr": config.learning_rate, "weight_decay": 0.0})
        return muon_params, adam_groups

    def _apply_cautious_weight_decay(self, model: BinancePolicyBase, lr: float) -> None:
        if not self.config.cautious_weight_decay or self.config.weight_decay <= 0:
            return
        wd = self.config.weight_decay
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is None or p.ndim <= 1:
                    continue
                if "embed" in name or "head" in name:
                    continue
                mask = (p.grad * p).gt(0).float()
                p.mul_(1.0 - lr * wd * mask)


__all__ = ["BinanceHourlyTrainer", "TrainingArtifacts", "TrainingHistoryEntry"]
