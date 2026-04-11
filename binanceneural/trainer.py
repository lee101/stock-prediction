from __future__ import annotations

import logging
import json
import os
import shutil
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
    simulate_rebalance,
    simulate_rebalance_fast,
)
try:
    from trainingefficiency.fast_differentiable_sim import (
        simulate_hourly_trades_fast,
        simulate_hourly_trades_compiled,
    )
except ImportError:
    simulate_hourly_trades_fast = None
    simulate_hourly_trades_compiled = None
try:
    from trainingefficiency.triton_sim_kernel import simulate_hourly_trades_triton
    HAS_TRITON_SIM = True
except ImportError:
    simulate_hourly_trades_triton = None
    HAS_TRITON_SIM = False
try:
    from trainingefficiency.compiled_sim_loss import compiled_sim_and_loss
    HAS_COMPILED_SIM_LOSS = True
except ImportError:
    compiled_sim_and_loss = None
    HAS_COMPILED_SIM_LOSS = False
from torch.nn.utils import clip_grad_norm_  # type: ignore
from traininglib.optim_factory import MultiOptim

from src.checkpoint_manager import TopKCheckpointManager
from src.serialization_utils import serialize_for_checkpoint
from src.torch_device_utils import should_auto_fallback_to_cpu
from src.torch_load_utils import torch_load_compat
from wandboard import WandBoardLogger

from .config import TrainingConfig
from .data import BinanceHourlyDataModule, FeatureNormalizer, MultiSymbolDataModule
from .model import BinancePolicyBase, PolicyConfig, build_policy


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


def _aggregate_validation_lag_metrics(
    losses: list[torch.Tensor],
    scores: list[torch.Tensor],
    sortinos: list[torch.Tensor],
    annual_returns: list[torch.Tensor],
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = str(mode or "minimax").strip().lower()
    if normalized == "mean":
        n = max(1, len(losses))
        return (
            sum(losses) / n,
            sum(scores) / n,
            sum(sortinos) / n,
            sum(annual_returns) / n,
        )
    if normalized != "minimax":
        normalized = "minimax"
    return (
        torch.stack(losses, dim=0).amax(dim=0),
        torch.stack(scores, dim=0).amin(dim=0),
        torch.stack(sortinos, dim=0).amin(dim=0),
        torch.stack(annual_returns, dim=0).amin(dim=0),
    )


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
        if str(config.trainer_backend or "torch") != "torch":
            raise ValueError("BinanceHourlyTrainer requires trainer_backend='torch'")
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

    def _should_fallback_to_cpu(self, exc: BaseException) -> bool:
        return should_auto_fallback_to_cpu(self.config.device, self.device, exc)

    def _fallback_model_to_cpu(self, model: BinancePolicyBase, *, context: str, exc: BaseException) -> BinancePolicyBase:
        logger.warning(
            "Falling back to CPU for BinanceHourlyTrainer because %s was unavailable during %s: %s",
            self.device,
            context,
            exc,
        )
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        self.device = torch.device("cpu")
        return model.to(self.device)

    def _move_model_to_device(self, model: BinancePolicyBase) -> BinancePolicyBase:
        try:
            return model.to(self.device)
        except Exception as exc:
            if self._should_fallback_to_cpu(exc):
                return self._fallback_model_to_cpu(model, context="model initialization", exc=exc)
            raise

    def _probe_model_runtime_device(self, model: BinancePolicyBase) -> BinancePolicyBase:
        if self.config.device is not None or self.device.type != "cuda":
            return model
        probe_loader = self.data.train_dataloader(self.config.batch_size, 0)
        try:
            probe_batch = next(iter(probe_loader))
        except StopIteration:
            logger.info("Skipping initial CUDA forward probe because train_dataloader() yielded no batches.")
            return model
        if probe_batch is None:
            logger.info("Skipping initial CUDA forward probe because the first training batch was None.")
            return model
        if not isinstance(probe_batch, dict):
            logger.info(
                "Skipping initial CUDA forward probe because the first training batch had unexpected type %s.",
                type(probe_batch).__name__,
            )
            return model
        if "features" not in probe_batch:
            logger.info(
                "Skipping initial CUDA forward probe because the first training batch keys %s did not include 'features'.",
                sorted(probe_batch.keys()),
            )
            return model

        was_training = model.training
        try:
            model.eval()
            with torch.inference_mode():
                features = probe_batch["features"].to(self.device, non_blocking=False)
                model(features)
            return model
        except Exception as exc:
            if self._should_fallback_to_cpu(exc):
                return self._fallback_model_to_cpu(model, context="the initial CUDA forward pass", exc=exc)
            raise
        finally:
            model.train(was_training)

    def _wandb_enabled(self) -> bool:
        return bool(self.config.wandb_project or os.getenv("WANDB_PROJECT"))

    def _wandb_tags(self) -> tuple[str, ...]:
        raw = str(getattr(self.config, "wandb_tags", "") or "")
        return tuple(token.strip() for token in raw.split(",") if token.strip())

    def _marketsim_eval(self, model: torch.nn.Module, epoch: int) -> dict | None:
        """Run full compounded PnL eval every N epochs."""
        try:
            import numpy as np
        except ImportError:
            return None

        val_frame = self.data.val_dataset.frame.copy()
        if len(val_frame) < self.config.sequence_length + 10:
            return None

        try:
            sd = {k: v.cpu() for k, v in model.state_dict().items()}
            from .model import build_policy
            policy_cfg = self._policy_cfg
            cpu_model = build_policy(policy_cfg).cpu().eval()
            cpu_model.load_state_dict(sd, strict=False)
        except Exception:
            return None

        use_rebal = bool(self.config.use_rebalance_sim)

        if use_rebal:
            results = self._rebalance_marketsim_eval(cpu_model, val_frame, np)
        else:
            results = self._limit_order_marketsim_eval(cpu_model, val_frame, np)

        del cpu_model, sd

        if results:
            parts = []
            for lag in [0, 1, 2]:
                ret = results.get(f"msim_lag{lag}_ret", 0)
                srt = results.get(f"msim_lag{lag}_sort", 0)
                parts.append(f"lag{lag}:{ret:+.2%}/s{srt:+.1f}")
            mean_a = results.get("msim_mean_alloc", 0)
            turnover = results.get("msim_turnover", 0)
            print(f"  [MarketsimEval ep{epoch}] {' | '.join(parts)} alloc={mean_a:.2f} turn={turnover:.2f}", flush=True)
        return results

    def _rebalance_marketsim_eval(self, cpu_model, val_frame, np) -> dict:
        """Eval using position-target rebalance sim on full val window."""
        feature_columns = list(self.data.feature_columns)
        normalizer = self.data.normalizer
        seq_len = self.config.sequence_length

        allow_short = bool(getattr(self.config, "rebalance_allow_short", False))
        alloc_fn = torch.tanh if allow_short else torch.sigmoid
        allocations = []
        for start in range(0, len(val_frame) - seq_len + 1):
            window = val_frame.iloc[start:start + seq_len]
            feats = normalizer.transform(window[feature_columns].values)
            feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            with torch.inference_mode():
                outputs = cpu_model(feats_t)
                alloc_logits = outputs.get("allocation_logits", outputs.get("buy_amount_logits"))
                alloc = float(alloc_fn(alloc_logits[0, -1, 0]))
            allocations.append(alloc)

        aligned = val_frame.iloc[seq_len - 1:seq_len - 1 + len(allocations)]
        closes_t = torch.tensor(aligned["close"].values, dtype=torch.float32)
        opens_t = torch.tensor(aligned["open"].values, dtype=torch.float32) if "open" in aligned.columns else None
        alloc_t = torch.tensor(allocations, dtype=torch.float32)

        alloc_np = alloc_t.numpy()
        results = {
            "msim_mean_alloc": float(np.mean(alloc_np)),
            "msim_turnover": float(np.mean(np.abs(np.diff(alloc_np)))),
        }
        for lag in [0, 1, 2]:
            try:
                result = simulate_rebalance(
                    closes=closes_t, opens=opens_t, allocation=alloc_t,
                    maker_fee=self.config.maker_fee, initial_cash=10_000.0,
                    decision_lag_bars=lag, allow_short=allow_short,
                )
                vals = result.portfolio_values.numpy()
                total_return = float((vals[-1] - 10000.0) / 10000.0)
                returns_np = result.returns.numpy()
                neg_r = returns_np[returns_np < 0]
                ds = float(np.std(neg_r)) if len(neg_r) > 0 else 1e-8
                sortino = float(np.mean(returns_np)) / max(ds, 1e-8) * np.sqrt(8760)
                peak = np.maximum.accumulate(vals)
                dd = float(((vals - peak) / np.clip(peak, 1e-8, None)).min())
                results[f"msim_lag{lag}_ret"] = total_return
                results[f"msim_lag{lag}_sort"] = sortino
                results[f"msim_lag{lag}_dd"] = dd
                results[f"msim_lag{lag}_trades"] = int((result.executed_buys > 0).sum() + (result.executed_sells > 0).sum())
            except Exception:
                pass
        return results

    def _limit_order_marketsim_eval(self, cpu_model, val_frame, np) -> dict:
        """Eval using limit-order BinanceMarketSimulator."""
        try:
            from .inference import generate_actions_from_frame
            from .marketsimulator import BinanceMarketSimulator, SimulationConfig
        except ImportError:
            return {}

        primary_horizon = getattr(self.data, "primary_horizon", 1)
        with torch.inference_mode():
            actions_df = generate_actions_from_frame(
                model=cpu_model, frame=val_frame,
                feature_columns=self.data.feature_columns,
                normalizer=self.data.normalizer,
                sequence_length=self.config.sequence_length,
                horizon=primary_horizon, device=torch.device("cpu"),
            )
        if actions_df.empty:
            return {}

        results = {}
        for lag in [0, 1, 2]:
            sim_cfg = SimulationConfig(
                maker_fee=self.config.maker_fee,
                fill_buffer_bps=self.config.fill_buffer_pct * 10000,
                decision_lag_bars=lag,
                max_hold_hours=int(self.config.max_hold_hours),
                initial_cash=10_000.0,
                market_order_entry=self.config.market_order_entry,
            )
            try:
                sim = BinanceMarketSimulator(sim_cfg)
                result = sim.run(val_frame, actions_df)
                m = result.metrics
                equity = result.combined_equity.to_numpy(dtype=float)
                peak = np.maximum.accumulate(equity)
                dd = float(((equity - peak) / np.clip(peak, 1e-8, None)).min())
                n_trades = sum(len(sr.trades) for sr in result.per_symbol.values())
                results[f"msim_lag{lag}_ret"] = m.get("total_return", 0.0)
                results[f"msim_lag{lag}_sort"] = m.get("sortino", 0.0)
                results[f"msim_lag{lag}_dd"] = dd
                results[f"msim_lag{lag}_trades"] = n_trades
            except Exception:
                pass
        return results

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
        if bool(self.config.use_compile) and hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
            torch._dynamo.config.suppress_errors = True
        if self.config.use_tf32:
            if hasattr(torch, "set_float32_matmul_precision"):
                try:  # pragma: no cover - compatibility helper handles torch version quirks
                    from src.torch_backend import maybe_set_float32_precision

                    maybe_set_float32_precision(torch, "high")
                except Exception:
                    pass
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
            attention_backend=self.config.attention_backend,
            flex_block_size=self.config.flex_block_size,
            use_residual_scalars=self.config.use_residual_scalars,
            residual_scale_init=self.config.residual_scale_init,
            skip_scale_init=self.config.skip_scale_init,
            use_value_embedding=self.config.use_value_embedding,
            value_embedding_every=self.config.value_embedding_every,
            value_embedding_scale=self.config.value_embedding_scale,
            use_midpoint_offsets=self.config.use_midpoint_offsets,
            num_outputs=self.config.num_outputs,
            max_hold_hours=self.config.max_hold_hours,
            num_memory_tokens=self.config.num_memory_tokens,
            dilated_strides=self.config.dilated_strides,
            use_flex_attention=self.config.use_flex_attention,
        )
        self._policy_cfg = policy_cfg
        model = self._move_model_to_device(build_policy(policy_cfg))

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

        model = self._probe_model_runtime_device(model)

        if self.config.use_compile and hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

        optimizer = self._build_optimizer(model)
        self._warmup_base_lrs = [group.get("lr", self.config.learning_rate) for group in optimizer.param_groups]
        self._weight_decay_groups = [
            (group, float(group.get("weight_decay", 0.0))) for group in self._iter_param_groups(optimizer)
        ]
        self._amp_context, self._scaler = self._build_amp()

        if self.device.type == "cuda" and hasattr(self.data, "gpu_cached_dataloader"):
            try:
                train_loader = self.data.gpu_cached_dataloader("train", self.config.batch_size, self.device, shuffle=True)
                val_loader = self.data.gpu_cached_dataloader("val", self.config.batch_size, self.device, shuffle=False)
                logger.info("Using GPU-cached dataloaders")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"GPU cache OOM, falling back to CPU streaming: {e}")
                    torch.cuda.empty_cache()
                    train_loader = self.data.train_dataloader(self.config.batch_size, self.config.num_workers)
                    val_loader = self.data.val_dataloader(self.config.batch_size, self.config.num_workers)
                else:
                    raise
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
        best_metric = float("-inf")
        ckpt_mgr = TopKCheckpointManager(
            self.checkpoint_dir,
            max_keep=max(1, int(self.config.top_k_checkpoints)),
            mode="max",
        )

        global_step = 0
        print(f"Training: {len(train_loader)} train batches, {len(val_loader)} val batches", flush=True)
        with WandBoardLogger(
            run_name=self.config.run_name,
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            group=self.config.wandb_group,
            tags=self._wandb_tags(),
            notes=self.config.wandb_notes,
            mode=self.config.wandb_mode,
            enable_wandb=self._wandb_enabled(),
            log_dir=self.config.log_dir,
            tensorboard_subdir=self.config.run_name,
            config=serialize_for_checkpoint(self.config),
            log_metrics=self.config.wandb_log_metrics,
        ) as metrics_logger:
            metrics_logger.log_text(
                "train/feature_columns",
                json.dumps(list(self.data.feature_columns)),
                step=0,
            )
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

                checkpoint_metric, checkpoint_metric_name, generalization_gap = self._checkpoint_metric(train_metrics, val_metrics)
                ckpt_path = self._save_checkpoint(model, epoch, val_metrics)
                ckpt_mgr.register(ckpt_path, checkpoint_metric, epoch=epoch)
                if val_metrics["score"] > best_score:
                    best_score = val_metrics["score"]
                if checkpoint_metric > best_metric:
                    best_metric = checkpoint_metric
                    best_checkpoint = ckpt_path
                    self._refresh_best_checkpoint_alias(ckpt_path)

                self._write_progress(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    checkpoint_metric=checkpoint_metric,
                    checkpoint_metric_name=checkpoint_metric_name,
                    generalization_gap=generalization_gap,
                    best_metric=best_metric,
                    best_checkpoint=best_checkpoint,
                )

                print(
                    f"Epoch {epoch}/{self.config.epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} Score: {train_metrics['score']:.4f} "
                    f"Sortino: {train_metrics['sortino']:.4f} Return: {train_metrics['return']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} Score: {val_metrics['score']:.4f} "
                    f"Sortino: {val_metrics['sortino']:.4f} Return: {val_metrics['return']:.4f} | "
                    f"Gap: {generalization_gap:.4f} {checkpoint_metric_name}: {checkpoint_metric:.4f}"
                )

                if epoch % 5 == 0 or epoch == 1:
                    try:
                        msim_metrics = self._marketsim_eval(model, epoch)
                    except Exception as e:
                        print(f"  [MarketsimEval ep{epoch}] skipped: {e}", flush=True)
                        msim_metrics = None
                    if msim_metrics:
                        msim_path = self.checkpoint_dir / "marketsim_eval.json"
                        existing = []
                        if msim_path.exists():
                            try:
                                existing = json.loads(msim_path.read_text())
                            except Exception:
                                pass
                        existing.append({"epoch": epoch, **msim_metrics})
                        msim_path.write_text(json.dumps(existing, indent=2))

                lr_now = optimizer.param_groups[0]["lr"] if hasattr(optimizer, "param_groups") else self.config.learning_rate
                metrics_logger.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/sortino": train_metrics["sortino"],
                    "train/return": train_metrics["return"],
                    "train/score": train_metrics["score"],
                    "val/loss": val_metrics["loss"],
                    "val/sortino": val_metrics["sortino"],
                    "val/return": val_metrics["return"],
                    "val/score": val_metrics["score"],
                    "val/generalization_gap": generalization_gap,
                    f"val/{checkpoint_metric_name}": checkpoint_metric,
                    "learning_rate": lr_now,
                }, step=epoch)

            if history:
                best_entry = max(history, key=lambda item: item.val_score or float("-inf"))
                metrics_logger.log_hparams(
                    {
                        "run_name": self.config.run_name,
                        "epochs": int(self.config.epochs),
                        "batch_size": int(self.config.batch_size),
                        "sequence_length": int(self.config.sequence_length),
                        "transformer_dim": int(self.config.transformer_dim),
                        "transformer_heads": int(self.config.transformer_heads),
                        "transformer_layers": int(self.config.transformer_layers),
                        "learning_rate": float(self.config.learning_rate),
                        "weight_decay": float(self.config.weight_decay),
                        "fill_temperature": float(self.config.fill_temperature),
                        "return_weight": float(self.config.return_weight),
                        "checkpoint_metric": str(self.config.checkpoint_metric),
                    },
                    {
                        "best/val_score": best_entry.val_score or float("-inf"),
                        "best/val_sortino": best_entry.val_sortino or float("-inf"),
                        "best/val_return": best_entry.val_return or float("-inf"),
                    },
                    step=best_entry.epoch,
                    table_name="torch_train_summary",
                )

        return TrainingArtifacts(
            state_dict=model.state_dict(),
            normalizer=self.data.normalizer,
            history=history,
            feature_columns=list(self.data.feature_columns),
            config=self.config,
            checkpoint_paths=list(self.checkpoint_dir.glob("*.pt")),
            best_checkpoint=best_checkpoint,
        )

    def _checkpoint_metric(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> tuple[float, str, float]:
        metric_name = str(self.config.checkpoint_metric or "robust_score").strip().lower()
        gap_penalty = float(self.config.checkpoint_gap_penalty)
        score_gap = max(float(train_metrics["score"]) - float(val_metrics["score"]), 0.0)
        sortino_gap = max(float(train_metrics["sortino"]) - float(val_metrics["sortino"]), 0.0)
        if metric_name == "val_score":
            return float(val_metrics["score"]), metric_name, score_gap
        if metric_name == "val_sortino":
            return float(val_metrics["sortino"]), metric_name, sortino_gap
        if metric_name == "val_return":
            return float(val_metrics["return"]), metric_name, score_gap
        if metric_name == "robust_sortino":
            metric = float(val_metrics["sortino"]) - gap_penalty * sortino_gap
            return metric, metric_name, sortino_gap
        metric = float(val_metrics["score"]) - gap_penalty * score_gap
        return metric, "robust_score", score_gap

    def _refresh_best_checkpoint_alias(self, checkpoint_path: Path) -> None:
        alias_path = self.checkpoint_dir / "best.pt"
        try:
            if alias_path.exists() or alias_path.is_symlink():
                alias_path.unlink()
            alias_path.symlink_to(checkpoint_path.name)
        except OSError:
            shutil.copy2(checkpoint_path, alias_path)

    def _write_progress(
        self,
        *,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        checkpoint_metric: float,
        checkpoint_metric_name: str,
        generalization_gap: float,
        best_metric: float,
        best_checkpoint: Path | None,
    ) -> None:
        progress_payload = {
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_name": self.config.run_name,
            "trainer_backend": str(self.config.trainer_backend),
            "epoch": int(epoch),
            "epochs": int(self.config.epochs),
            "checkpoint_metric_name": checkpoint_metric_name,
            "checkpoint_metric": float(checkpoint_metric),
            "checkpoint_gap_penalty": float(self.config.checkpoint_gap_penalty),
            "generalization_gap": float(generalization_gap),
            "train_metrics": {key: float(value) for key, value in train_metrics.items()},
            "val_metrics": {key: float(value) for key, value in val_metrics.items()},
            "best_metric": float(best_metric),
            "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        }
        path = self.checkpoint_dir / "training_progress.json"
        path.write_text(json.dumps(progress_payload, indent=2))

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
        device = self.device
        loss_sum = torch.zeros(1, device=device)
        score_sum = torch.zeros(1, device=device)
        sortino_sum = torch.zeros(1, device=device)
        return_sum = torch.zeros(1, device=device)
        steps = 0

        mark_step_begin = None
        if bool(self.config.use_compile):
            compiler = getattr(torch, "compiler", None)
            mark_step_begin = getattr(compiler, "cudagraph_mark_step_begin", None) if compiler is not None else None

        nb = device.type == "cuda"
        accum_steps = max(1, self.config.accumulation_steps) if train else 1

        split_amp = bool(self.config.split_amp) and self._amp_context is not None
        use_vsim = bool(self.config.use_vectorized_sim) and simulate_hourly_trades_fast is not None
        sim_context = nullcontext() if split_amp else self._amp_context
        scale = float(self.config.trade_amount_scale)
        base_lag = int(self.config.decision_lag_bars)
        lag_range_str = self.config.decision_lag_range.strip()
        lag_list = [int(x) for x in lag_range_str.split(",") if x.strip()] if lag_range_str else [base_lag]

        if HAS_TRITON_SIM and device.type == "cuda":
            sim_fn = simulate_hourly_trades_triton
        elif use_vsim:
            sim_fn = simulate_hourly_trades_fast
        else:
            sim_fn = simulate_hourly_trades

        grad_context = torch.inference_mode() if not train else nullcontext()
        with grad_context:
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(loader):
                if mark_step_begin is not None:
                    mark_step_begin()
                features = batch["features"].to(device, non_blocking=nb)
                if train and self.config.feature_noise_std > 0:
                    features = features + self.config.feature_noise_std * torch.randn_like(features)
                opens = batch["open"].to(device, non_blocking=nb) if "open" in batch else None
                highs = batch["high"].to(device, non_blocking=nb)
                lows = batch["low"].to(device, non_blocking=nb)
                closes = batch["close"].to(device, non_blocking=nb)
                reference_close = batch["reference_close"].to(device, non_blocking=nb)
                chronos_high = batch["chronos_high"].to(device, non_blocking=nb)
                chronos_low = batch["chronos_low"].to(device, non_blocking=nb)

                with self._amp_context:
                    outputs = model(features)
                    actions = model.decode_actions(
                        outputs,
                        reference_close=reference_close,
                        chronos_high=chronos_high,
                        chronos_low=chronos_low,
                    )

                with sim_context:
                    use_rebal = bool(self.config.use_rebalance_sim)

                    if use_rebal:
                        allow_short = bool(getattr(self.config, "rebalance_allow_short", False))
                        alloc_raw = actions.get("allocation_fraction")
                        if alloc_raw is None:
                            alloc_logits = outputs.get("allocation_logits", outputs.get("buy_amount_logits"))
                            if allow_short:
                                alloc_raw = torch.tanh(alloc_logits.squeeze(-1))
                            else:
                                alloc_raw = torch.sigmoid(alloc_logits.squeeze(-1))
                        regime_mode = bool(getattr(self.config, "rebalance_regime_mode", False))
                        if regime_mode:
                            alloc_raw = alloc_raw[..., -1:].expand_as(alloc_raw)
                        sim_alloc = alloc_raw.float() if split_amp else alloc_raw
                        sim_closes_r = closes.float() if split_amp else closes
                        sim_opens_r = (opens.float() if split_amp else opens) if opens is not None else None
                        rebal_base = {
                            "closes": sim_closes_r,
                            "opens": sim_opens_r,
                            "allocation": sim_alloc,
                            "maker_fee": batch.get("maker_fee", self.config.maker_fee),
                            "initial_cash": self.config.initial_cash,
                            "margin_annual_rate": float(self.config.margin_annual_rate),
                            "max_leverage": self.config.max_leverage,
                            "allow_short": allow_short,
                        }

                        def _run_sim(lag_i, is_val=False):
                            return simulate_rebalance(**rebal_base, decision_lag_bars=lag_i)
                    else:
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

                        def _run_sim(lag_i, is_val=False):
                            if is_val and bool(self.config.validation_use_binary_fills):
                                return simulate_hourly_trades_binary(**sim_kwargs, decision_lag_bars=lag_i)
                            return sim_fn(**sim_kwargs, temperature=float(self.config.fill_temperature), decision_lag_bars=lag_i)

                    if train and len(lag_list) > 1:
                        all_losses = []
                        all_scores = []
                        all_sortinos = []
                        all_returns_val = []
                        for lag_i in lag_list:
                            sim_i = _run_sim(lag_i, is_val=False)
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
                            all_scores.append(sc_i)
                            all_sortinos.append(so_i)
                            all_returns_val.append(ar_i)
                        loss = sum(all_losses) / len(all_losses)
                        with torch.no_grad():
                            score = sum(all_scores) / len(all_scores)
                            sortino = sum(all_sortinos) / len(all_sortinos)
                            annual_return = sum(all_returns_val) / len(all_returns_val)
                    elif (not train) and len(lag_list) > 1:
                        val_losses = []
                        val_scores = []
                        val_sortinos = []
                        val_returns = []
                        for lag_i in lag_list:
                            sim_i = _run_sim(lag_i, is_val=True)
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
                            val_losses.append(lo_i)
                            val_scores.append(sc_i)
                            val_sortinos.append(so_i)
                            val_returns.append(ar_i)
                        loss, score, sortino, annual_return = _aggregate_validation_lag_metrics(
                            val_losses,
                            val_scores,
                            val_sortinos,
                            val_returns,
                            mode=str(getattr(self.config, "validation_lag_aggregation", "minimax")),
                        )
                    else:
                        val_lag = max(lag_list) if not train else base_lag
                        sim = _run_sim(val_lag, is_val=not train)

                if not ((train and len(lag_list) > 1) or ((not train) and len(lag_list) > 1)):
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

                if train and self.config.spread_penalty > 0 and not use_rebal:
                    bp = actions["buy_price"]
                    sp = actions["sell_price"]
                    tgt = self.config.spread_target
                    buy_gap = (bp - lows[..., : bp.shape[-1]]) / closes[..., : bp.shape[-1]].clamp(min=1e-4)
                    sell_gap = (highs[..., : sp.shape[-1]] - sp) / closes[..., : sp.shape[-1]].clamp(min=1e-4)
                    buy_pen = torch.relu(tgt - buy_gap).mean()
                    sell_pen = torch.relu(tgt - sell_gap).mean()
                    loss = loss + self.config.spread_penalty * (buy_pen + sell_pen)

                if train and use_rebal:
                    ent_w = getattr(self.config, "rebalance_entropy_weight", 0.0)
                    if ent_w > 0:
                        if allow_short:
                            p = ((sim_alloc + 1.0) / 2.0).clamp(1e-6, 1.0 - 1e-6)
                        else:
                            p = sim_alloc.clamp(1e-6, 1.0 - 1e-6)
                        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
                        loss = loss - ent_w * entropy
                    smooth_w = getattr(self.config, "rebalance_smoothness_weight", 0.0)
                    if smooth_w > 0:
                        delta_alloc = (sim_alloc[..., 1:] - sim_alloc[..., :-1]).abs().mean()
                        loss = loss + smooth_w * delta_alloc

                if train and optimizer is not None:
                    loss_scaled = loss / accum_steps
                    if self._scaler is not None:
                        self._scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()
                    if (batch_idx + 1) % accum_steps == 0:
                        global_step = self._optimizer_step(model, optimizer, global_step)

                with torch.no_grad():
                    loss_sum += loss.detach().mean()
                    score_sum += score.detach().mean()
                    sortino_sum += sortino.detach().mean()
                    return_sum += annual_return.detach().mean()
                steps += 1

                if self.config.dry_train_steps and steps >= self.config.dry_train_steps:
                    break

            if train and optimizer is not None and steps % accum_steps != 0:
                global_step = self._optimizer_step(model, optimizer, global_step, flush=True)

        if steps == 0:
            raise RuntimeError("No batches available for training/validation")
        n = float(steps)
        metrics = {
            "loss": float(loss_sum.item()) / n,
            "score": float(score_sum.item()) / n,
            "sortino": float(sortino_sum.item()) / n,
            "return": float(return_sum.item()) / n,
        }
        return metrics, global_step

    def _optimizer_step(
        self,
        model: BinancePolicyBase,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        flush: bool = False,
    ) -> int:
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)
        if self.config.grad_clip:
            clip_grad_norm_(model.parameters(), self.config.grad_clip)
        if not flush:
            self._apply_schedules(optimizer, global_step)
            if self.config.warmup_steps and global_step < self.config.warmup_steps:
                warmup_frac = float(global_step + 1) / float(self.config.warmup_steps)
                for group, base_lr in zip(optimizer.param_groups, self._warmup_base_lrs):
                    group["lr"] = float(base_lr) * warmup_frac
        if self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()
        if not flush:
            self._apply_cautious_weight_decay(model, self.config.muon_lr)
        optimizer.zero_grad(set_to_none=True)
        return global_step + 1

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
