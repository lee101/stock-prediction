#!/usr/bin/env python3
"""Train unified neural policy on stocks + crypto forecasts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.data import MultiSymbolDataModule
from binanceneural.config import DatasetConfig, TrainingConfig
from loguru import logger
from src.trade_directions import DEFAULT_ALPACA_LIVE8_STOCKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-symbols", default=",".join(DEFAULT_ALPACA_LIVE8_STOCKS))
    parser.add_argument("--crypto-symbols", default="SOLUSD,AVAXUSD,ETHUSD,UNIUSD")
    parser.add_argument("--stock-data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--crypto-data-root", type=Path, default=Path("trainingdatahourly/crypto"))
    parser.add_argument("--stock-cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--crypto-cache-root", type=Path, default=Path("binanceneural/forecast_cache"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("unified_hourly_experiment/checkpoints"))
    parser.add_argument("--preload", type=Path, default=None, help="Preload checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--symbols", type=str, default=None, help="Override stock symbols")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--forecast-horizons", type=str, default="1,24", help="Comma-sep horizons")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr-schedule", type=str, default="none", choices=["none", "cosine", "linear_warmdown"])
    parser.add_argument("--lr-min-ratio", type=float, default=0.0)
    parser.add_argument("--return-weight", type=float, default=0.08)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--fill-temperature", type=float, default=5e-4)
    parser.add_argument("--logits-softcap", type=float, default=12.0)
    parser.add_argument("--smoothness-penalty", type=float, default=0.0)
    parser.add_argument("--loss-type", type=str, default="sortino", choices=["sortino", "sharpe", "calmar", "log_wealth", "sortino_dd"])
    parser.add_argument("--dd-penalty", type=float, default=1.0)
    parser.add_argument("--feature-noise-std", type=float, default=0.0)
    parser.add_argument("--use-residual-scalars", action="store_true")
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--margin-annual-rate", type=float, default=0.0)
    parser.add_argument("--decision-lag-bars", type=int, default=0)
    parser.add_argument("--decision-lag-range", type=str, default="")
    parser.add_argument("--market-order-entry", action="store_true")
    parser.add_argument("--fill-buffer-pct", type=float, default=0.0)
    parser.add_argument("--spread-penalty", type=float, default=0.0)
    parser.add_argument("--spread-target", type=float, default=0.0013)
    parser.add_argument("--fill-buffer-warmup-epochs", type=int, default=0)
    parser.add_argument("--validation-use-binary-fills", action="store_true")
    parser.add_argument("--model-arch", type=str, default="classic", choices=["classic", "nano"])
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon", "muon_mix"])
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--cooldown-fraction", type=float, default=0.0)
    parser.add_argument("--muon-momentum-end", type=float, default=0.85)
    parser.add_argument("--cautious-wd", action="store_true")
    parser.add_argument("--embed-lr-mult", type=float, default=1.0)
    parser.add_argument("--head-lr-mult", type=float, default=1.0)
    parser.add_argument("--embed-wd", type=float, default=None)
    parser.add_argument("--head-wd", type=float, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--dilated-strides", type=str, default="")
    parser.add_argument("--num-memory-tokens", type=int, default=0)
    parser.add_argument("--use-value-embedding", action="store_true")
    parser.add_argument("--value-embedding-every", type=int, default=2)
    args = parser.parse_args()

    if args.symbols:
        stocks = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        stocks = [s.strip().upper() for s in args.stock_symbols.split(",") if s.strip()]
    cryptos = [s.strip().upper() for s in args.crypto_symbols.split(",") if s.strip()]

    horizons = [int(h.strip()) for h in args.forecast_horizons.split(",")]
    logger.info("Training unified policy on {} stocks + {} crypto, horizons={}", len(stocks), len(cryptos), horizons)

    stock_config = DatasetConfig(
        symbol=stocks[0],
        data_root=str(args.stock_data_root),
        forecast_cache_root=str(args.stock_cache_root),
        forecast_horizons=horizons,
        sequence_length=args.sequence_length,
        val_fraction=0.15,
        min_history_hours=100,
        validation_days=30,
        cache_only=True,
    )

    crypto_config = DatasetConfig(
        symbol=cryptos[0] if cryptos else "SOLUSD",
        data_root=str(args.crypto_data_root),
        forecast_cache_root=str(args.crypto_cache_root),
        forecast_horizons=horizons,
        sequence_length=args.sequence_length,
        val_fraction=0.1,
        cache_only=True,
    )

    # Load stock data module
    logger.info("Loading stock data for {} symbols", len(stocks))
    stock_data = MultiSymbolDataModule(stocks, stock_config)

    # Load crypto data module if symbols provided
    crypto_data = None
    if cryptos:
        logger.info("Loading crypto data for {} symbols", len(cryptos))
        try:
            crypto_data = MultiSymbolDataModule(cryptos, crypto_config)
        except Exception as e:
            logger.warning("Failed to load crypto data: {}", e)

    # Use stock data as primary (most symbols)
    data_module = stock_data

    logger.info("Training on {} features", len(data_module.feature_columns))
    logger.info("Train samples: {}", len(data_module.train_dataset))

    train_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.sequence_length,
        checkpoint_root=args.checkpoint_root,
        run_name=args.checkpoint_name or args.run_name,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        transformer_dim=args.hidden_dim,
        transformer_heads=args.num_heads,
        transformer_layers=args.num_layers,
        model_arch=args.model_arch,
        optimizer_name=args.optimizer,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        cooldown_fraction=args.cooldown_fraction,
        muon_momentum_end=args.muon_momentum_end,
        cautious_weight_decay=args.cautious_wd,
        embed_lr_mult=args.embed_lr_mult,
        head_lr_mult=args.head_lr_mult,
        embed_weight_decay=args.embed_wd,
        head_weight_decay=args.head_wd,
        use_amp=not args.no_amp,
        num_kv_heads=args.num_kv_heads,
        dilated_strides=args.dilated_strides,
        num_memory_tokens=args.num_memory_tokens,
        use_value_embedding=args.use_value_embedding,
        value_embedding_every=args.value_embedding_every,
        transformer_dropout=args.dropout,
        lr_schedule=args.lr_schedule,
        lr_min_ratio=args.lr_min_ratio,
        return_weight=args.return_weight,
        grad_clip=args.grad_clip,
        fill_temperature=args.fill_temperature,
        logits_softcap=args.logits_softcap,
        smoothness_penalty=args.smoothness_penalty,
        loss_type=args.loss_type,
        dd_penalty=args.dd_penalty,
        feature_noise_std=args.feature_noise_std,
        use_residual_scalars=args.use_residual_scalars,
        maker_fee=args.maker_fee,
        max_leverage=args.max_leverage,
        margin_annual_rate=args.margin_annual_rate,
        decision_lag_bars=args.decision_lag_bars,
        decision_lag_range=args.decision_lag_range,
        market_order_entry=args.market_order_entry,
        fill_buffer_pct=args.fill_buffer_pct,
        spread_penalty=args.spread_penalty,
        spread_target=args.spread_target,
        fill_buffer_warmup_epochs=args.fill_buffer_warmup_epochs,
        validation_use_binary_fills=args.validation_use_binary_fills,
        preload_checkpoint_path=str(args.preload) if args.preload else None,
        use_compile=not args.no_compile,
        seed=args.seed,
    )

    trainer = BinanceHourlyTrainer(train_config, data_module)
    artifacts = trainer.train()

    logger.success("Training complete!")
    logger.info("Best checkpoint: {}", artifacts.best_checkpoint)
    logger.info("Final history: epochs={}", len(artifacts.history))

    if artifacts.history:
        last = artifacts.history[-1]
        logger.info("Last val sortino: {:.4f}, return: {:.4f}",
                   last.val_sortino or 0, last.val_return or 0)

    # Save config
    config_path = trainer.checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "stock_symbols": stocks,
            "crypto_symbols": cryptos,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "sequence_length": args.sequence_length,
            "feature_columns": data_module.feature_columns,
            "transformer_dim": args.hidden_dim,
            "transformer_heads": args.num_heads,
            "transformer_layers": args.num_layers,
            "model_arch": args.model_arch,
            "normalizer": data_module.normalizer.to_dict(),
        }, f, indent=2)

    history_rows = []
    for row in artifacts.history:
        history_rows.append({
            "epoch": row.epoch,
            "train_loss": row.train_loss,
            "train_score": row.train_score,
            "train_sortino": row.train_sortino,
            "train_return": row.train_return,
            "val_loss": row.val_loss,
            "val_score": row.val_score,
            "val_sortino": row.val_sortino,
            "val_return": row.val_return,
        })
    best = max(artifacts.history, key=lambda h: h.val_sortino or float("-inf")) if artifacts.history else None
    meta_path = trainer.checkpoint_dir / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "run_name": args.checkpoint_name or args.run_name,
            "symbols": stocks,
            "epochs": args.epochs,
            "sequence_length": args.sequence_length,
            "transformer_dim": args.hidden_dim,
            "transformer_heads": args.num_heads,
            "transformer_layers": args.num_layers,
            "model_arch": args.model_arch,
            "feature_columns": data_module.feature_columns,
            "decision_lag_bars": args.decision_lag_bars,
            "decision_lag_range": args.decision_lag_range,
            "smoothness_penalty": args.smoothness_penalty,
            "return_weight": args.return_weight,
            "fill_temperature": args.fill_temperature,
            "history": history_rows,
            "best_epoch": best.epoch if best else None,
            "best_val_sortino": best.val_sortino if best else None,
            "best_val_return": best.val_return if best else None,
            "best_checkpoint": str(artifacts.best_checkpoint) if artifacts.best_checkpoint else None,
        }, f, indent=2)

    logger.info("Config saved to {}", config_path)
    logger.info("Training metadata saved to {}", meta_path)


if __name__ == "__main__":
    main()
