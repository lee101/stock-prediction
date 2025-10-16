#!/usr/bin/env python3
"""
Launch a longer Toto training run on GPU using the enhanced trainer.

This script configures a moderately deeper model, runs for additional epochs,
and keeps the top-4 checkpoints by validation loss for later evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from toto_trainer import TrainerConfig, DataLoaderConfig, TotoTrainer


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this training run.")

    save_dir = Path("tototraining") / "checkpoints" / "gpu_run"
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer_config = TrainerConfig(
        patch_size=12,
        stride=6,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        mlp_hidden_dim=256,
        dropout=0.1,
        spacewise_every_n_layers=2,
        scaler_cls="<class 'model.scaler.StdMeanScaler'>",
        output_distribution_classes=["<class 'model.distribution.StudentTOutput'>"],
        learning_rate=3e-4,
        weight_decay=0.01,
        batch_size=8,
        accumulation_steps=1,
        max_epochs=24,
        warmup_epochs=2,
        optimizer="adamw",
        scheduler="cosine",
        gradient_clip_val=1.0,
        use_mixed_precision=True,
        require_gpu=True,
        distributed=False,
        save_dir=str(save_dir),
        save_every_n_epochs=1,
        keep_last_n_checkpoints=8,
        best_k_checkpoints=4,
        validation_frequency=1,
        early_stopping_patience=10,
        early_stopping_delta=1e-4,
        compute_train_metrics=True,
        compute_val_metrics=True,
        metrics_log_frequency=20,
        gradient_checkpointing=False,
        memory_efficient_attention=False,
        pin_memory=False,
        log_level="INFO",
        log_file=str(save_dir / "training.log"),
        wandb_project=None,
        experiment_name="toto_gpu_run",
        log_to_tensorboard=False,
        tensorboard_log_dir="tensorboard_logs",
        export_pretrained_dir=str(save_dir / "hf_export"),
        export_on_best=False,
        random_seed=1337,
    )

    loader_config = DataLoaderConfig(
        train_data_path="trainingdata/train",
        test_data_path="trainingdata/test",
        patch_size=trainer_config.patch_size,
        stride=trainer_config.stride,
        sequence_length=120,
        prediction_length=24,
        normalization_method="robust",
        handle_missing="interpolate",
        outlier_threshold=3.0,
        batch_size=trainer_config.batch_size,
        validation_split=0.2,
        test_split_days=30,
        cv_folds=3,
        cv_gap=24,
        min_sequence_length=200,
        max_symbols=None,
        ohlc_features=["Open", "High", "Low", "Close"],
        additional_features=[],
        target_feature="Close",
        add_technical_indicators=False,
        rsi_period=14,
        ma_periods=[5, 10],
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        random_seed=1337,
    )

    trainer = TotoTrainer(trainer_config, loader_config)
    trainer.prepare_data()
    trainer.setup_model()
    trainer.train()

    val_metrics = trainer.evaluate("val")
    test_metrics = trainer.evaluate("test")

    summary_path = save_dir / "final_metrics.json"
    summary_path.write_text(
        json.dumps(
            {
                "val": val_metrics,
                "test": test_metrics,
            },
            indent=2,
        )
    )
    print("FINAL_VAL_METRICS", val_metrics)
    print("FINAL_TEST_METRICS", test_metrics)
    print(f"Saved metrics summary to {summary_path}")


if __name__ == "__main__":
    main()
