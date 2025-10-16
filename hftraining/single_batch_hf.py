#!/usr/bin/env python3
"""
Single-batch HF training + realistic profit eval

Runs one optimizer step on a single batch using the HF-style
TransformerTradingModel, then evaluates profit metrics using
ProfitTracker (with commission/slippage) to approximate realistic PnL.

Returns a concise metrics dict so higher-level scripts can compare
against the classic training variant.
"""

import os
import sys
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.append(os.path.dirname(current_dir))

from hf_trainer import HFTrainingConfig, TransformerTradingModel
from train_hf import StockDataset
from data_utils import load_training_data
from profit_tracker import ProfitTracker


def run_single_batch_hf() -> Dict[str, Any]:
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Small config for quick single-batch step
    config = HFTrainingConfig(
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        learning_rate=3e-4,
        warmup_steps=0,
        max_steps=1,
        batch_size=32,
        sequence_length=30,
        prediction_horizon=5,
        use_mixed_precision=False,
        use_data_parallel=False,
        output_dir="hftraining/output",
        logging_dir="hftraining/logs",
    )

    # Load local or synthetic features (no network dependency)
    data = load_training_data(data_dir="trainingdata")

    # Normalize
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data_norm = (data - data_mean) / (data_std + 1e-8)

    # Train subset large enough for one batch
    min_len = config.sequence_length + config.prediction_horizon + config.batch_size + 10
    if len(data_norm) < min_len:
        # Extend a bit if synthetic is small (shouldn't happen given defaults)
        import numpy as np
        reps = int((min_len // max(1, len(data_norm))) + 1)
        data_norm = np.vstack([data_norm for _ in range(reps)])

    train_data = data_norm[: max(len(data_norm) // 2, min_len)]

    # Dataset/DataLoader
    train_ds = StockDataset(
        train_data,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon,
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)

    # Model / Optimizer
    model = TransformerTradingModel(config, input_dim=data.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)

    # One batch
    batch = next(iter(train_loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    # Forward
    model.train()
    outputs = model(batch['input_ids'], attention_mask=batch.get('attention_mask'))

    # Targets: price = close column (index 3) over horizon
    price_targets = batch['labels'][:, : config.prediction_horizon, 3]
    price_pred = outputs['price_predictions']

    # Losses
    price_loss = F.mse_loss(price_pred, price_targets)
    action_loss = F.cross_entropy(outputs['action_logits'], batch['action_labels'])
    total_loss = 0.7 * price_loss + 0.3 * action_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Evaluate profit with commission/slippage
    tracker = ProfitTracker(initial_capital=10000.0, commission=0.001, slippage=0.0005)
    with torch.no_grad():
        outputs_eval = model(batch['input_ids'], attention_mask=batch.get('attention_mask'))
    eval_targets = batch['labels'][:, : config.prediction_horizon, 3]
    metrics = tracker.calculate_metrics_from_predictions(
        predictions=outputs_eval['price_predictions'],
        actual_prices=eval_targets,
        action_logits=outputs_eval['action_logits'],
    )

    # Prepare concise result
    result = {
        'actor_loss': float(action_loss.item()),
        'price_loss': float(price_loss.item()),
        'total_loss': float(total_loss.item()),
        'total_return': float(metrics.total_return),
        'sharpe_ratio': float(metrics.sharpe_ratio),
        'max_drawdown': float(metrics.max_drawdown),
        'win_rate': float(metrics.win_rate),
        'total_trades': int(metrics.total_trades),
    }

    print("HF single-batch results:", result)
    return result


if __name__ == '__main__':
    run_single_batch_hf()

