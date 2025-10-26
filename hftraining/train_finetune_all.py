#!/usr/bin/env python3
"""
Train a base model on all instruments in trainingdata/, then fine-tune per instrument.

This orchestrates:
1) Aggregated training across all CSVs under trainingdata/train
2) Per-symbol fine-tuning initialized from the base checkpoint

Outputs are written under hftraining/output/ with timestamped subfolders.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Allow running as a script from repo root
this_dir = Path(__file__).parent
sys.path.insert(0, str(this_dir))
sys.path.append(str(this_dir.parent))

from train_hf import HFTrainer, StockDataset
from hf_trainer import HFTrainingConfig, TransformerTradingModel


def find_trainingdata_root() -> Path:
    """Return the path to trainingdata directory (repo_root/trainingdata)."""
    # Prefer cwd/trainingdata; fallback to sibling of hftraining
    cwd_candidate = Path.cwd() / 'trainingdata'
    if cwd_candidate.exists():
        return cwd_candidate
    sibling = this_dir.parent / 'trainingdata'
    return sibling


def load_csv_ohlc(csv_path: Path) -> Optional[np.ndarray]:
    """Load a single CSV and return OHLC (or OHLCV if consistent); returns None on failure."""
    try:
        df = pd.read_csv(csv_path)
        if {'Open', 'High', 'Low', 'Close'}.issubset(df.columns):
            cols = ['Open', 'High', 'Low', 'Close']
        elif {'open', 'high', 'low', 'close'}.issubset(df.columns):
            cols = ['open', 'high', 'low', 'close']
        else:
            # Fallback: try positional OHLCV with timestamp at first column
            if df.shape[1] >= 5:
                return pd.DataFrame(df.iloc[:, 1:5]).apply(pd.to_numeric, errors='coerce').ffill().fillna(0).values
            return None
        data = df[cols].values
        data = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').ffill().fillna(0).values
        return data
    except Exception:
        return None


def load_all_training_data(root: Path, max_files: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
    """Load and vertically stack all train/*.csv; return (data, symbols)."""
    train_dir = root / 'train'
    csvs = sorted(train_dir.glob('*.csv'))
    if max_files:
        csvs = csvs[:max_files]
    all_data: List[np.ndarray] = []
    symbols: List[str] = []
    for p in csvs:
        arr = load_csv_ohlc(p)
        if arr is None or len(arr) < 100:
            continue
        all_data.append(arr)
        symbols.append(p.stem)
    if not all_data:
        raise RuntimeError(f"No usable CSVs found in {train_dir}")
    data = np.vstack(all_data)
    return data, symbols


def split_and_normalize(data: np.ndarray, seq_len: int, pred_horizon: int,
                        mean: Optional[np.ndarray] = None,
                        std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize data (Z-score) using provided mean/std (if any) and split into train/val/test (80/10/10)."""
    if mean is None or std is None:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
    norm = (data - mean) / (std + 1e-8)
    n = len(norm)
    train_n = int(0.8 * n)
    val_n = int(0.1 * n)
    train = norm[:train_n]
    val = norm[train_n:train_n + val_n]
    test = norm[train_n + val_n:]
    return train, val, test, mean, std


def _adjust_step_intervals(config: HFTrainingConfig) -> None:
    """Ensure logging/eval/save intervals remain valid even for short runs."""
    for field_name in ('logging_steps', 'eval_steps', 'save_steps'):
        value = getattr(config, field_name, None)
        if value is None:
            continue
        if config.max_steps <= 0:
            setattr(config, field_name, 1)
            continue
        if value > config.max_steps:
            setattr(config, field_name, max(1, config.max_steps))


def train_base_model(
    base_output_dir: Path,
    device_str: str = 'cuda',
    *,
    max_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    sequence_length: int = 60,
    prediction_horizon: int = 5,
    max_files: Optional[int] = None,
) -> Tuple[str, HFTrainingConfig, np.ndarray, np.ndarray, List[str], str]:
    """Train on all trainingdata/train and return (final_ckpt_path, config, mean, std, symbols, device)."""
    data_root = find_trainingdata_root()
    data_all, symbols = load_all_training_data(data_root, max_files=max_files)
    seq_len = sequence_length
    pred_h = prediction_horizon
    train, val, _test, mean, std = split_and_normalize(data_all, seq_len, pred_h)

    config = HFTrainingConfig(
        hidden_size=256 if device_str == 'cuda' else 128,
        num_layers=6 if device_str == 'cuda' else 3,
        num_heads=8 if device_str == 'cuda' else 4,
        learning_rate=learning_rate or 3e-4,
        warmup_steps=100,
        max_steps=max_steps or (5000 if device_str == 'cuda' else 1000),
        batch_size=batch_size or (128 if device_str == 'cuda' else 64),
        optimizer_name='lion' if device_str == 'cuda' else 'adamw',
        weight_decay=0.01,
        eval_steps=100,
        save_steps=500,
        logging_steps=20,
        max_grad_norm=1.0,
        use_adaptive_grad_clip=True,
        agc_clip_factor=0.01,
        agc_eps=1e-3,
        skip_non_finite_grads=True,
        gradient_accumulation_steps=1 if device_str == 'cuda' else 2,
        use_mixed_precision=(device_str == 'cuda'),
        use_bfloat16=True,
        use_compile=(device_str == 'cuda'),
        allow_tf32=True,
        dataloader_num_workers=4 if device_str == 'cuda' else 0,
        persistent_workers=True,
        prefetch_factor=2,
        input_noise_std=0.001,
        input_noise_prob=0.5,
        input_noise_clip=0.02,
        early_stopping_patience=15,
        early_stopping_threshold=0.001,
        output_dir=str(base_output_dir),
        logging_dir=str(base_output_dir / 'logs')
    )
    config.sequence_length = seq_len
    config.prediction_horizon = pred_h
    _adjust_step_intervals(config)

    train_ds = StockDataset(train, sequence_length=config.sequence_length, prediction_horizon=config.prediction_horizon)
    val_ds = StockDataset(val, sequence_length=config.sequence_length, prediction_horizon=config.prediction_horizon) if len(val) > seq_len + pred_h else None

    model = TransformerTradingModel(config, input_dim=data_all.shape[1])
    trainer = HFTrainer(model=model, config=config, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    actual_device = getattr(trainer, "device", torch.device(device_str))
    final_ckpt = Path(config.output_dir) / 'final_model.pth'
    stats_path = base_output_dir / 'normalization_stats.npz'
    try:
        np.savez(stats_path, mean=mean, std=std)
    except Exception as exc:
        print(f"Warning: failed to persist normalization stats to {stats_path}: {exc}")
    actual_device_str = actual_device.type if hasattr(actual_device, "type") else str(actual_device)
    return str(final_ckpt), config, mean, std, symbols, actual_device_str


def finetune_symbol(csv_path: Path, base_ckpt: str, base_config: HFTrainingConfig, mean: np.ndarray, std: np.ndarray,
                    output_dir: Path, device_str: str = 'cuda', steps: int = 1500) -> str:
    """Fine-tune a model initialized from base_ckpt on a single symbol CSV. Returns final checkpoint path."""
    data = load_csv_ohlc(csv_path)
    if data is None or len(data) < 100:
        raise RuntimeError(f"Not enough data in {csv_path}")
    norm = (data - mean) / (std + 1e-8)
    n = len(norm)
    train_n = int(0.9 * n)
    train = norm[:train_n]
    val = norm[train_n:]

    # Shallow copy base config and adjust
    cfg = HFTrainingConfig(**vars(base_config))
    cfg.max_steps = steps
    cfg.learning_rate = max(1e-5, base_config.learning_rate / 10)
    cfg.output_dir = str(output_dir)
    cfg.logging_dir = str(output_dir / 'logs')
    _adjust_step_intervals(cfg)

    train_ds = StockDataset(train, sequence_length=cfg.sequence_length, prediction_horizon=cfg.prediction_horizon)
    val_ds = StockDataset(val, sequence_length=cfg.sequence_length, prediction_horizon=cfg.prediction_horizon) if len(val) > cfg.sequence_length + cfg.prediction_horizon else None

    model = TransformerTradingModel(cfg, input_dim=norm.shape[1])
    # Load base weights
    try:
        state = torch.load(base_ckpt, map_location='cpu', weights_only=False)
        state_dict = state.get('model_state_dict', state)
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        missing = model.load_state_dict(state_dict, strict=False)
        if missing.missing_keys:
            print(f"Warning: missing keys when loading base checkpoint for {csv_path.stem}: {missing.missing_keys}")
    except Exception as e:
        print(f"Warning: failed to load base checkpoint {base_ckpt}: {e}")

    trainer = HFTrainer(model=model, config=cfg, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    return str(Path(cfg.output_dir) / 'final_model.pth')


def compute_pnl_for_dataset(model: TransformerTradingModel, dataset: StockDataset, device: torch.device,
                            fee_bps: float = 10.0, starting_capital: float = 10000.0) -> Dict[str, float]:
    """Compute simple PnL over dataset using 1-step decisions.

    - Position mapping: buy=+1, hold=0, sell=-1
    - Return at step t is (close_{t+1}-close_t)/close_t
    - Transaction cost applied when position changes (fee_bps / 10,000)
    - Uses batch_size=1 sequentially for a consistent time series
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    model.to(device)
    equity = starting_capital
    peak = starting_capital
    prev_pos = 0.0
    fee = float(fee_bps) / 10000.0
    rets: List[float] = []
    wins = 0
    trades = 0
    with torch.no_grad():
        for sample in loader:
            seq = sample['input_ids'].to(device)
            attn = sample['attention_mask'].to(device)
            outs = model(seq, attention_mask=attn)
            action = int(torch.argmax(outs['action_logits'][0]).item())
            pos = {0: 1.0, 1: 0.0, 2: -1.0}[action]
            current_close = float(sample['input_ids'][0, -1, 3].item())
            next_close = float(sample['labels'][0, 0, 3].item())
            ret = (next_close - current_close) / (current_close + 1e-8)
            # Trading cost on position change
            cost = fee if pos != prev_pos else 0.0
            step_ret = pos * ret - cost
            rets.append(step_ret)
            equity *= (1.0 + step_ret)
            peak = max(peak, equity)
            if step_ret > 0:
                wins += 1
            if pos != prev_pos:
                trades += 1
            prev_pos = pos
    total_ret = (equity / starting_capital) - 1.0
    arr = np.array(rets, dtype=np.float64)
    ann_factor = np.sqrt(252.0)  # if roughly daily
    sharpe = float((arr.mean() / (arr.std() + 1e-12)) * ann_factor) if arr.size > 1 else 0.0
    mdd = 0.0
    # Simple max drawdown on equity curve
    eq = starting_capital
    peak_eq = eq
    for r in rets:
        eq *= (1.0 + r)
        peak_eq = max(peak_eq, eq)
        mdd = min(mdd, (eq / peak_eq) - 1.0)
    return {
        'final_equity': float(equity),
        'total_return': float(total_ret),
        'sharpe': sharpe,
        'max_drawdown': float(mdd),
        'num_trades': int(trades),
        'win_rate': float(wins / max(1, len(rets)))
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train base model and per-symbol fine-tunes.")
    parser.add_argument('--device', choices=('cuda', 'cpu'), default=None, help="Force training device.")
    parser.add_argument('--max-base-steps', type=int, default=None, help="Override base training max_steps.")
    parser.add_argument('--base-batch-size', type=int, default=None, help="Override base training batch size.")
    parser.add_argument('--base-learning-rate', type=float, default=None, help="Override base learning rate.")
    parser.add_argument('--base-sequence-length', type=int, default=60, help="Sequence length for base training.")
    parser.add_argument('--base-prediction-horizon', type=int, default=5, help="Prediction horizon for base training.")
    parser.add_argument('--base-max-files', type=int, default=None, help="Limit number of CSV files for base training.")
    parser.add_argument('--finetune-steps', type=int, default=1500, help="Steps for each symbol fine-tune.")
    parser.add_argument('--finetune-symbol-limit', type=int, default=None, help="Limit number of symbols to fine-tune.")
    parser.add_argument('--symbols', nargs='+', default=None, help="Only fine-tune specified symbol tickers.")
    parser.add_argument('--skip-eval', action='store_true', help="Skip post-finetune evaluation metrics.")
    parser.add_argument('--output-root', type=Path, default=None, help="Root directory for outputs.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args(argv)


def maybe_set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    maybe_set_seed(args.seed)

    auto_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_str = args.device or auto_device
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("Requested CUDA but no GPU detected, falling back to CPU.")
        device_str = 'cpu'

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = args.output_root or (this_dir / 'output')
    base_out = output_root / f'base_{ts}'
    base_out.mkdir(parents=True, exist_ok=True)

    print(f"Training base model on aggregated trainingdata → {base_out}")
    base_ckpt, base_cfg, mean, std, base_symbols, actual_device = train_base_model(
        base_out,
        device_str=device_str,
        max_steps=args.max_base_steps,
        batch_size=args.base_batch_size,
        learning_rate=args.base_learning_rate,
        sequence_length=args.base_sequence_length,
        prediction_horizon=args.base_prediction_horizon,
        max_files=args.base_max_files,
    )
    print(f"Base checkpoint: {base_ckpt}")

    stats_meta = {
        "timestamp": ts,
        "base_checkpoint": base_ckpt,
        "symbols": base_symbols,
        "sequence_length": base_cfg.sequence_length,
        "prediction_horizon": base_cfg.prediction_horizon,
        "device": actual_device,
        "max_steps": base_cfg.max_steps,
    }
    try:
        with open(base_out / 'run_metadata.json', 'w') as fh:
            json.dump(stats_meta, fh, indent=2)
    except Exception as exc:
        print(f"Warning: unable to write metadata file: {exc}")

    # Fine-tune per symbol and evaluate PnL on held-out slice
    data_root = find_trainingdata_root()
    symbols_dir = data_root / 'train'
    csvs = sorted(symbols_dir.glob('*.csv'))
    if args.symbols:
        target_set = {sym.upper() for sym in args.symbols}
        csvs = [csv for csv in csvs if csv.stem.upper() in target_set]
    if args.finetune_symbol_limit is not None:
        csvs = csvs[:args.finetune_symbol_limit]
    print(f"Found {len(csvs)} symbols to fine-tune")

    finetune_root = output_root / f'finetune_{ts}'
    finetune_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, str] = {}
    metrics_rows: List[Dict[str, object]] = []
    for csv in csvs:
        sym = csv.stem
        out_dir = finetune_root / sym
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"→ Fine-tuning {sym} → {out_dir}")
        try:
            # Fine-tune
            ckpt = finetune_symbol(csv, base_ckpt, base_cfg, mean, std, out_dir, device_str=device_str, steps=args.finetune_steps)
            results[sym] = ckpt
            if args.skip_eval:
                continue
            # Evaluate on held-out 10% for this symbol
            data = load_csv_ohlc(csv)
            if data is None:
                raise RuntimeError("failed to load back for eval")
            norm = (data - mean) / (std + 1e-8)
            n = len(norm)
            train_n = int(0.9 * n)
            test = norm[train_n:]
            if len(test) <= base_cfg.sequence_length + base_cfg.prediction_horizon:
                raise RuntimeError("not enough test data after split")
            test_ds = StockDataset(test, sequence_length=base_cfg.sequence_length, prediction_horizon=base_cfg.prediction_horizon)
            # Load model for eval
            cfg = HFTrainingConfig(**vars(base_cfg))
            cfg.output_dir = str(out_dir)
            cfg.logging_dir = str(out_dir / 'logs')
            model = TransformerTradingModel(cfg, input_dim=norm.shape[1])
            try:
                state = torch.load(ckpt, map_location='cpu', weights_only=False)
                state_dict = state.get('model_state_dict', state)
                if all(k.startswith('module.') for k in state_dict.keys()):
                    state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"  Warning: failed to load fine-tuned weights for {sym}: {e}")
            metrics = compute_pnl_for_dataset(model, test_ds, device=torch.device(actual_device))
            metrics_rows.append({'symbol': sym, **metrics})
            print(f"  {sym} PnL — total_return: {metrics['total_return']*100:.2f}% final_equity: ${metrics['final_equity']:.2f} sharpe: {metrics['sharpe']:.2f} mdd: {metrics['max_drawdown']*100:.2f}% trades: {metrics['num_trades']} win_rate: {metrics['win_rate']*100:.1f}%")
        except Exception as e:
            print(f"  Skipped {sym}: {e}")

    print("Done. Fine-tuned checkpoints:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    if results:
        try:
            with open(finetune_root / 'checkpoint_index.json', 'w') as fh:
                json.dump(results, fh, indent=2)
        except Exception as exc:
            print(f"Warning: unable to write checkpoint index: {exc}")

    # Save metrics summary
    if metrics_rows:
        import csv as _csv
        summary_path = finetune_root / 'metrics_summary.csv'
        with open(summary_path, 'w', newline='') as f:
            writer = _csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)
        # Print aggregated metric
        avg_total_return = float(np.mean([r['total_return'] for r in metrics_rows]))
        print(f"Average total return across symbols (held-out): {avg_total_return*100:.2f}%")
        print(f"Metrics summary saved to {summary_path}")


if __name__ == '__main__':
    main()
