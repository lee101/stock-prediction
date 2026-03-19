#!/usr/bin/env python3
"""
Compare three training approaches on the hourly crypto market simulator:

1. RL (PPO via Stable-Baselines3 / GymRL portfolio env)
2. HuggingFace Trainer (TransformerTradingModel with custom loss)
3. Custom Qwen fine-tuning (LoRA on Qwen3.5-0.8B for trading plan generation)

Each approach trains on the same data window, then generates buy/sell actions
which are evaluated by the shared-cash BinanceMarketSimulator. Metrics:
total return, Sortino ratio, max drawdown, PnL smoothness, goodness score.

Usage:
    source .venv313/bin/activate
    python experiments/compare_rl_hf_qwen.py --symbols BTCUSD,ETHUSD,SOLUSD
    python experiments/compare_rl_hf_qwen.py --symbols BTCUSD,ETHUSD --quick
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from src.robust_trading_metrics import (
    compute_market_sim_goodness_score,
    compute_max_drawdown,
    compute_pnl_smoothness,
    compute_return_series,
    compute_ulcer_index,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("compare")

# ── Data loading ──────────────────────────────────────────────────────────

HOURLY_DATA_DIR = REPO / "trainingdatahourly" / "crypto"
DAILY_DATA_DIR = REPO / "tradingdata" / "train"

DEFAULT_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]


def load_hourly_bars(symbol: str, data_dir: Path = HOURLY_DATA_DIR) -> pd.DataFrame:
    """Load hourly OHLCV bars for a single symbol."""
    csv_path = data_dir / f"{symbol}.csv"
    if not csv_path.exists():
        # Try daily data as fallback
        csv_path = DAILY_DATA_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No data for {symbol} in {data_dir} or {DAILY_DATA_DIR}")

    df = pd.read_csv(csv_path, parse_dates=True)
    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("date", "datetime", "timestamp", "time"):
            col_map[c] = "timestamp"
        elif cl == "open":
            col_map[c] = "open"
        elif cl == "high":
            col_map[c] = "high"
        elif cl == "low":
            col_map[c] = "low"
        elif cl == "close":
            col_map[c] = "close"
        elif cl == "volume":
            col_map[c] = "volume"
    df = df.rename(columns=col_map)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {csv_path}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df


def make_train_test_split(
    df: pd.DataFrame, train_frac: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split."""
    n = len(df)
    split = int(n * train_frac)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


# ── Metric computation helper ────────────────────────────────────────────


def compute_metrics(equity_curve: np.ndarray) -> Dict[str, float]:
    """Compute standard trading metrics from an equity curve."""
    returns = compute_return_series(equity_curve)
    total_return = float(equity_curve[-1] / equity_curve[0] - 1) if equity_curve[0] > 0 else 0.0
    max_dd = compute_max_drawdown(equity_curve)
    smoothness = compute_pnl_smoothness(returns)
    ulcer = compute_ulcer_index(equity_curve)

    downside = returns[returns < 0]
    down_std = float(np.std(downside)) if len(downside) > 0 else 0.0
    mean_ret = float(np.mean(returns))
    sortino = (mean_ret / down_std * np.sqrt(8760)) if down_std > 0 else 0.0

    goodness = compute_market_sim_goodness_score(
        total_return=total_return,
        sortino=sortino,
        max_drawdown=max_dd,
        pnl_smoothness=smoothness,
        ulcer_index=ulcer,
        trade_rate=0.5,
        period_count=len(returns),
    )

    return {
        "total_return": total_return,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "pnl_smoothness": smoothness,
        "ulcer_index": ulcer,
        "goodness_score": goodness,
        "n_bars": len(equity_curve),
    }


# ── Approach 1: RL (PPO with GymRL) ──────────────────────────────────────


def run_rl_experiment(
    train_data: Dict[str, pd.DataFrame],
    test_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    output_dir: Path,
    quick: bool = False,
) -> Dict[str, Any]:
    """Train PPO on the portfolio env, evaluate on test data."""
    logger.info("=== RL (PPO / GymRL) ===")
    start_t = time.time()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from gymrl import PortfolioEnv, PortfolioEnvConfig

    # Build feature arrays: simple normalized returns as features
    all_returns = []
    all_prices = []
    symbol_list = sorted(symbols)
    min_len = min(len(train_data[s]) for s in symbol_list)

    for s in symbol_list:
        df = train_data[s].iloc[:min_len]
        closes = df["close"].values
        rets = np.diff(closes) / (closes[:-1] + 1e-8)
        all_returns.append(rets)
        all_prices.append(closes[1:])

    returns_matrix = np.column_stack(all_returns)  # (T, N_assets)
    prices_matrix = np.column_stack(all_prices)

    # Build a simple feature cube: [T, N_assets, N_features]
    # Features: return, abs_return, rolling_mean_5, rolling_std_5
    T, N = returns_matrix.shape
    n_features = 4
    features = np.zeros((T, N, n_features), dtype=np.float32)
    features[:, :, 0] = returns_matrix
    features[:, :, 1] = np.abs(returns_matrix)
    for i in range(N):
        r = returns_matrix[:, i]
        for t in range(T):
            window = r[max(0, t - 5) : t + 1]
            features[t, i, 2] = np.mean(window)
            features[t, i, 3] = np.std(window) if len(window) > 1 else 0.0

    env_config = PortfolioEnvConfig(
        costs_bps=5.0,
        turnover_penalty=5e-4,
        include_cash=True,
    )

    # realized_returns must be (T, N) or (T, N, 1)
    realized_returns = returns_matrix.astype(np.float32)

    def make_env():
        env = PortfolioEnv(
            features=features,
            realized_returns=realized_returns,
            config=env_config,
        )
        return env

    vec_env = DummyVecEnv([make_env])

    timesteps = 100_000 if quick else 500_000
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=min(2048, T - 1),
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        device="cpu",  # SB3 PPO MlpPolicy runs better on CPU
    )

    logger.info(f"Training PPO for {timesteps} timesteps on {N} assets...")
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - start_t
    logger.info(f"RL training done in {train_time:.1f}s")

    # Evaluate on test data
    test_min_len = min(len(test_data[s]) for s in symbol_list)
    test_returns = []
    test_prices = []
    for s in symbol_list:
        df = test_data[s].iloc[:test_min_len]
        closes = df["close"].values
        rets = np.diff(closes) / (closes[:-1] + 1e-8)
        test_returns.append(rets)
        test_prices.append(closes[1:])

    test_returns_matrix = np.column_stack(test_returns)
    test_prices_matrix = np.column_stack(test_prices)
    T_test = len(test_returns_matrix)

    test_features = np.zeros((T_test, N, n_features), dtype=np.float32)
    test_features[:, :, 0] = test_returns_matrix
    test_features[:, :, 1] = np.abs(test_returns_matrix)
    for i in range(N):
        r = test_returns_matrix[:, i]
        for t in range(T_test):
            window = r[max(0, t - 5) : t + 1]
            test_features[t, i, 2] = np.mean(window)
            test_features[t, i, 3] = np.std(window) if len(window) > 1 else 0.0

    # Simulate portfolio using trained policy
    # Use the env itself to get correct observation shape
    test_env_config = PortfolioEnvConfig(
        costs_bps=5.0,
        turnover_penalty=5e-4,
        include_cash=True,
    )
    test_env = PortfolioEnv(
        features=test_features,
        realized_returns=test_returns_matrix.astype(np.float32),
        config=test_env_config,
    )

    initial_cash = 10_000.0
    obs, _ = test_env.reset()
    # Env portfolio_value starts at 1.0 (multiplicative)
    equity_curve = [initial_cash]

    for t in range(1, T_test):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        # Scale env's multiplicative portfolio value to cash terms
        pv = info.get("portfolio_value", 1.0) * initial_cash
        equity_curve.append(pv)

        if terminated or truncated:
            break

    equity_arr = np.array(equity_curve)
    metrics = compute_metrics(equity_arr)
    metrics["train_time_s"] = train_time
    metrics["approach"] = "RL (PPO/GymRL)"

    # Save
    rl_dir = output_dir / "rl"
    rl_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(equity_arr).to_csv(rl_dir / "equity_curve.csv", index=False)
    with open(rl_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    model.save(str(rl_dir / "ppo_model"))

    logger.info(f"RL results: return={metrics['total_return']:.4f} sortino={metrics['sortino']:.2f} "
                f"max_dd={metrics['max_drawdown']:.4f} goodness={metrics['goodness_score']:.2f}")
    return metrics


# ── Approach 2: HuggingFace Trainer ───────────────────────────────────────


def run_hf_experiment(
    train_data: Dict[str, pd.DataFrame],
    test_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    output_dir: Path,
    quick: bool = False,
) -> Dict[str, Any]:
    """Train a TransformerTradingModel with custom HF-style trainer, evaluate."""
    logger.info("=== HuggingFace Trainer (TransformerTradingModel) ===")
    start_t = time.time()

    # Free GPU memory from previous experiments
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc; gc.collect()

    sys.path.insert(0, str(REPO / "hftraining"))
    from hftraining.hf_trainer import HFTrainingConfig, TransformerTradingModel
    from hftraining.data_utils import StockDataProcessor

    # Prepare multi-symbol training data
    all_train_arrays = []
    all_test_arrays = []
    symbol_list = sorted(symbols)

    for s in symbol_list:
        df = train_data[s]
        arr = df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
        all_train_arrays.append(arr)

        df_test = test_data[s]
        arr_test = df_test[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
        all_test_arrays.append(arr_test)

    # Use first symbol's data for single-symbol training (simplest comparison)
    train_arr = all_train_arrays[0]
    test_arr = all_test_arrays[0]

    # Normalize
    processor = StockDataProcessor(sequence_length=60, prediction_horizon=5)
    processor.fit_scalers(train_arr)
    norm_train = processor.transform(train_arr)
    norm_test = processor.transform(test_arr)

    # Create datasets
    from hftraining.train_hf import StockDataset
    train_dataset = StockDataset(norm_train, sequence_length=60, prediction_horizon=5, processor=processor, symbol=symbol_list[0])
    test_dataset = StockDataset(norm_test, sequence_length=60, prediction_horizon=5, processor=processor, symbol=symbol_list[0])

    # Configure model
    n_features = train_arr.shape[1]
    config = HFTrainingConfig(
        hidden_size=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        learning_rate=1e-3,
        batch_size=64,
        max_steps=2000 if quick else 10000,
        early_stopping_patience=5,
        use_mixed_precision=True,
        precision="bf16",
        prediction_horizon=5,
        use_wandb=False,
        use_data_parallel=False,
        optimizer_name="adamw",
    )

    model = TransformerTradingModel(config, input_dim=n_features)

    from hftraining.train_hf import HFTrainer
    trainer = HFTrainer(model, config, train_dataset, eval_dataset=test_dataset)

    logger.info(f"Training HF model for {config.max_steps} steps on {symbol_list[0]}...")
    trainer.train()
    train_time = time.time() - start_t
    logger.info(f"HF training done in {train_time:.1f}s")

    # Generate predictions on test set and simulate
    model_ref = trainer.model
    if hasattr(model_ref, "module"):
        model_ref = model_ref.module
    model_ref.eval()
    device = trainer.device

    cash = 10_000.0
    equity_curve = [cash]
    position = 0.0  # shares held
    fee_rate = 0.001
    close_idx = 3  # OHLCV ordering

    test_closes = test_arr[:, close_idx]

    with torch.no_grad():
        for i in range(60, len(norm_test) - 5):
            seq = torch.from_numpy(norm_test[i - 60 : i]).float().unsqueeze(0).to(device)
            mask = torch.ones(1, 60).to(device)

            out = model_ref(seq, mask)
            # Use action_logits (buy=0, hold=1, sell=2) and price predictions
            action_logits = out["action_logits"].cpu().numpy().squeeze()
            action = int(np.argmax(action_logits))
            # Also check allocations signal for directional confidence
            alloc = out["allocations"].cpu().item()

            current_price = test_closes[i]
            portfolio_value = cash + position * current_price

            # Use allocation signal as tiebreaker when action is hold
            if action == 1 and abs(alloc) > 0.3:
                action = 0 if alloc > 0 else 2

            if action == 0 and position <= 0:  # BUY
                # Buy
                buy_value = portfolio_value * 0.95
                shares = buy_value / current_price
                cost = buy_value * fee_rate
                position += shares
                cash -= buy_value + cost
            elif action == 2 and position > 0:  # SELL
                # Sell
                sell_value = position * current_price
                cost = sell_value * fee_rate
                cash += sell_value - cost
                position = 0.0

            next_price = test_closes[min(i + 1, len(test_closes) - 1)]
            portfolio_value = cash + position * next_price
            equity_curve.append(portfolio_value)

    equity_arr = np.array(equity_curve)
    metrics = compute_metrics(equity_arr)
    metrics["train_time_s"] = train_time
    metrics["approach"] = "HuggingFace Trainer"

    hf_dir = output_dir / "hf_trainer"
    hf_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(equity_arr).to_csv(hf_dir / "equity_curve.csv", index=False)
    with open(hf_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"HF results: return={metrics['total_return']:.4f} sortino={metrics['sortino']:.2f} "
                f"max_dd={metrics['max_drawdown']:.4f} goodness={metrics['goodness_score']:.2f}")

    # Aggressively free GPU memory for next experiment
    del model_ref, trainer, model, train_dataset, test_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc; gc.collect()
        torch.cuda.empty_cache()

    return metrics


# ── Approach 3: Custom Qwen Fine-Tuning ───────────────────────────────────


def _generate_qwen_training_data(
    train_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    output_path: Path,
    max_examples: int = 2000,
):
    """Generate chat-format JSONL training data for Qwen fine-tuning."""
    examples = []
    for symbol in symbols:
        df = train_data[symbol]
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        volumes = df["volume"].values

        for i in range(24, len(df) - 6):
            if len(examples) >= max_examples:
                break

            # Context window
            recent_closes = closes[i - 24 : i]
            recent_highs = highs[i - 24 : i]
            recent_lows = lows[i - 24 : i]
            recent_vols = volumes[i - 24 : i]

            current_price = closes[i]
            # Compute features
            ret_1h = (closes[i] - closes[i - 1]) / (closes[i - 1] + 1e-8)
            ret_4h = (closes[i] - closes[i - 4]) / (closes[i - 4] + 1e-8)
            ret_24h = (closes[i] - closes[i - 24]) / (closes[i - 24] + 1e-8)
            volatility = float(np.std(np.diff(recent_closes) / (recent_closes[:-1] + 1e-8)))
            avg_volume = float(np.mean(recent_vols))

            # Future return (label)
            future_price = closes[min(i + 4, len(closes) - 1)]
            future_return = (future_price - current_price) / (current_price + 1e-8)

            # Determine optimal action from hindsight
            if future_return > 0.005:
                action = "BUY"
                confidence = min(future_return / 0.02, 1.0)
            elif future_return < -0.005:
                action = "SELL"
                confidence = min(abs(future_return) / 0.02, 1.0)
            else:
                action = "HOLD"
                confidence = 0.5

            user_msg = (
                f"Symbol: {symbol}\n"
                f"Current price: {current_price:.2f}\n"
                f"1h return: {ret_1h:.4f}\n"
                f"4h return: {ret_4h:.4f}\n"
                f"24h return: {ret_24h:.4f}\n"
                f"Volatility (24h): {volatility:.6f}\n"
                f"Avg volume (24h): {avg_volume:.0f}\n"
                f"Recent closes: {','.join(f'{c:.2f}' for c in recent_closes[-8:])}\n"
                f"Generate a trading plan."
            )

            assistant_msg = (
                f"Action: {action}\n"
                f"Confidence: {confidence:.2f}\n"
                f"Entry: {current_price:.2f}\n"
                f"Stop: {current_price * (0.98 if action == 'BUY' else 1.02):.2f}\n"
                f"Target: {current_price * (1 + future_return):.2f}\n"
                f"Reasoning: Based on {ret_24h:.1%} 24h momentum and {volatility:.4f} volatility."
            )

            examples.append({
                "messages": [
                    {"role": "system", "content": "You are a crypto trading assistant. Analyze market data and provide actionable trading plans."},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    logger.info(f"Generated {len(examples)} Qwen training examples -> {output_path}")
    return len(examples)


def run_qwen_experiment(
    train_data: Dict[str, pd.DataFrame],
    test_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    output_dir: Path,
    quick: bool = False,
) -> Dict[str, Any]:
    """Fine-tune Qwen3.5-0.8B with LoRA, evaluate on test data."""
    logger.info("=== Custom Qwen Fine-Tuning (LoRA) ===")
    start_t = time.time()

    # Free GPU memory from previous experiments
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc; gc.collect()

    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from torch.utils.data import DataLoader

    qwen_dir = output_dir / "qwen"
    qwen_dir.mkdir(parents=True, exist_ok=True)

    # Generate training data
    train_jsonl = qwen_dir / "train.jsonl"
    val_jsonl = qwen_dir / "val.jsonl"
    max_train = 500 if quick else 2000
    _generate_qwen_training_data(train_data, symbols, train_jsonl, max_examples=max_train)
    # Use small portion of test data for validation
    val_symbols_data = {s: test_data[s].iloc[: len(test_data[s]) // 3] for s in symbols}
    _generate_qwen_training_data(val_symbols_data, symbols, val_jsonl, max_examples=max_train // 4)

    # Load model
    model_id = "Qwen/Qwen3.5-0.8B"
    logger.info(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Qwen trainable: {trainable / 1e6:.1f}M / {total_params / 1e6:.1f}M")

    # Load dataset (reuse finetune_qwen.py's ChatJSONLDataset)
    sys.path.insert(0, str(REPO / "rl-trainingbinance"))
    from finetune_qwen import ChatJSONLDataset

    train_ds = ChatJSONLDataset(train_jsonl, tokenizer, max_len=192)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    epochs = 1 if quick else 3
    accum_steps = 8
    total_steps = (len(train_loader) // accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(total_steps * 0.05)), num_training_steps=max(1, total_steps)
    )

    device = next(model.parameters()).device

    logger.info(f"Training Qwen LoRA for {epochs} epochs, {len(train_ds)} examples...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / accum_steps
            loss.backward()
            total_loss += out.loss.item()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 50 == 0:
                avg = total_loss / (step + 1)
                logger.info(f"  Qwen ep{epoch} step{step + 1}/{len(train_loader)} loss={avg:.4f}")

        avg_loss = total_loss / max(len(train_loader), 1)
        logger.info(f"Qwen ep{epoch} train_loss={avg_loss:.4f}")

    train_time = time.time() - start_t
    logger.info(f"Qwen training done in {train_time:.1f}s")

    # Save model
    model.save_pretrained(qwen_dir / "final")
    tokenizer.save_pretrained(qwen_dir / "final")

    # Evaluate: generate trading plans on test data and simulate
    model.eval()
    cash = 10_000.0
    equity_curve = [cash]
    position = 0.0
    fee_rate = 0.001

    symbol = symbols[0]
    test_df = test_data[symbol]
    test_closes = test_df["close"].values
    test_highs = test_df["high"].values
    test_vols = test_df["volume"].values

    # In quick mode, limit to ~200 predictions; otherwise ~500
    step_size = max(4, len(test_df) // (200 if quick else 500))
    logger.info(f"Qwen inference: {len(test_df)} bars, step={step_size}, ~{len(test_df)//step_size} predictions")
    with torch.no_grad():
        for i in range(24, len(test_df) - 1, step_size):
            # Build prompt
            recent_closes = test_closes[i - 24 : i]
            current_price = test_closes[i]
            ret_1h = (test_closes[i] - test_closes[i - 1]) / (test_closes[i - 1] + 1e-8)
            ret_4h = (test_closes[i] - test_closes[i - 4]) / (test_closes[i - 4] + 1e-8)
            ret_24h = (test_closes[i] - test_closes[i - 24]) / (test_closes[i - 24] + 1e-8)
            volatility = float(np.std(np.diff(recent_closes) / (recent_closes[:-1] + 1e-8)))
            avg_volume = float(np.mean(test_vols[i - 24 : i]))

            user_msg = (
                f"Symbol: {symbol}\n"
                f"Current price: {current_price:.2f}\n"
                f"1h return: {ret_1h:.4f}\n"
                f"4h return: {ret_4h:.4f}\n"
                f"24h return: {ret_24h:.4f}\n"
                f"Volatility (24h): {volatility:.6f}\n"
                f"Avg volume (24h): {avg_volume:.0f}\n"
                f"Recent closes: {','.join(f'{c:.2f}' for c in recent_closes[-8:])}\n"
                f"Generate a trading plan."
            )

            messages = [
                {"role": "system", "content": "You are a crypto trading assistant. Analyze market data and provide actionable trading plans."},
                {"role": "user", "content": user_msg},
            ]

            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)

            gen = model.generate(
                **enc,
                max_new_tokens=40,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
            response = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

            # Parse action from response
            action = "HOLD"
            if "Action: BUY" in response or "action: BUY" in response.upper()[:50]:
                action = "BUY"
            elif "Action: SELL" in response or "action: SELL" in response.upper()[:50]:
                action = "SELL"

            portfolio_value = cash + position * current_price

            if action == "BUY" and position <= 0:
                buy_value = portfolio_value * 0.95
                shares = buy_value / current_price
                cost = buy_value * fee_rate
                position += shares
                cash -= buy_value + cost
            elif action == "SELL" and position > 0:
                sell_value = position * current_price
                cost = sell_value * fee_rate
                cash += sell_value - cost
                position = 0.0

            # Fill equity for skipped bars
            for j in range(step_size):
                idx = min(i + j + 1, len(test_closes) - 1)
                pv = cash + position * test_closes[idx]
                equity_curve.append(pv)

    equity_arr = np.array(equity_curve)
    metrics = compute_metrics(equity_arr)
    metrics["train_time_s"] = train_time
    metrics["approach"] = "Custom Qwen Fine-Tuning (LoRA)"

    pd.Series(equity_arr).to_csv(qwen_dir / "equity_curve.csv", index=False)
    with open(qwen_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Qwen results: return={metrics['total_return']:.4f} sortino={metrics['sortino']:.2f} "
                f"max_dd={metrics['max_drawdown']:.4f} goodness={metrics['goodness_score']:.2f}")
    return metrics


# ── Main comparison ───────────────────────────────────────────────────────


def print_comparison_table(results: List[Dict[str, Any]]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("COMPARISON: RL vs HF Trainer vs Custom Qwen Fine-Tuning")
    print("=" * 90)
    header = f"{'Approach':<35} {'Return':>10} {'Sortino':>10} {'MaxDD':>10} {'Goodness':>10} {'Time(s)':>10}"
    print(header)
    print("-" * 90)
    for r in results:
        print(
            f"{r['approach']:<35} "
            f"{r['total_return']:>9.4f} "
            f"{r['sortino']:>9.2f} "
            f"{r['max_drawdown']:>9.4f} "
            f"{r['goodness_score']:>9.2f} "
            f"{r.get('train_time_s', 0):>9.1f}"
        )
    print("=" * 90)

    # Determine winner
    best = max(results, key=lambda x: x["goodness_score"])
    print(f"\nWinner by goodness score: {best['approach']} ({best['goodness_score']:.2f})")
    print()


def main():
    ap = argparse.ArgumentParser(description="Compare RL vs HF Trainer vs Custom Qwen")
    ap.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--output-dir", type=Path, default=REPO / "experiments" / "comparison_results")
    ap.add_argument("--quick", action="store_true", help="Use smaller configs for faster iteration")
    ap.add_argument("--skip-rl", action="store_true", help="Skip RL experiment")
    ap.add_argument("--skip-hf", action="store_true", help="Skip HF experiment")
    ap.add_argument("--skip-qwen", action="store_true", help="Skip Qwen experiment")
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data for {symbols}...")
    train_data = {}
    test_data = {}
    for s in symbols:
        try:
            df = load_hourly_bars(s)
            train_df, test_df = make_train_test_split(df, args.train_frac)
            train_data[s] = train_df
            test_data[s] = test_df
            logger.info(f"  {s}: {len(train_df)} train, {len(test_df)} test bars")
        except FileNotFoundError as e:
            logger.warning(f"  Skipping {s}: {e}")

    symbols = list(train_data.keys())
    if not symbols:
        logger.error("No data found for any symbol!")
        sys.exit(1)

    results = []

    if not args.skip_rl:
        try:
            rl_metrics = run_rl_experiment(train_data, test_data, symbols, output_dir, quick=args.quick)
            results.append(rl_metrics)
        except Exception as e:
            logger.error(f"RL experiment failed: {e}", exc_info=True)

    if not args.skip_hf:
        try:
            hf_metrics = run_hf_experiment(train_data, test_data, symbols, output_dir, quick=args.quick)
            results.append(hf_metrics)
        except Exception as e:
            logger.error(f"HF experiment failed: {e}", exc_info=True)

    if not args.skip_qwen:
        try:
            qwen_metrics = run_qwen_experiment(train_data, test_data, symbols, output_dir, quick=args.quick)
            results.append(qwen_metrics)
        except Exception as e:
            logger.error(f"Qwen experiment failed: {e}", exc_info=True)

    if results:
        print_comparison_table(results)

        # Save combined results
        summary = {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "results": results,
        }
        with open(output_dir / "comparison_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to {output_dir / 'comparison_summary.json'}")
    else:
        logger.error("No experiments completed successfully!")


if __name__ == "__main__":
    main()
