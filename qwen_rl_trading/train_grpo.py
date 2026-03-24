"""GRPO training of Qwen for structured crypto trading plans.

Usage:
    source .venv312/bin/activate
    python -m qwen_rl_trading.train_grpo --model-size 0.6B --time-budget 600
    python -m qwen_rl_trading.train_grpo --model-size 1.8B --sft-warmstart --n-symbols 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

log = logging.getLogger("qwen_grpo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

QWEN_MODELS = {
    "0.6B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.8B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
}


@dataclass
class QwenGRPOConfig:
    model_size: str = "0.6B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    group_size: int = 8
    lr: float = 5e-6
    kl_coef: float = 0.05
    max_completion_length: int = 512
    max_prompt_length: int = 1024
    num_train_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    n_symbols: int = 10
    eval_horizon_hours: int = 24
    lookback_hours: int = 24
    stride: int = 4
    prompt_variant: str = "detailed"
    reward_type: str = "sortino_only"
    eval_every_steps: int = 50
    early_stop_patience: int = 3
    top_k_checkpoints: int = 5
    sft_warmstart: bool = False
    sft_epochs: int = 1
    seed: int = 42
    data_dir: str = "trainingdatahourly/crypto"
    forecast_cache_dir: str = "binanceneural/forecast_cache"
    sft_data: str = "rl-trainingbinance/trading_plans_train.jsonl"
    output_dir: str = "qwen_rl_trading/checkpoints"
    time_budget: int = 0
    description: str = ""


def build_model_and_tokenizer(config: QwenGRPOConfig):
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = QWEN_MODELS[config.model_size]
    log.info("loading %s (%s)", config.model_size, model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if config.model_size == "7B":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    return model, tokenizer, lora_config


def run_sft_warmstart(config: QwenGRPOConfig, model, tokenizer):
    """1-epoch SFT on existing trading plan JSONL to teach JSON format."""
    from peft import get_peft_model, LoraConfig, TaskType
    from torch.utils.data import DataLoader

    sft_path = REPO / config.sft_data
    if not sft_path.exists():
        log.warning("SFT data not found: %s, skipping warmstart", sft_path)
        return model

    log.info("SFT warmstart from %s", sft_path)

    # reuse ChatJSONLDataset pattern from finetune_qwen.py
    examples = []
    with open(sft_path) as f:
        for line in f:
            obj = json.loads(line)
            examples.append(obj["messages"])

    class SFTDataset(torch.utils.data.Dataset):
        def __init__(self, examples, tokenizer, max_len=512):
            self.examples = examples
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            messages = self.examples[idx]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            enc = self.tokenizer(text, max_length=self.max_len, truncation=True,
                                 padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            # mask non-assistant tokens
            assistant_text = messages[-1]["content"]
            assistant_enc = self.tokenizer(assistant_text, add_special_tokens=False)
            assistant_len = len(assistant_enc["input_ids"])
            seq_len = attention_mask.sum().item()
            if assistant_len < seq_len:
                labels[:seq_len - assistant_len] = -100
            labels[attention_mask == 0] = -100
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    dataset = SFTDataset(examples, tokenizer, max_len=config.max_completion_length + config.max_prompt_length)
    loader = DataLoader(dataset, batch_size=config.per_device_batch_size, shuffle=True, drop_last=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr * 10, weight_decay=0.01)

    for epoch in range(config.sft_epochs):
        total_loss = 0
        n = 0
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n += 1
            if n >= 200:  # cap at 200 steps for warmstart
                break
        log.info("SFT epoch %d: loss=%.4f (%d steps)", epoch, total_loss / max(n, 1), n)

    return model


def build_grpo_dataset(config: QwenGRPOConfig):
    """Build prompt dataset and snapshot map for GRPO training."""
    from .data_prompt import PromptDataset, SYMBOLS_30, build_chat_messages

    symbols = SYMBOLS_30[:config.n_symbols]
    data_dir = REPO / config.data_dir
    forecast_dir = REPO / config.forecast_cache_dir

    train_ds = PromptDataset(
        data_dir, symbols,
        forecast_cache_dir=forecast_dir,
        lookback=config.lookback_hours,
        eval_horizon=config.eval_horizon_hours,
        stride=config.stride,
        prompt_variant=config.prompt_variant,
        val_fraction=0.15,
        val_mode=False,
    )
    val_ds = PromptDataset(
        data_dir, symbols,
        forecast_cache_dir=forecast_dir,
        lookback=config.lookback_hours,
        eval_horizon=config.eval_horizon_hours,
        stride=config.stride,
        prompt_variant=config.prompt_variant,
        val_fraction=0.15,
        val_mode=True,
    )

    snapshot_map = {}
    train_prompts = []
    for i in range(len(train_ds)):
        prompt_text, snapshot = train_ds[i]
        snapshot_map[snapshot.window_id] = snapshot
        messages = build_chat_messages(prompt_text)
        train_prompts.append({"prompt": messages})

    val_snapshots = []
    for i in range(len(val_ds)):
        prompt_text, snapshot = val_ds[i]
        snapshot_map[snapshot.window_id] = snapshot
        val_snapshots.append((prompt_text, snapshot))

    log.info("train prompts: %d, val snapshots: %d, symbols: %d",
             len(train_prompts), len(val_snapshots), len(symbols))
    return train_prompts, val_snapshots, snapshot_map


def run_validation(model, tokenizer, val_snapshots, config: QwenGRPOConfig) -> dict:
    """Quick validation: generate plans on val set, compute mean reward."""
    from .reward import compute_reward_detailed
    from .data_prompt import build_chat_messages

    model.eval()
    results = []
    n_eval = min(20, len(val_snapshots))  # limit for speed

    for prompt_text, snapshot in val_snapshots[:n_eval]:
        messages = build_chat_messages(prompt_text)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=config.max_prompt_length).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_completion_length,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        metrics = compute_reward_detailed(completion, snapshot)
        results.append(metrics)

    model.train()
    rewards = [r["reward"] for r in results]
    sortinos = [r.get("sortino", 0) for r in results]
    returns = [r.get("total_return", 0) for r in results]
    valid_pct = sum(1 for r in results if r.get("error") is None) / max(len(results), 1)

    return {
        "val_mean_reward": float(sum(rewards) / max(len(rewards), 1)),
        "val_mean_sortino": float(sum(sortinos) / max(len(sortinos), 1)),
        "val_mean_return": float(sum(returns) / max(len(returns), 1)),
        "val_valid_json_pct": valid_pct,
        "val_n_samples": len(results),
    }


def train(config: QwenGRPOConfig) -> dict:
    """Main GRPO training loop with early stopping."""
    from trl import GRPOConfig, GRPOTrainer
    from .reward import GRPORewardFn

    t0 = time.time()
    deadline = t0 + config.time_budget if config.time_budget > 0 else float("inf")

    model, tokenizer, lora_config = build_model_and_tokenizer(config)

    if config.sft_warmstart:
        from peft import get_peft_model
        model = get_peft_model(model, lora_config)
        model = run_sft_warmstart(config, model, tokenizer)
        # save SFT checkpoint
        sft_dir = Path(config.output_dir) / "sft_warmstart"
        sft_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(sft_dir))
        log.info("SFT warmstart saved to %s", sft_dir)
        # for GRPO, we pass the already-PEFT model
        lora_config = None  # already applied

    train_prompts, val_snapshots, snapshot_map = build_grpo_dataset(config)

    if not train_prompts:
        log.error("no training data found")
        return {"error": "no_data"}

    reward_fn = GRPORewardFn(snapshot_map)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        num_generations=config.group_size,
        max_completion_length=config.max_completion_length,
        max_prompt_length=config.max_prompt_length,
        beta=config.kl_coef,
        seed=config.seed,
        logging_steps=10,
        save_steps=config.eval_every_steps,
        save_total_limit=config.top_k_checkpoints,
        bf16=True,
        report_to="none",
        log_level="warning",
    )

    peft_config_arg = lora_config  # None if SFT warmstart already applied it
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_prompts,
        peft_config=peft_config_arg,
        processing_class=tokenizer,
    )

    # early stopping state
    best_val_reward = -float("inf")
    patience_counter = 0
    val_history = []
    best_checkpoint_path = None

    class EarlyStopCallback:
        def on_step_end(self_cb, args, state, control, **kwargs):
            nonlocal best_val_reward, patience_counter, val_history, best_checkpoint_path
            if time.time() > deadline:
                log.info("time budget exceeded, stopping")
                control.should_training_stop = True
                return

            if state.global_step > 0 and state.global_step % config.eval_every_steps == 0:
                log.info("step %d: running validation", state.global_step)
                val_metrics = run_validation(model, tokenizer, val_snapshots, config)
                val_reward = val_metrics["val_mean_reward"]
                val_history.append(val_reward)
                log.info("step %d: val_reward=%.4f sortino=%.4f valid_json=%.1f%%",
                         state.global_step, val_reward, val_metrics["val_mean_sortino"],
                         val_metrics["val_valid_json_pct"] * 100)

                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    patience_counter = 0
                    ckpt_path = output_dir / f"best_step{state.global_step}"
                    ckpt_path.mkdir(parents=True, exist_ok=True)
                    trainer.save_model(str(ckpt_path))
                    best_checkpoint_path = str(ckpt_path)
                    log.info("new best: %.4f saved to %s", val_reward, ckpt_path)
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stop_patience:
                        log.info("early stopping: %d evals without improvement", patience_counter)
                        control.should_training_stop = True

    from transformers import TrainerCallback

    class _CB(TrainerCallback, EarlyStopCallback):
        pass

    trainer.add_callback(_CB())

    log.info("starting GRPO training: %s", config.description or config.model_size)
    trainer.train()

    # save final
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))

    training_time = time.time() - t0
    final_val = run_validation(model, tokenizer, val_snapshots, config)

    result = {
        "description": config.description or f"qwen_{config.model_size}",
        "model_size": config.model_size,
        "best_checkpoint": best_checkpoint_path or str(final_dir),
        "training_time_s": training_time,
        "val_history": val_history,
        **final_val,
        "config": asdict(config),
    }

    # save result metadata
    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info("training complete: %.0fs, best_reward=%.4f, final_reward=%.4f",
             training_time, best_val_reward, final_val["val_mean_reward"])
    return result


def main():
    p = argparse.ArgumentParser(description="Qwen GRPO trading plan trainer")
    p.add_argument("--model-size", default="0.6B", choices=list(QWEN_MODELS))
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--n-symbols", type=int, default=10)
    p.add_argument("--eval-horizon", type=int, default=24)
    p.add_argument("--reward-type", default="sortino_only")
    p.add_argument("--prompt-variant", default="detailed")
    p.add_argument("--time-budget", type=int, default=0)
    p.add_argument("--sft-warmstart", action="store_true")
    p.add_argument("--data-dir", default="trainingdatahourly/crypto")
    p.add_argument("--output-dir", default="qwen_rl_trading/checkpoints")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--description", default="")
    args = p.parse_args()

    config = QwenGRPOConfig(
        model_size=args.model_size,
        lora_r=args.lora_r,
        group_size=args.group_size,
        lr=args.lr,
        kl_coef=args.kl_coef,
        max_completion_length=args.max_completion_length,
        n_symbols=args.n_symbols,
        eval_horizon_hours=args.eval_horizon,
        reward_type=args.reward_type,
        prompt_variant=args.prompt_variant,
        time_budget=args.time_budget,
        sft_warmstart=args.sft_warmstart,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        eval_every_steps=args.eval_every,
        early_stop_patience=args.early_stop_patience,
        description=args.description,
    )

    result = train(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
