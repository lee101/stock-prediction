"""Evaluate Qwen trading plans on holdout windows."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def evaluate_on_holdout(
    model,
    tokenizer,
    val_snapshots: list,
    n_windows: int = 50,
    temperature: float = 0.1,
    max_completion_length: int = 512,
    max_prompt_length: int = 1024,
    sim_config=None,
) -> dict:
    """Run model on N holdout windows, return aggregate metrics."""
    from .reward import compute_reward_detailed, REWARD_SIM_CONFIG
    from .data_prompt import build_chat_messages

    cfg = sim_config or REWARD_SIM_CONFIG
    model.eval()

    per_window = []
    indices = np.random.default_rng(42).choice(len(val_snapshots), size=min(n_windows, len(val_snapshots)), replace=False)

    for idx in indices:
        prompt_text, snapshot = val_snapshots[int(idx)]
        messages = build_chat_messages(prompt_text)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_prompt_length).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_completion_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        metrics = compute_reward_detailed(completion, snapshot, cfg)
        metrics["window_id"] = snapshot.window_id
        per_window.append(metrics)

    rewards = [w["reward"] for w in per_window]
    sortinos = [w.get("sortino", 0) for w in per_window]
    returns = [w.get("total_return", 0) for w in per_window]
    drawdowns = [w.get("max_drawdown", 0) for w in per_window]
    valid_count = sum(1 for w in per_window if w.get("error") is None)

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_sortino": float(np.mean(sortinos)),
        "median_return": float(np.median(returns)),
        "p10_return": float(np.percentile(returns, 10)),
        "mean_max_drawdown": float(np.mean(drawdowns)),
        "negative_window_rate": float(sum(1 for r in returns if r < 0) / max(len(returns), 1)),
        "valid_json_rate": float(valid_count / max(len(per_window), 1)),
        "n_windows": len(per_window),
        "per_window": per_window,
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    model_size: str,
    val_snapshots: list,
    n_windows: int = 50,
) -> dict:
    """Load a checkpoint and evaluate on holdout."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .train_grpo import QWEN_MODELS

    model_id = QWEN_MODELS[model_size]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    return evaluate_on_holdout(model, tokenizer, val_snapshots, n_windows=n_windows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-size", default="0.6B")
    p.add_argument("--n-windows", type=int, default=50)
    p.add_argument("--n-symbols", type=int, default=10)
    p.add_argument("--data-dir", default="trainingdatahourly/crypto")
    args = p.parse_args()

    from .data_prompt import PromptDataset, SYMBOLS_30

    symbols = SYMBOLS_30[:args.n_symbols]
    ds = PromptDataset(
        REPO / args.data_dir, symbols,
        forecast_cache_dir=REPO / "binanceneural/forecast_cache",
        val_fraction=0.15, val_mode=True,
    )
    val_snapshots = [ds[i] for i in range(len(ds))]

    result = evaluate_checkpoint(args.checkpoint, args.model_size, val_snapshots, args.n_windows)
    print(json.dumps({k: v for k, v in result.items() if k != "per_window"}, indent=2))


if __name__ == "__main__":
    main()
