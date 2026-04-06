"""Generalized GRPO launcher for trading-plan training with TRL + vLLM."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from .config import TRLTradingConfig
from .dataset import build_dataset_bundle
from .methods import recommend_trainer


log = logging.getLogger("trltraining.grpo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_grpo_kwargs(config: TRLTradingConfig, *, supported_args: set[str] | None = None) -> dict[str, object]:
    """Build filtered kwargs for TRL's GRPOConfig constructor."""
    config.validate()
    kwargs: dict[str, object] = {
        "output_dir": str(config.output_path),
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_generations": config.group_size,
        "max_completion_length": config.max_completion_length,
        "max_prompt_length": config.max_prompt_length,
        "beta": config.kl_coef,
        "seed": config.seed,
        "logging_steps": 10,
        "save_steps": config.eval_every_steps,
        "save_total_limit": config.top_k_checkpoints,
        "bf16": config.bf16,
        "report_to": list(config.report_to) if config.report_to else "none",
        "log_level": "warning",
    }
    if config.use_vllm:
        kwargs.update(
            {
                "use_vllm": True,
                "vllm_mode": config.vllm_mode,
                "vllm_tensor_parallel_size": config.vllm_tensor_parallel_size,
                "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
            }
        )

    if supported_args is None:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in supported_args}


def _legacy_qwen_config(config: TRLTradingConfig):
    from qwen_rl_trading.train_grpo import QwenGRPOConfig

    model_size_map = {
        "qwen2_05b_instruct": "0.6B",
        "qwen2_15b_instruct": "1.8B",
        "qwen2_3b_instruct": "3B",
        "qwen2_7b_instruct": "7B",
    }
    return QwenGRPOConfig(
        model_size=model_size_map[config.model_preset],
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        group_size=config.group_size,
        lr=config.learning_rate,
        kl_coef=config.kl_coef,
        max_completion_length=config.max_completion_length,
        max_prompt_length=config.max_prompt_length,
        num_train_epochs=config.num_train_epochs,
        per_device_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        n_symbols=config.n_symbols,
        eval_horizon_hours=config.eval_horizon_hours,
        lookback_hours=config.lookback_hours,
        stride=config.stride,
        prompt_variant=config.prompt_variant,
        reward_type=config.reward_type,
        eval_every_steps=config.eval_every_steps,
        early_stop_patience=config.early_stop_patience,
        top_k_checkpoints=config.top_k_checkpoints,
        sft_warmstart=config.sft_warmstart,
        sft_epochs=config.sft_epochs,
        seed=config.seed,
        data_dir=config.data_dir,
        forecast_cache_dir=config.forecast_cache_dir,
        sft_data=config.sft_data,
        output_dir=config.output_dir,
        time_budget=config.time_budget,
        description=config.description,
    )


def train(config: TRLTradingConfig) -> dict[str, object]:
    """Run GRPO training with the repo's existing trading reward and dataset."""
    config.validate()
    if config.trainer_type != "grpo":
        raise ValueError(f"trainer_type must be 'grpo', got {config.trainer_type!r}")

    from datasets import Dataset
    from qwen_rl_trading.reward import GRPORewardFn
    from qwen_rl_trading.train_grpo import (
        build_model_and_tokenizer,
        run_sft_warmstart,
        run_validation,
    )
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    t0 = time.time()
    deadline = t0 + config.time_budget if config.time_budget > 0 else float("inf")
    legacy = _legacy_qwen_config(config)
    model, tokenizer, lora_config = build_model_and_tokenizer(legacy)

    if config.sft_warmstart:
        from peft import get_peft_model

        model = get_peft_model(model, lora_config)
        model = run_sft_warmstart(legacy, model, tokenizer)
        sft_dir = Path(config.output_dir) / "sft_warmstart"
        sft_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(sft_dir))
        lora_config = None

    dataset_bundle = build_dataset_bundle(config)
    if not dataset_bundle.train_prompts:
        return {"error": "no_data"}

    reward_fn = GRPORewardFn(dataset_bundle.snapshot_map, reward_type=config.reward_type)
    train_dataset = Dataset.from_list(dataset_bundle.train_prompts)

    config.output_path.mkdir(parents=True, exist_ok=True)
    supported_grpo_args = set(inspect.signature(GRPOConfig.__init__).parameters)
    grpo_config = GRPOConfig(**build_grpo_kwargs(config, supported_args=supported_grpo_args))

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    best_val_reward = -float("inf")
    patience_counter = 0
    val_history: list[float] = []
    best_checkpoint_path: str | None = None

    class _EarlyStopCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
            nonlocal best_val_reward, patience_counter, best_checkpoint_path
            if time.time() > deadline:
                control.should_training_stop = True
                return

            if state.global_step <= 0 or state.global_step % config.eval_every_steps != 0:
                return

            val_metrics = run_validation(model, tokenizer, dataset_bundle.val_snapshots, legacy)
            val_reward = float(val_metrics["val_mean_reward"])
            val_history.append(val_reward)
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                patience_counter = 0
                ckpt_path = config.output_path / f"best_step{state.global_step}"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                trainer.save_model(str(ckpt_path))
                best_checkpoint_path = str(ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= config.early_stop_patience:
                    control.should_training_stop = True

    trainer.add_callback(_EarlyStopCallback())
    trainer.train()

    final_dir = config.output_path / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    final_val = run_validation(model, tokenizer, dataset_bundle.val_snapshots, legacy)

    result = {
        "description": config.description or config.model_preset,
        "trainer_type": config.trainer_type,
        "model_name": config.model_name,
        "best_checkpoint": best_checkpoint_path or str(final_dir),
        "training_time_s": time.time() - t0,
        "val_history": val_history,
        **final_val,
        "config": asdict(config),
        "recommendation": asdict(recommend_trainer()),
    }
    with open(config.output_path / "result.json", "w") as handle:
        json.dump(result, handle, indent=2, default=str)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading-plan GRPO trainer with TRL + vLLM")
    parser.add_argument("--model-preset", default="qwen2_05b_instruct")
    parser.add_argument("--output-dir", default="trltraining/checkpoints/default")
    parser.add_argument("--prompt-variant", default="with_chronos2")
    parser.add_argument("--reward-type", default="sortino_drawdown")
    parser.add_argument("--n-symbols", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--use-vllm", action="store_true", default=True)
    parser.add_argument("--no-vllm", action="store_false", dest="use_vllm")
    parser.add_argument("--vllm-mode", default="colocate")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--time-budget", type=int, default=0)
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    config = TRLTradingConfig(
        model_preset=args.model_preset,
        output_dir=args.output_dir,
        prompt_variant=args.prompt_variant,
        reward_type=args.reward_type,
        n_symbols=args.n_symbols,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        time_budget=args.time_budget,
        description=args.description,
    )
    print(json.dumps(train(config), indent=2, default=str))


if __name__ == "__main__":
    main()
