"""On-policy distillation of a small Qwen student from a larger teacher on trading prompts.

Uses trl 1.2.0 `DistillationTrainer` (experimental). Reads existing trading plan
prompts from rl-trainingbinance/trading_plans_train.jsonl (27k crypto plans,
chat-formatted with system+user+assistant). The assistant message is discarded;
the teacher regenerates completions on-policy via the student's own rollouts,
and the student learns to match the teacher's next-token distribution (JSD loss).

Launch:
    .venv313/bin/python -m trltraining.distill_trading --max-steps 200 --time-budget-s 300

Smallest usable student+teacher pair in the Qwen3.5 dense series:
    student: Qwen/Qwen3.5-0.8B  (1.2 GB bf16, cached)
    teacher: Qwen/Qwen3.5-4B    (~8 GB bf16)

Both models share the Qwen3.5 tokenizer so forward/reverse KL with top-k
support is well-defined.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
from trl.experimental.distillation import DistillationConfig, DistillationTrainer


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO / "rl-trainingbinance" / "trading_plans_train.jsonl"


def load_prompt_dataset(jsonl_path: Path, limit: int | None) -> Dataset:
    """Read trading_plans_train.jsonl and keep only system+user messages as prompts."""
    rows: list[dict] = []
    with jsonl_path.open("r") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            rec = json.loads(line)
            msgs = rec.get("messages") or []
            prompt_msgs = [m for m in msgs if m.get("role") in ("system", "user")]
            if not prompt_msgs:
                continue
            rows.append({"messages": prompt_msgs})
    if not rows:
        raise RuntimeError(f"No prompts loaded from {jsonl_path}")
    return Dataset.from_list(rows)


class TimeBudgetCallback(TrainerCallback):
    """Stop training after `time_budget_s` seconds of wall clock. Keeps the
    5-minute smoke-test honest when the step count overshoots."""

    def __init__(self, time_budget_s: float):
        self.time_budget_s = float(time_budget_s)
        self._start: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._start = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self._start is None or self.time_budget_s <= 0:
            return control
        if time.monotonic() - self._start >= self.time_budget_s:
            control.should_training_stop = True
        return control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--dataset-limit", type=int, default=2048,
                    help="How many prompts to keep from the jsonl")
    ap.add_argument("--output-dir", default="trltraining/checkpoints/distill_trading_smoke")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--time-budget-s", type=float, default=300.0,
                    help="Wall clock stop after this many seconds (0 disables)")
    ap.add_argument("--lmbda", type=float, default=1.0,
                    help="1.0=on-policy, 0.0=off-policy, blend in between")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="0=forward KL, 0.5=JSD, 1=reverse KL")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--loss-top-k", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--max-completion-length", type=int, default=192)
    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--log-completions", action="store_true")
    args = ap.parse_args()

    jsonl_path = Path(args.dataset)
    assert jsonl_path.exists(), f"dataset missing: {jsonl_path}"

    print(f"[distill] loading prompts from {jsonl_path} (limit={args.dataset_limit})", flush=True)
    train_dataset = load_prompt_dataset(jsonl_path, args.dataset_limit)
    print(f"[distill] {len(train_dataset)} prompt rows loaded", flush=True)

    # Shared tokenizer (same family — Qwen3.5 variants share vocab).
    tok = AutoTokenizer.from_pretrained(args.student)

    cfg = DistillationConfig(
        output_dir=args.output_dir,
        teacher_model_name_or_path=args.teacher,
        teacher_model_init_kwargs={"torch_dtype": "bfloat16"},
        model_init_kwargs={"torch_dtype": "bfloat16"},
        lmbda=args.lmbda,
        beta=args.beta,
        temperature=args.temperature,
        loss_top_k=args.loss_top_k,
        max_length=args.max_length,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        report_to=["none"],
        log_completions=args.log_completions,
        disable_dropout=True,
        remove_unused_columns=False,
    )

    print(f"[distill] student={args.student} teacher={args.teacher}", flush=True)
    print(f"[distill] lmbda={cfg.lmbda} beta={cfg.beta} temp={cfg.temperature} "
          f"top_k={cfg.loss_top_k} max_steps={cfg.max_steps} time_budget={args.time_budget_s}s",
          flush=True)

    trainer = DistillationTrainer(
        model=args.student,
        args=cfg,
        train_dataset=train_dataset,
        processing_class=tok,
        callbacks=[TimeBudgetCallback(args.time_budget_s)] if args.time_budget_s > 0 else None,
    )
    trainer.train()
    trainer.save_model()
    print(f"[distill] done. saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
