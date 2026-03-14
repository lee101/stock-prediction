"""LoRA fine-tune Qwen3.5-0.8B on trading plan JSONL data (BF16)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"


class ChatJSONLDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                self.examples.append(obj["messages"])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        messages = self.examples[idx]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        enc = self.tokenizer(
            text, max_length=self.max_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # mask loss on non-assistant tokens
        labels = input_ids.clone()
        # find assistant response start
        assistant_text = messages[-1]["content"]
        assistant_enc = self.tokenizer(assistant_text, add_special_tokens=False)
        assistant_len = len(assistant_enc["input_ids"])
        # mask everything before the assistant response
        seq_len = attention_mask.sum().item()
        if assistant_len < seq_len:
            labels[: seq_len - assistant_len] = -100
        labels[attention_mask == 0] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train(args):
    print(f"loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)", flush=True)

    train_ds = ChatJSONLDataset(Path(args.train_data), tokenizer, args.max_len)
    val_ds = ChatJSONLDataset(Path(args.val_data), tokenizer, args.max_len) if args.val_data else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True) if val_ds else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    accum_steps = args.grad_accum
    total_steps = (len(train_loader) // accum_steps) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    device = next(model.parameters()).device

    for epoch in range(args.epochs):
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

            if (step + 1) % args.log_every == 0 or step == 0:
                avg = total_loss / (step + 1)
                lr_now = scheduler.get_last_lr()[0]
                print(f"  ep{epoch} step{step+1}/{len(train_loader)} loss={avg:.4f} lr={lr_now:.2e}", flush=True)

        avg_train = total_loss / len(train_loader)
        print(f"ep{epoch} train_loss={avg_train:.4f}")

        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model(**batch)
                    val_loss += out.loss.item()
            avg_val = val_loss / len(val_loader)
            print(f"ep{epoch} val_loss={avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_path = out_dir / "best"
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"  saved best -> {save_path}")

        # save every epoch
        ep_path = out_dir / f"epoch_{epoch:03d}"
        model.save_pretrained(ep_path)
        tokenizer.save_pretrained(ep_path)

    # save final
    model.save_pretrained(out_dir / "final")
    tokenizer.save_pretrained(out_dir / "final")
    print(f"done. checkpoints in {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--output-dir", default="rl-trainingbinance/checkpoints/qwen_trading")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
