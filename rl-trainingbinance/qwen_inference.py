from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"


def load_model(model_id: str = MODEL_ID, device: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, *, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def market_forecast(model, tokenizer, symbol: str, prices: list[float], volumes: list[float] | None = None) -> str:
    price_str = ", ".join(f"{p:.2f}" for p in prices[-24:])
    prompt = (
        f"You are a quantitative analyst. Given the last 24 hourly close prices for {symbol}:\n"
        f"[{price_str}]\n"
        f"Provide a brief 1-2 sentence forecast for the next 1-6 hours. "
        f"Include: direction (up/down/sideways), confidence (low/medium/high), key level to watch."
    )
    return generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--symbol", type=str, default=None)
    ap.add_argument("--prices", type=str, default=None, help="comma-separated prices")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    print(f"loading {args.model}...")
    model, tokenizer = load_model(args.model)
    print("loaded")

    if args.symbol and args.prices:
        prices = [float(x) for x in args.prices.split(",")]
        out = market_forecast(model, tokenizer, args.symbol, prices)
        print(f"\n{args.symbol} forecast:\n{out}")
    elif args.prompt:
        out = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature)
        print(f"\n{out}")
    else:
        print("interactive mode (ctrl-c to exit)")
        while True:
            try:
                prompt = input("\n> ")
                if not prompt.strip():
                    continue
                out = generate(model, tokenizer, prompt, max_new_tokens=args.max_tokens, temperature=args.temperature)
                print(out)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
