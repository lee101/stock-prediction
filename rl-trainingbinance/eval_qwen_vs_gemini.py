"""Evaluate Qwen LoRA trading plan predictions vs Gemini ground-truth."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def load_val_examples(path: Path, limit: int | None = None) -> list[dict]:
    examples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            examples.append(json.loads(line))
    return examples


def extract_user_prompt(example: dict) -> str:
    for msg in example["messages"]:
        if msg["role"] == "user":
            return msg["content"]
    return ""


def extract_ground_truth(example: dict) -> dict | None:
    for msg in example["messages"]:
        if msg["role"] == "assistant":
            try:
                return json.loads(msg["content"])
            except (json.JSONDecodeError, TypeError):
                return None
    return None


def extract_system_prompt(example: dict) -> str:
    for msg in example["messages"]:
        if msg["role"] == "system":
            return msg["content"]
    return ""


def parse_prediction(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def direction_agrees(pred: dict, truth: dict) -> bool:
    return pred.get("action", "").lower() == truth.get("action", "").lower()


def compute_metrics(predictions: list[dict | None], ground_truths: list[dict]) -> dict:
    n_total = len(ground_truths)
    n_parsed = sum(1 for p in predictions if p is not None)
    n_valid = 0
    direction_matches = 0
    confidences_pred = []
    confidences_truth = []
    entry_devs = []
    stop_devs = []
    target_devs = []
    hold_devs = []
    action_counts_pred: dict[str, int] = {}
    action_counts_truth: dict[str, int] = {}

    for pred, truth in zip(predictions, ground_truths):
        t_action = truth.get("action", "").lower()
        action_counts_truth[t_action] = action_counts_truth.get(t_action, 0) + 1
        if pred is None:
            continue
        p_action = pred.get("action", "").lower()
        if p_action not in ("long", "short", "flat"):
            continue
        n_valid += 1
        action_counts_pred[p_action] = action_counts_pred.get(p_action, 0) + 1

        if direction_agrees(pred, truth):
            direction_matches += 1

        p_conf = pred.get("confidence")
        t_conf = truth.get("confidence")
        if isinstance(p_conf, (int, float)) and isinstance(t_conf, (int, float)):
            confidences_pred.append(float(p_conf))
            confidences_truth.append(float(t_conf))

        t_entry = truth.get("entry", 0)
        if t_entry and t_entry > 0:
            p_entry = pred.get("entry", 0)
            if isinstance(p_entry, (int, float)) and p_entry > 0:
                entry_devs.append(abs(p_entry - t_entry) / t_entry * 100)
            p_stop = pred.get("stop", 0)
            t_stop = truth.get("stop", 0)
            if isinstance(p_stop, (int, float)) and isinstance(t_stop, (int, float)) and t_entry > 0:
                stop_devs.append(abs(p_stop - t_stop) / t_entry * 100)
            p_target = pred.get("target", 0)
            t_target = truth.get("target", 0)
            if isinstance(p_target, (int, float)) and isinstance(t_target, (int, float)) and t_entry > 0:
                target_devs.append(abs(p_target - t_target) / t_entry * 100)

        p_hold = pred.get("hold_hours")
        t_hold = truth.get("hold_hours")
        if isinstance(p_hold, (int, float)) and isinstance(t_hold, (int, float)):
            hold_devs.append(abs(float(p_hold) - float(t_hold)))

    conf_corr = float(np.corrcoef(confidences_pred, confidences_truth)[0, 1]) if len(confidences_pred) >= 2 else 0.0
    if np.isnan(conf_corr):
        conf_corr = 0.0

    return {
        "n_total": n_total,
        "n_parsed": n_parsed,
        "n_valid": n_valid,
        "parse_rate": n_parsed / n_total if n_total > 0 else 0.0,
        "valid_rate": n_valid / n_total if n_total > 0 else 0.0,
        "direction_agreement": direction_matches / n_valid if n_valid > 0 else 0.0,
        "confidence_correlation": conf_corr,
        "confidence_mae": float(np.mean([abs(p - t) for p, t in zip(confidences_pred, confidences_truth)])) if confidences_pred else 0.0,
        "entry_deviation_pct": float(np.mean(entry_devs)) if entry_devs else 0.0,
        "stop_deviation_pct": float(np.mean(stop_devs)) if stop_devs else 0.0,
        "target_deviation_pct": float(np.mean(target_devs)) if target_devs else 0.0,
        "hold_hours_mae": float(np.mean(hold_devs)) if hold_devs else 0.0,
        "action_distribution_pred": action_counts_pred,
        "action_distribution_truth": action_counts_truth,
    }


def format_report(metrics: dict) -> str:
    lines = [
        "=== Qwen vs Gemini Evaluation Report ===",
        "",
        f"Total examples:       {metrics['n_total']}",
        f"Parsed responses:     {metrics['n_parsed']} ({metrics['parse_rate']:.1%})",
        f"Valid predictions:     {metrics['n_valid']} ({metrics['valid_rate']:.1%})",
        "",
        "--- Agreement ---",
        f"Direction agreement:  {metrics['direction_agreement']:.1%}",
        f"Confidence corr:      {metrics['confidence_correlation']:.3f}",
        f"Confidence MAE:       {metrics['confidence_mae']:.3f}",
        "",
        "--- Price Deviation (% of entry) ---",
        f"Entry deviation:      {metrics['entry_deviation_pct']:.3f}%",
        f"Stop deviation:       {metrics['stop_deviation_pct']:.3f}%",
        f"Target deviation:     {metrics['target_deviation_pct']:.3f}%",
        "",
        "--- Hold Hours ---",
        f"Hold hours MAE:       {metrics['hold_hours_mae']:.2f}",
        "",
        "--- Action Distribution ---",
        f"  Predicted: {metrics['action_distribution_pred']}",
        f"  Truth:     {metrics['action_distribution_truth']}",
    ]
    return "\n".join(lines)


def run_inference_batch(
    model, tokenizer, prompts: list[str], system_prompt: str,
    *, max_new_tokens: int = 256, temperature: float = 0.1, batch_size: int = 8,
) -> list[str]:
    import torch
    results = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_texts = []
        for p in batch_prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p},
            ]
            batch_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        for j, seq in enumerate(out):
            input_len = inputs["input_ids"].shape[1]
            decoded = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            results.append(decoded)
        if (i + batch_size) % 100 < batch_size:
            print(f"  processed {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return results


def load_qwen_lora(model_id: str, lora_path: str | None, device: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def evaluate(
    val_path: Path,
    model_id: str,
    lora_path: str | None = None,
    limit: int | None = None,
    batch_size: int = 8,
    device: str = "auto",
    output_path: Path | None = None,
) -> dict:
    examples = load_val_examples(val_path, limit=limit)
    ground_truths = []
    prompts = []
    system_prompt = ""
    for ex in examples:
        gt = extract_ground_truth(ex)
        if gt is None:
            continue
        ground_truths.append(gt)
        prompts.append(extract_user_prompt(ex))
        if not system_prompt:
            system_prompt = extract_system_prompt(ex)

    print(f"loaded {len(prompts)} val examples")
    print(f"loading model {model_id}" + (f" + LoRA {lora_path}" if lora_path else ""))
    model, tokenizer = load_qwen_lora(model_id, lora_path, device=device)
    print("running inference...")
    raw_outputs = run_inference_batch(model, tokenizer, prompts, system_prompt, batch_size=batch_size)
    predictions = [parse_prediction(text) for text in raw_outputs]
    metrics = compute_metrics(predictions, ground_truths)
    report = format_report(metrics)
    print(report)

    result = {"metrics": metrics, "report": report}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        details = []
        for i, (pred_raw, pred, gt) in enumerate(zip(raw_outputs, predictions, ground_truths)):
            details.append({
                "idx": i,
                "prompt": prompts[i],
                "ground_truth": gt,
                "raw_output": pred_raw,
                "parsed_prediction": pred,
                "direction_match": direction_agrees(pred, gt) if pred else False,
            })
        result["details"] = details
        output_path.write_text(json.dumps(result, indent=2))
        print(f"\ndetailed results saved to {output_path}")
    return result


def main():
    ap = argparse.ArgumentParser(description="Evaluate Qwen LoRA vs Gemini ground truth")
    ap.add_argument("--val-data", type=Path, default=Path("rl-trainingbinance/trading_plans_val.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--lora-path", default=None, help="path to LoRA adapter dir")
    ap.add_argument("--limit", type=int, default=None, help="max examples to evaluate")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", type=Path, default=None, help="save detailed JSON results")
    args = ap.parse_args()
    evaluate(
        val_path=args.val_data,
        model_id=args.model,
        lora_path=args.lora_path,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
