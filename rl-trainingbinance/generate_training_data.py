"""Generate fine-tuning data for Qwen trading plan model.

Reads hourly candle CSVs, computes features over sliding windows,
labels each window with actual future returns, and outputs JSONL
chat-format training examples.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


def load_candles(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame, idx: int, lookback: int = 24) -> dict | None:
    if idx < lookback or idx + 6 >= len(df):
        return None
    window = df.iloc[idx - lookback : idx + 1]
    closes = window["close"].values
    volumes = window["volume"].values
    current = closes[-1]
    ret_1h = (closes[-1] / closes[-2] - 1) * 100 if closes[-2] > 0 else 0
    ret_6h = (closes[-1] / closes[-6] - 1) * 100 if closes[-6] > 0 else 0
    ret_24h = (closes[-1] / closes[0] - 1) * 100 if closes[0] > 0 else 0
    high_24h = window["high"].max()
    low_24h = window["low"].min()
    vol_24h = np.std(np.diff(closes) / closes[:-1]) * 100 if len(closes) > 1 else 0
    avg_vol = np.mean(volumes) if len(volumes) > 0 else 0
    vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
    ema12 = pd.Series(closes).ewm(span=12).mean().iloc[-1]
    ema_pct = (current / ema12 - 1) * 100 if ema12 > 0 else 0

    # future returns for labeling
    fwd = df.iloc[idx + 1 : idx + 7]
    fwd_closes = fwd["close"].values
    fwd_1h = (fwd_closes[0] / current - 1) * 100 if len(fwd_closes) >= 1 else 0
    fwd_3h = (fwd_closes[2] / current - 1) * 100 if len(fwd_closes) >= 3 else 0
    fwd_6h = (fwd_closes[5] / current - 1) * 100 if len(fwd_closes) >= 6 else 0
    fwd_max = (fwd["high"].max() / current - 1) * 100
    fwd_min = (fwd["low"].min() / current - 1) * 100

    return {
        "current": round(current, 2),
        "ret_1h": round(ret_1h, 3),
        "ret_6h": round(ret_6h, 3),
        "ret_24h": round(ret_24h, 3),
        "high_24h": round(high_24h, 2),
        "low_24h": round(low_24h, 2),
        "vol_24h": round(vol_24h, 3),
        "vol_ratio": round(vol_ratio, 2),
        "ema_pct": round(ema_pct, 3),
        "fwd_1h": round(fwd_1h, 3),
        "fwd_3h": round(fwd_3h, 3),
        "fwd_6h": round(fwd_6h, 3),
        "fwd_max": round(fwd_max, 3),
        "fwd_min": round(fwd_min, 3),
    }


def make_trading_plan(symbol: str, feat: dict) -> dict:
    fwd_3h = feat["fwd_3h"]
    fwd_6h = feat["fwd_6h"]
    fwd_max = feat["fwd_max"]
    fwd_min = feat["fwd_min"]
    current = feat["current"]

    # determine optimal action from hindsight
    long_edge = fwd_max - abs(fwd_min * 0.3)
    short_edge = abs(fwd_min) - fwd_max * 0.3

    if long_edge > 0.15 and fwd_6h > 0.05:
        action = "long"
        confidence = min(0.95, 0.5 + long_edge / 3)
        stop_pct = max(0.3, abs(fwd_min) * 0.8)
        target_pct = fwd_max * 0.7
        hold = 6 if fwd_6h > fwd_3h else 3
    elif short_edge > 0.15 and fwd_6h < -0.05:
        action = "short"
        confidence = min(0.95, 0.5 + short_edge / 3)
        stop_pct = max(0.3, fwd_max * 0.8)
        target_pct = abs(fwd_min) * 0.7
        hold = 6 if fwd_6h < feat["fwd_3h"] else 3
    else:
        action = "flat"
        confidence = 0.6
        stop_pct = 0
        target_pct = 0
        hold = 0

    fwd_3h = feat["fwd_3h"]
    stop = round(current * (1 - stop_pct / 100), 2) if action == "long" else round(current * (1 + stop_pct / 100), 2) if action == "short" else 0
    target = round(current * (1 + target_pct / 100), 2) if action == "long" else round(current * (1 - target_pct / 100), 2) if action == "short" else 0

    # reasoning from features
    reasons = []
    if feat["ret_1h"] > 0.3:
        reasons.append("strong 1h momentum")
    elif feat["ret_1h"] < -0.3:
        reasons.append("1h selling pressure")
    if feat["ret_24h"] > 1:
        reasons.append("24h uptrend")
    elif feat["ret_24h"] < -1:
        reasons.append("24h downtrend")
    if feat["vol_ratio"] > 1.5:
        reasons.append("elevated volume")
    if feat["ema_pct"] > 0.5:
        reasons.append("above EMA12")
    elif feat["ema_pct"] < -0.5:
        reasons.append("below EMA12")
    if feat["vol_24h"] > 1.5:
        reasons.append("high volatility")
    if not reasons:
        reasons.append("mixed signals")

    return {
        "action": action,
        "confidence": round(confidence, 2),
        "entry": current,
        "stop": stop,
        "target": target,
        "hold_hours": hold,
        "reasoning": ", ".join(reasons),
    }


def format_prompt(symbol: str, feat: dict) -> str:
    return (
        f"{symbol} | price={feat['current']} "
        f"ret1h={feat['ret_1h']}% ret6h={feat['ret_6h']}% ret24h={feat['ret_24h']}% "
        f"hi24={feat['high_24h']} lo24={feat['low_24h']} "
        f"vol={feat['vol_24h']}% vratio={feat['vol_ratio']} ema={feat['ema_pct']}%"
    )


def format_response(plan: dict) -> str:
    return json.dumps(plan, separators=(",", ":"))


SYSTEM_PROMPT = (
    "You are a crypto trading assistant. Given market data, output a JSON trading plan "
    "with fields: action (long/short/flat), confidence (0-1), entry, stop, target, "
    "hold_hours (1-6), reasoning."
)


def generate_examples(data_dir: Path, symbols: list[str] | None, stride: int, max_per_symbol: int) -> list[dict]:
    csvs = sorted(data_dir.glob("*.csv"))
    if symbols:
        sym_set = {s.upper() for s in symbols}
        csvs = [c for c in csvs if c.stem.upper() in sym_set]

    examples = []
    for csv_path in csvs:
        sym = csv_path.stem.upper()
        print(f"  {sym}: ", end="", flush=True)
        df = load_candles(csv_path)
        if len(df) < 48:
            print("skip (too short)")
            continue

        count = 0
        indices = list(range(24, len(df) - 6, stride))
        random.shuffle(indices)
        for idx in indices:
            if count >= max_per_symbol:
                break
            feat = compute_features(df, idx)
            if feat is None:
                continue
            plan = make_trading_plan(sym, feat)
            prompt = format_prompt(sym, feat)
            response = format_response(plan)
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
            })
            count += 1
        print(f"{count} examples")

    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("trainingdatahourly"))
    ap.add_argument("--output", type=Path, default=Path("rl-trainingbinance/trading_plans_train.jsonl"))
    ap.add_argument("--symbols", type=str, default=None, help="comma-separated, e.g. BTCUSD,ETHUSD")
    ap.add_argument("--stride", type=int, default=4, help="hours between samples")
    ap.add_argument("--max-per-symbol", type=int, default=5000)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    syms = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    print(f"generating from {args.data_dir}...")
    examples = generate_examples(args.data_dir, syms, args.stride, args.max_per_symbol)
    random.shuffle(examples)

    val_n = int(len(examples) * args.val_split)
    val_examples = examples[:val_n]
    train_examples = examples[val_n:]

    train_path = args.output
    val_path = train_path.with_name(train_path.stem.replace("train", "val") + train_path.suffix)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"train: {len(train_examples)} -> {train_path}")
    print(f"val:   {len(val_examples)} -> {val_path}")


if __name__ == "__main__":
    main()
