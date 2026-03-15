#!/usr/bin/env python3
"""
Production daily mixed stock+crypto trading bot.

Uses the C env binding directly for inference — guaranteed to match training.
Exports fresh daily bars to MKTD binary, runs policy forward pass through
the C env observation pipeline, decodes action to trade signal.

Usage:
    # Generate today's signal
    python -u trade_mixed_daily_prod.py --once

    # Run as daily daemon (executes at midnight UTC)
    python -u trade_mixed_daily_prod.py --daemon

    # Dry run (print signal without executing)
    python -u trade_mixed_daily_prod.py --once --dry-run
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "PLTR", "NET",
    "JPM", "V", "SPY", "QQQ", "NFLX", "AMD",
    "BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD", "DOGEUSD", "LINKUSD", "AAVEUSD",
]

DEFAULT_CHECKPOINT = "pufferlib_market/checkpoints/autoresearch_mixed23_daily/ent_anneal/best.pt"
DATA_DIR = "trainingdata/train"
TEMP_BIN = "/tmp/mixed23_daily_live.bin"


def export_live_binary(symbols: list[str], data_dir: str, output: str, lookback_days: int = 120):
    """Export latest daily bars to MKTD binary for C env inference."""
    from pufferlib_market.export_data_daily import compute_daily_features, FEATURES_PER_SYM, PRICE_FEATURES, MAGIC

    all_prices = {}
    for sym in symbols:
        for subdir in ["", "crypto", "stocks"]:
            p = Path(data_dir) / subdir / f"{sym}.csv" if subdir else Path(data_dir) / f"{sym}.csv"
            if p.exists():
                df = pd.read_csv(p)
                df.columns = [c.lower() for c in df.columns]
                ts = "timestamp" if "timestamp" in df.columns else "date"
                df["timestamp"] = pd.to_datetime(df[ts], utc=True)
                df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
                req = ["open", "high", "low", "close", "volume"]
                if all(c in df.columns for c in req):
                    all_prices[sym] = df.set_index("timestamp")[req].astype(float)
                break

    if len(all_prices) != len(symbols):
        missing = set(symbols) - set(all_prices.keys())
        print(f"WARNING: Missing data for {missing}")

    # Use last lookback_days
    latest = max(df.index.max() for df in all_prices.values())
    start = latest - pd.Timedelta(days=lookback_days + 30)  # extra buffer for MA computation
    full_index = pd.date_range(start.floor("D"), latest.floor("D"), freq="D", tz="UTC")

    num_symbols = len(symbols)
    num_timesteps = len(full_index)

    feature_arr = np.zeros((num_timesteps, num_symbols, FEATURES_PER_SYM), dtype=np.float32)
    price_arr = np.zeros((num_timesteps, num_symbols, PRICE_FEATURES), dtype=np.float32)
    mask_arr = np.ones((num_timesteps, num_symbols), dtype=np.uint8)

    for si, sym in enumerate(symbols):
        if sym not in all_prices:
            mask_arr[:, si] = 0
            continue
        df = all_prices[sym]
        mask = full_index.isin(df.index).astype(np.uint8)
        re = df.reindex(full_index, method="ffill")
        re["volume"] = re["volume"].where(mask.astype(bool), 0.0)
        re = re.bfill().fillna(0.0)
        feats = compute_daily_features(re)
        feature_arr[:, si, :] = feats.values.astype(np.float32, copy=False)
        price_arr[:, si, :] = re[["open", "high", "low", "close", "volume"]].values.astype(np.float32, copy=False)
        mask_arr[:, si] = mask

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        header = struct.pack("<4sIIIII40s", MAGIC, 2, num_symbols, num_timesteps,
                            FEATURES_PER_SYM, PRICE_FEATURES, b"\x00" * 40)
        f.write(header)
        for sym in symbols:
            raw = sym.encode("ascii", errors="ignore")[:15]
            f.write(raw + b"\x00" * (16 - len(raw)))
        f.write(feature_arr.tobytes(order="C"))
        f.write(price_arr.tobytes(order="C"))
        f.write(mask_arr.tobytes(order="C"))

    print(f"Exported {output}: {num_symbols} symbols, {num_timesteps} days")
    print(f"  Latest date: {full_index[-1].strftime('%Y-%m-%d')}")
    return output, num_symbols, num_timesteps


def run_inference(checkpoint: str, data_bin: str, hidden_size: int = 1024):
    """Run single-step C env inference to get today's signal."""
    from pufferlib_market.train import TradingPolicy
    import pufferlib_market.binding as binding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read binary header
    with open(data_bin, "rb") as f:
        header = f.read(64)
    _, _, num_symbols, num_timesteps, _, _ = struct.unpack("<4sIIIII", header[:24])

    # Read symbol names
    with open(data_bin, "rb") as f:
        f.seek(64)
        sym_names = []
        for _ in range(num_symbols):
            name = f.read(16).decode("ascii", errors="ignore").rstrip("\x00")
            sym_names.append(name)

    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = 1 + 2 * num_symbols

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = TradingPolicy(obs_size, num_actions, hidden_size)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    policy.to(device)

    # Use C env for proper observation construction
    binding.shared(data_path=str(Path(data_bin).resolve()))

    # Run a single episode starting from the END of data (most recent)
    num_envs = 1
    obs_buf = np.zeros((num_envs, obs_size), dtype=np.float32)
    act_buf = np.zeros((num_envs,), dtype=np.int32)
    rew_buf = np.zeros((num_envs,), dtype=np.float32)
    term_buf = np.zeros((num_envs,), dtype=np.uint8)
    trunc_buf = np.zeros((num_envs,), dtype=np.uint8)

    vec_handle = binding.vec_init(
        obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
        num_envs, 42,  # seed
        max_steps=90,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=365.0,
        action_allocation_bins=1,
        action_level_bins=1,
        action_max_offset_bps=0.0,
        fill_slippage_bps=5.0,
    )

    # Reset to get initial observation (starts near end of data)
    binding.vec_reset(vec_handle, 42)

    # Get policy action for current state
    obs_tensor = torch.from_numpy(obs_buf).to(device)
    with torch.no_grad():
        logits = policy.actor(policy.encoder(obs_tensor))
        value = policy.critic(policy.encoder(obs_tensor))

    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    action = int(logits.argmax().item())
    confidence = float(probs[action])

    # Decode action
    if action == 0:
        direction = "FLAT"
        symbol = None
    elif action <= num_symbols:
        direction = "LONG"
        symbol = sym_names[action - 1]
    else:
        direction = "SHORT"
        symbol = sym_names[action - num_symbols - 1]

    # Get current prices for context
    # Read last row of price data
    with open(data_bin, "rb") as f:
        f.seek(64 + num_symbols * 16)  # skip header + symbol table
        f.seek(num_timesteps * num_symbols * 16 * 4, 1)  # skip features
        # Go to last timestep of prices
        f.seek((num_timesteps - 1) * num_symbols * 5 * 4, 1)
        last_prices = np.frombuffer(f.read(num_symbols * 5 * 4), dtype=np.float32).reshape(num_symbols, 5)

    binding.vec_close(vec_handle)

    # Build signal report
    print(f"\n{'='*60}")
    print(f"DAILY SIGNAL — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")
    print(f"  Direction:  {direction}")
    print(f"  Symbol:     {symbol or 'N/A (flat)'}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"  Value est:  {float(value.item()):.4f}")

    if symbol:
        sym_idx = sym_names.index(symbol)
        price = last_prices[sym_idx]
        print(f"  Price:      O={price[0]:.2f} H={price[1]:.2f} L={price[2]:.2f} C={price[3]:.2f}")

    # Top 5 actions by probability
    print(f"\n  Top 5 actions:")
    top5 = np.argsort(probs)[-5:][::-1]
    for idx in top5:
        if idx == 0:
            label = "FLAT"
        elif idx <= num_symbols:
            label = f"LONG  {sym_names[idx-1]}"
        else:
            label = f"SHORT {sym_names[idx-num_symbols-1]}"
        print(f"    {label:20s} prob={probs[idx]:.3f}")

    return {
        "direction": direction,
        "symbol": symbol,
        "confidence": confidence,
        "value": float(value.item()),
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Production daily mixed trading bot")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--once", action="store_true", help="Generate one signal and exit")
    parser.add_argument("--daemon", action="store_true", help="Run daily at midnight UTC")
    parser.add_argument("--dry-run", action="store_true", help="Print signal without executing")
    parser.add_argument("--hidden-size", type=int, default=1024)
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS

    if args.once or args.dry_run:
        # Export latest data and run inference
        print(f"Exporting daily data for {len(symbols)} symbols...")
        data_bin, _, _ = export_live_binary(symbols, args.data_dir, TEMP_BIN)
        signal = run_inference(args.checkpoint, data_bin, args.hidden_size)

        if not args.dry_run and signal["symbol"]:
            print(f"\n  ACTION: {signal['direction']} {signal['symbol']}")
            print(f"  TODO: Execute via Alpaca/Binance API")
        elif args.dry_run:
            print(f"\n  [DRY RUN] Would {signal['direction']} {signal['symbol'] or 'stay flat'}")

    elif args.daemon:
        print(f"Starting daily daemon for {len(symbols)} symbols...")
        while True:
            now = datetime.now(timezone.utc)
            # Run at 00:05 UTC daily
            next_run = now.replace(hour=0, minute=5, second=0, microsecond=0)
            if next_run <= now:
                next_run += pd.Timedelta(days=1)
            wait_s = (next_run - now).total_seconds()
            print(f"Next run: {next_run.isoformat()} (in {wait_s/3600:.1f}h)")
            time.sleep(wait_s)

            try:
                data_bin, _, _ = export_live_binary(symbols, args.data_dir, TEMP_BIN)
                signal = run_inference(args.checkpoint, data_bin, args.hidden_size)
                print(f"\n  ACTION: {signal['direction']} {signal['symbol'] or 'FLAT'}")
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
