#!/usr/bin/env python3
"""Quick RL-based stock screener: trains a mini model on each stock paired with SPY.

For each stock, builds a 2-symbol dataset (STOCK + SPY), trains 2M steps,
evaluates OOS, and ranks by learnability. Fast: ~1-3 min per stock.

Usage:
    python scripts/screen_stocks_rl.py \
        --symbols AAPL,MSFT,NVDA,LLY,BSX,COST \
        --out-dir pufferlib_market/checkpoints/pair_screen \
        --n-seeds 2

Pairs each stock with SPY as market context.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.export_data_daily import export_binary


SCREENED_SYMBOLS = [
    "LLY", "BSX", "ABBV", "VRTX", "SYK", "WELL",
    "JPM", "GS", "V", "MA", "AXP", "MS",
    "AAPL", "MSFT", "NVDA", "KLAC", "CRWD", "META",
    "COST", "AZO", "TJX",
    "CAT", "PH", "RTX",
    "BKNG", "MAR", "HLT",
    "PLTR", "SPY", "QQQ", "AMZN", "GOOG",
]

TRAIN_START = "2019-01-01"
TRAIN_END   = "2025-05-31"
VAL_START   = "2025-06-01"
VAL_END     = "2025-11-30"
STEPS_PER_RUN = 3_000_000    # ~3 min per seed at 15k sps
MAX_STEPS = 126               # ~6 months for daily
VOL_SCALES = [0.7, 1.0, 1.3]  # 3x vol augmentation
TRADE_PENALTY = 0.02


def build_pair_dataset(
    stock: str,
    anchor: str,
    csv_dir: Path,
    out_dir: Path,
    *,
    vol_scales: list[float] = VOL_SCALES,
) -> tuple[Path, Path] | None:
    """Build train+val binaries for a (stock, anchor) pair. Returns (train_path, val_path) or None."""
    pair_key = f"{stock}_{anchor}".lower()
    train_out = out_dir / f"{pair_key}_train.bin"
    val_out   = out_dir / f"{pair_key}_val.bin"

    if train_out.exists() and val_out.exists():
        return train_out, val_out  # cached

    symbols = [s for s in [stock, anchor] if s != stock or s == stock]
    # deduplicate preserving order
    seen = set()
    syms_dedup = []
    for s in [stock, anchor]:
        if s not in seen:
            syms_dedup.append(s)
            seen.add(s)

    # Build base binary
    base_train = out_dir / f"{pair_key}_base_train.bin"
    base_val   = out_dir / f"{pair_key}_base_val.bin"

    try:
        export_binary(syms_dedup, csv_dir, base_train,
                      start_date=TRAIN_START, end_date=TRAIN_END, min_days=150)
        export_binary(syms_dedup, csv_dir, base_val,
                      start_date=VAL_START, end_date=VAL_END, min_days=30)
    except Exception as e:
        print(f"  [{stock}] export failed: {e}")
        return None

    if not base_train.exists() or not base_val.exists():
        return None

    # Apply vol-scale augmentation on train only
    import struct
    import numpy as np

    HEADER_SIZE = 64
    SCALE_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
    CLIPS = {0:(-0.5,0.5),1:(-1.0,1.0),2:(-2.0,2.0),3:(0,1),4:(0,1),
             5:(-0.5,0.5),6:(-0.5,0.5),7:(-0.5,0.5),8:(0,0.5),9:(0,0.5),
             11:(-3,3),12:(-1,0),13:(-1,0)}

    def _read(p):
        with open(p, "rb") as f:
            hdr = f.read(HEADER_SIZE)
            magic, ver, nsym, nts, nfeat, nprice = struct.unpack_from("<4sIIIII", hdr)
            sym_table = f.read(nsym * 16)
            feat = np.fromfile(f, dtype=np.float32, count=nts*nsym*nfeat).reshape(nts,nsym,nfeat)
            price = np.fromfile(f, dtype=np.float32, count=nts*nsym*nprice).reshape(nts,nsym,nprice)
            mask_raw = f.read()
            mask = np.frombuffer(mask_raw, dtype=np.uint8).reshape(nts,nsym) if mask_raw else None
        return hdr, sym_table, feat, price, mask, nsym, nts, nfeat, nprice

    def _write(p, hdr, sym_table, feat, price, mask):
        nts = feat.shape[0]
        new_hdr = bytearray(hdr)
        struct.pack_into("<I", new_hdr, 12, nts)
        with open(p, "wb") as f:
            f.write(bytes(new_hdr))
            f.write(sym_table)
            f.write(feat.astype(np.float32).tobytes("C"))
            f.write(price.astype(np.float32).tobytes("C"))
            if mask is not None:
                f.write(mask.astype(np.uint8).tobytes("C"))

    hdr, sym_table, feat, price, mask, nsym, nts, nfeat, nprice = _read(base_train)

    parts_feat, parts_price, parts_mask = [], [], []
    for sigma in vol_scales:
        f_aug = feat.copy()
        for fi in SCALE_IDX:
            f_aug[:, :, fi] *= sigma
            if fi in CLIPS:
                lo, hi = CLIPS[fi]
                f_aug[:, :, fi] = f_aug[:, :, fi].clip(lo, hi)
        parts_feat.append(f_aug)
        parts_price.append(price)
        if mask is not None:
            parts_mask.append(mask)

    aug_feat = np.concatenate(parts_feat, axis=0)
    aug_price = np.concatenate(parts_price, axis=0)
    aug_mask = np.concatenate(parts_mask, axis=0) if parts_mask else None
    _write(train_out, hdr, sym_table, aug_feat, aug_price, aug_mask)

    # Val: no augmentation (just copy)
    import shutil
    shutil.copy2(base_val, val_out)

    return train_out, val_out


def train_and_eval(
    stock: str,
    train_bin: Path,
    val_bin: Path,
    ckpt_dir: Path,
    seed: int,
    python_exe: str,
    tmpdir: str,
) -> dict | None:
    """Train a mini RL model and evaluate. Returns eval dict or None."""
    seed_dir = ckpt_dir / f"s{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    eval_out = seed_dir / "eval_lag2.json"

    if eval_out.exists():
        try:
            return json.loads(eval_out.read_text())
        except Exception:
            pass

    # Train
    train_log = seed_dir / "train.log"
    train_cmd = [
        python_exe, "-u", "-m", "pufferlib_market.train",
        "--data-path", str(train_bin),
        "--val-data-path", str(val_bin),
        "--total-timesteps", str(STEPS_PER_RUN),
        "--max-steps", str(MAX_STEPS),
        "--trade-penalty", str(TRADE_PENALTY),
        "--hidden-size", "256",   # smaller model for fast screening
        "--anneal-lr",
        "--disable-shorts",
        "--val-eval-windows", "20",
        "--num-envs", "64",
        "--seed", str(seed),
        "--checkpoint-dir", str(seed_dir),
    ]
    env = {**os.environ, "TMPDIR": tmpdir}
    with open(train_log, "w") as logf:
        ret = subprocess.run(train_cmd, cwd=str(REPO), env=env, stdout=logf, stderr=logf)
    if ret.returncode != 0:
        return None

    # Eval
    ckpt = seed_dir / "val_best.pt"
    if not ckpt.exists():
        ckpt = seed_dir / "best.pt"
    if not ckpt.exists():
        return None

    eval_cmd = [
        python_exe, "-m", "pufferlib_market.evaluate_holdout",
        "--checkpoint", str(ckpt),
        "--data-path", str(val_bin),
        "--eval-hours", "40",
        "--n-windows", "20",
        "--fee-rate", "0.001",
        "--fill-buffer-bps", "5.0",
        "--decision-lag", "2",
        "--deterministic",
        "--no-early-stop",
    ]
    result = subprocess.run(eval_cmd, cwd=str(REPO), capture_output=True, text=True, env=env)
    if result.returncode != 0 or not result.stdout.strip():
        return None

    try:
        d = json.loads(result.stdout)
        eval_out.write_text(json.dumps(d, indent=2))
        return d
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="RL stock pair screener")
    parser.add_argument("--symbols", default=",".join(SCREENED_SYMBOLS),
                        help="Comma-separated list of stocks to screen")
    parser.add_argument("--anchor", default="SPY", help="Anchor/context symbol (default: SPY)")
    parser.add_argument("--csv-dir", default="trainingdata")
    parser.add_argument("--out-dir", default="pufferlib_market/checkpoints/pair_screen")
    parser.add_argument("--n-seeds", type=int, default=3, help="Seeds per stock")
    parser.add_argument("--python", default=None, help="Python executable")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    anchor = args.anchor.upper()
    csv_dir = REPO / args.csv_dir
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = str(REPO / ".tmp_train")
    Path(tmpdir).mkdir(exist_ok=True)

    python_exe = args.python or sys.executable

    leaderboard = []

    print(f"Screening {len(symbols)} stocks with anchor={anchor}, {args.n_seeds} seeds each")
    print(f"  Steps per run: {STEPS_PER_RUN:,} | Max episode: {MAX_STEPS}")
    print()

    for stock in symbols:
        if stock == anchor:
            continue
        print(f"[{stock}]")
        if args.dry_run:
            print(f"  (dry-run) would build pair dataset and train {args.n_seeds} seeds")
            continue

        pair_dir = out_dir / "data"
        pair_dir.mkdir(exist_ok=True)
        result = build_pair_dataset(stock, anchor, csv_dir, pair_dir)
        if result is None:
            print(f"  SKIP: failed to build dataset")
            continue
        train_bin, val_bin = result

        seed_results = []
        for seed in range(1, args.n_seeds + 1):
            ckpt_dir = out_dir / stock
            d = train_and_eval(stock, train_bin, val_bin, ckpt_dir, seed, python_exe, tmpdir)
            if d:
                med = d.get("median_total_return", 0) * 100
                p10 = d.get("p10_total_return", 0) * 100
                neg = d.get("negative_windows", 99)
                print(f"  s{seed}: med={med:.2f}% p10={p10:.2f}% neg={neg}/20")
                seed_results.append({"seed": seed, "med": med, "p10": p10, "neg": neg})
            else:
                print(f"  s{seed}: FAILED")

        if seed_results:
            best = max(seed_results, key=lambda x: x["med"])
            leaderboard.append({"stock": stock, **best})

    if not args.dry_run:
        leaderboard.sort(key=lambda x: x["med"], reverse=True)
        lb_path = out_dir / "leaderboard.csv"
        with open(lb_path, "w") as f:
            f.write("stock,seed,med_pct,p10_pct,neg_count\n")
            for r in leaderboard:
                f.write(f"{r['stock']},{r['seed']},{r['med']:.2f},{r['p10']:.2f},{r['neg']}\n")
        print(f"\n=== STOCK LEARNABILITY LEADERBOARD ===")
        for r in leaderboard[:15]:
            print(f"  {r['stock']:8}: med={r['med']:.2f}% p10={r['p10']:.2f}% neg={r['neg']}/20")
        print(f"\nSaved to {lb_path}")


if __name__ == "__main__":
    main()
