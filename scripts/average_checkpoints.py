#!/usr/bin/env python3
"""
Stochastic Weight Averaging (SWA) for Chronos2 checkpoints.

Averages the weights of the last N checkpoints saved during training.
This produces a smoother model that often generalises better than a
single checkpoint (Model Soups / SWA technique).

Usage:
    # Average last 3 checkpoints of v2, save to averaged-ckpt/
    python scripts/average_checkpoints.py \
        --trainer-workspace chronos2_finetuned/stocks_all_v2/trainer_workspace \
        --output chronos2_finetuned/stocks_all_v2/swa-ckpt \
        --n-last 3

    # Average all checkpoints
    python scripts/average_checkpoints.py \
        --trainer-workspace chronos2_finetuned/stocks_all_v3/trainer_workspace \
        --output chronos2_finetuned/stocks_all_v3/swa-ckpt \
        --n-last 0  # 0 means all

The output is a regular model directory that can be used with
chronos2_linear_calibration.py and benchmark_chronos2.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import torch
from safetensors.torch import load_file, save_file


def find_checkpoints(workspace: Path, n_last: int) -> List[Path]:
    """Return sorted checkpoint paths from trainer_workspace, optionally limited to last N."""
    ckpts = sorted(
        [d for d in workspace.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if not ckpts:
        raise ValueError(f"No checkpoints found in {workspace}")
    if n_last > 0:
        ckpts = ckpts[-n_last:]
    return ckpts


def average_weights(ckpt_dirs: List[Path], exp_decay: float = 0.0) -> dict:
    """Load and average safetensors weights from multiple checkpoint directories.

    exp_decay: if > 0, apply exponential weighting — more recent checkpoints
        (higher index in sorted order) receive weight proportional to
        exp(exp_decay * i). exp_decay=0 gives uniform weights (default / SWA).
        exp_decay=1.0 gives the most recent checkpoint ~2.7x more weight than
        the second-most-recent (and e^n more than the oldest).
    """
    import numpy as np

    n = len(ckpt_dirs)
    # Compute per-checkpoint weights
    if exp_decay > 0.0:
        raw = np.array([math.exp(exp_decay * i) for i in range(n)], dtype=np.float64)
        weights = raw / raw.sum()
    else:
        weights = np.ones(n, dtype=np.float64) / n

    print(f"Averaging {n} checkpoints (exp_decay={exp_decay:.2f}):")
    for d, w in zip(ckpt_dirs, weights):
        print(f"  {d.name}  weight={w:.4f}")

    accumulated: Optional[dict] = None

    for ckpt_dir, w in zip(ckpt_dirs, weights):
        weights_path = ckpt_dir / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"model.safetensors not found in {ckpt_dir}")
        state = load_file(str(weights_path))
        if accumulated is None:
            accumulated = {k: v.float() * float(w) for k, v in state.items()}
        else:
            for k, v in state.items():
                accumulated[k] = accumulated[k] + v.float() * float(w)

    assert accumulated is not None
    # Convert back to the original dtype (bfloat16 or float32)
    # Use bfloat16 to match training precision
    return {k: v.to(torch.bfloat16) for k, v in accumulated.items()}


def copy_config(src_ckpt: Path, dst_dir: Path) -> None:
    """Copy config.json and other non-weight files from the latest checkpoint."""
    for fname in ["config.json", "training_args.bin"]:
        src = src_ckpt / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Average Chronos2 checkpoints (SWA / Model Soups)")
    p.add_argument("--trainer-workspace", required=True,
                   help="Path to trainer_workspace directory (contains checkpoint-XXXX dirs)")
    p.add_argument("--output", required=True,
                   help="Output directory for averaged checkpoint")
    p.add_argument("--n-last", type=int, default=3,
                   help="Number of last checkpoints to average (0 = all)")
    p.add_argument("--copy-chronos-config", default=None,
                   help="Also copy chronos_config from this path (e.g. finetuned-ckpt dir) "
                        "so the output is a full Chronos2 pipeline directory")
    p.add_argument("--exp-decay", type=float, default=0.0,
                   help="Exponential weighting decay for checkpoints: 0=uniform SWA (default), "
                        ">0=more recent checkpoints get higher weight (e.g. 0.5 or 1.0)")
    args = p.parse_args(argv)

    workspace = Path(args.trainer_workspace)
    output = Path(args.output)

    if not workspace.exists():
        print(f"ERROR: workspace not found: {workspace}")
        return 1

    ckpts = find_checkpoints(workspace, args.n_last)
    print(f"Found {len(ckpts)} checkpoints, using last {len(ckpts)}")

    output.mkdir(parents=True, exist_ok=True)

    averaged = average_weights(ckpts, exp_decay=getattr(args, "exp_decay", 0.0))
    out_weights = output / "model.safetensors"
    save_file(averaged, str(out_weights))
    print(f"Saved averaged weights → {out_weights} ({out_weights.stat().st_size / 1e6:.1f} MB)")

    # Copy config from the latest checkpoint
    copy_config(ckpts[-1], output)

    # Optionally copy full Chronos2 pipeline structure from finetuned-ckpt
    if args.copy_chronos_config:
        src = Path(args.copy_chronos_config)
        for fname in src.iterdir():
            dst = output / fname.name
            if fname.is_file() and not dst.exists():
                shutil.copy2(fname, dst)
                print(f"  Copied {fname.name}")
            elif fname.is_dir() and not dst.exists():
                shutil.copytree(fname, dst)
                print(f"  Copied dir {fname.name}/")

    print(f"\nDone. Averaged checkpoint → {output}")
    print(f"Test with: python chronos2_linear_calibration.py --model-id {output} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
