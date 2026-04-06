"""Multi-window evaluation of calibrator checkpoints.

Tests trained calibrators across multiple overlapping time windows
to verify robustness (no single-window overfitting).

Usage:
    source .venv313/bin/activate
    python rl_trading_agent_binance/eval_calibrator_multiwindow.py \
        --checkpoint-dir rl_trading_agent_binance/calibrator_checkpoints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from signal_calibrator import CalibrationConfig, SignalCalibrator, load_calibrator
from train_calibrator import prepare_symbol_tensors, run_sim


DEPLOYED_SYMBOLS = ("BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AAVEUSD", "LINKUSD")
FEE_MAP = {
    "BTCUSD": 0.0,
    "ETHUSD": 0.0,
    "SOLUSD": 0.001,
    "DOGEUSD": 0.001,
    "AAVEUSD": 0.001,
    "LINKUSD": 0.001,
}
WINDOWS_HOURS = [24 * 3, 24 * 7, 24 * 14, 24 * 30, 24 * 60, 24 * 90]
WINDOW_NAMES = ["3d", "7d", "14d", "30d", "60d", "90d"]


def eval_window(
    cal: SignalCalibrator,
    data: dict,
    start: int,
    end: int,
    maker_fee: float,
    device: str = "cpu",
) -> dict:
    sl = slice(start, end)
    with torch.no_grad():
        _, metrics = run_sim(
            cal,
            data["features"][sl],
            data["closes"][sl],
            data["highs"][sl],
            data["lows"][sl],
            data["opens"][sl],
            maker_fee=maker_fee,
            fill_buffer_pct=0.0005,
            decision_lag=2,
            binary=True,
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="rl_trading_agent_binance/calibrator_checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)

    for symbol in DEPLOYED_SYMBOLS:
        ckpt_path = ckpt_dir / f"{symbol}_calibrator.pt"
        if not ckpt_path.exists():
            print(f"{symbol}: no checkpoint found")
            continue

        cal, cfg = load_calibrator(ckpt_path, device=args.device)
        data = prepare_symbol_tensors(symbol, device=args.device)
        n = data["n_bars"]
        fee = FEE_MAP.get(symbol, 0.001)

        baseline_cfg = CalibrationConfig()
        baseline = SignalCalibrator(baseline_cfg).to(args.device)
        baseline.eval()

        print(f"\n{'=' * 70}")
        print(f"{symbol} (n={n}, fee={fee})")
        print(
            f"  Config: buy_off={cfg.base_buy_offset:.4f} sell_off={cfg.base_sell_offset:.4f} "
            f"intensity={cfg.base_intensity:.2f} max_adj={cfg.max_price_adj_bps}bps"
        )
        print(f"  {'Window':>6} {'Cal Sort':>10} {'Cal Ret':>10} {'Base Sort':>10} {'Base Ret':>10} {'Delta':>8}")
        print(f"  {'-' * 60}")

        for wname, whours in zip(WINDOW_NAMES, WINDOWS_HOURS):
            if whours > n:
                continue
            start = n - whours
            end = n
            cal_m = eval_window(cal, data, start, end, fee, args.device)
            base_m = eval_window(baseline, data, start, end, fee, args.device)
            delta = cal_m["sortino"] - base_m["sortino"]
            print(
                f"  {wname:>6} {cal_m['sortino']:+10.2f} {cal_m['return']:+10.4f} "
                f"{base_m['sortino']:+10.2f} {base_m['return']:+10.4f} {delta:+8.2f}"
            )


if __name__ == "__main__":
    main()
