"""Fast directional calibrator training on recent data windows.

Uses shorter windows (default 4000h train, 1500h val, 1500h test from recent data)
for fast iteration. Directional mode with separate buy/sell intensities.

Usage:
    source .venv313/bin/activate
    python rl_trading_agent_binance/train_directional.py \
        --symbols BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from signal_calibrator import SignalCalibrator, CalibrationConfig, save_calibrator
from train_calibrator import (
    DEPLOYED_SYMBOLS, prepare_symbol_tensors, run_sim,
)
from differentiable_loss_utils import (
    combined_sortino_pnl_loss,
    DEFAULT_MAKER_FEE_RATE,
)

FEE_MAP = {
    "BTCUSD": 0.0, "ETHUSD": 0.0,
    "SOLUSD": 0.001, "DOGEUSD": 0.001, "AAVEUSD": 0.001, "LINKUSD": 0.001,
}

def recent_split(n: int, train_hours: int = 4000, val_hours: int = 1500, test_hours: int = 1500):
    total = train_hours + val_hours + test_hours
    if total > n:
        scale = n / total
        train_hours = int(train_hours * scale)
        val_hours = int(val_hours * scale)
        test_hours = n - train_hours - val_hours
    start = n - total
    return slice(start, start + train_hours), slice(start + train_hours, start + train_hours + val_hours), slice(start + train_hours + val_hours, n)


def train_one(
    symbol: str,
    data: dict,
    config: CalibrationConfig,
    epochs: int,
    lr: float,
    weight_decay: float,
    maker_fee: float,
    temperature: float,
    decision_lag: int,
    device: str,
    return_weight: float,
    train_sl: slice,
    val_sl: slice,
    test_sl: slice,
    save_dir: Optional[Path] = None,
) -> dict:
    cal = SignalCalibrator(config).to(device)
    opt = torch.optim.AdamW(cal.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    best_val_sort = -float("inf")
    best_state = None
    best_ep = -1
    history = []

    f, c, h, l, o = data["features"], data["closes"], data["highs"], data["lows"], data["opens"]

    t0 = time.time()
    for ep in range(epochs):
        cal.train()
        opt.zero_grad()
        ret, tm = run_sim(cal, f[train_sl], c[train_sl], h[train_sl], l[train_sl], o[train_sl],
                          maker_fee=maker_fee, temperature=temperature, decision_lag=decision_lag, binary=False)
        loss = combined_sortino_pnl_loss(ret.unsqueeze(0), return_weight=return_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cal.parameters(), 1.0)
        opt.step()
        sched.step()

        cal.eval()
        with torch.no_grad():
            _, vm = run_sim(cal, f[val_sl], c[val_sl], h[val_sl], l[val_sl], o[val_sl],
                            maker_fee=maker_fee, decision_lag=decision_lag, binary=True)

        history.append({"ep": ep, "train_sort": tm["sortino"], "train_ret": tm["return"],
                        "val_sort": vm["sortino"], "val_ret": vm["return"]})

        if vm["sortino"] > best_val_sort:
            best_val_sort = vm["sortino"]
            best_state = {k: v.clone() for k, v in cal.state_dict().items()}
            best_ep = ep

        if ep % 10 == 0 or ep == epochs - 1:
            elapsed = time.time() - t0
            print(f"  [{symbol}] ep={ep:3d} t_sort={tm['sortino']:+.2f} t_ret={tm['return']:+.4f} "
                  f"v_sort={vm['sortino']:+.2f} v_ret={vm['return']:+.4f} best={best_val_sort:+.2f}@{best_ep} "
                  f"[{elapsed:.0f}s]")

    if best_state:
        cal.load_state_dict(best_state)
    cal.eval()
    with torch.no_grad():
        _, test_m = run_sim(cal, f[test_sl], c[test_sl], h[test_sl], l[test_sl], o[test_sl],
                            maker_fee=maker_fee, decision_lag=decision_lag, binary=True)

    print(f"  [{symbol}] TEST sort={test_m['sortino']:+.2f} ret={test_m['return']:+.4f} @ep{best_ep}")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_calibrator(cal, save_dir / f"{symbol}_calibrator.pt", config, metadata={
            "symbol": symbol, "best_epoch": best_ep,
            "test_sortino": test_m["sortino"], "test_return": test_m["return"],
            "val_sortino": best_val_sort, "maker_fee": maker_fee,
        })

    return {"symbol": symbol, "best_epoch": best_ep, "best_val_sortino": best_val_sort,
            "test_metrics": test_m, "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=",".join(DEPLOYED_SYMBOLS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max-price-bps", type=float, default=100.0)
    parser.add_argument("--max-amount-adj", type=float, default=0.5)
    parser.add_argument("--base-buy-offset", type=float, default=-0.005)
    parser.add_argument("--base-sell-offset", type=float, default=0.010)
    parser.add_argument("--base-intensity", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--return-weight", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="rl_trading_agent_binance/calibrator_checkpoints/directional")
    parser.add_argument("--train-hours", type=int, default=4000)
    parser.add_argument("--val-hours", type=int, default=1500)
    parser.add_argument("--test-hours", type=int, default=1500)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    save_dir = Path(args.save_dir)

    print(f"Symbols: {symbols}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Windows: train={args.train_hours}h, val={args.val_hours}h, test={args.test_hours}h")
    print()

    results = []
    for sym in symbols:
        print(f"--- {sym} ---")
        fee = FEE_MAP.get(sym, 0.001)
        try:
            data = prepare_symbol_tensors(sym, device=args.device)
            n = data["n_bars"]
            train_sl, val_sl, test_sl = recent_split(n, args.train_hours, args.val_hours, args.test_hours)
            print(f"  {n} bars, fee={fee}, train={train_sl}, val={val_sl}, test={test_sl}")

            config = CalibrationConfig(
                hidden=args.hidden,
                max_price_adj_bps=args.max_price_bps,
                max_amount_adj=args.max_amount_adj,
                base_buy_offset=args.base_buy_offset,
                base_sell_offset=args.base_sell_offset,
                base_intensity=args.base_intensity,
                directional=True,
            )

            # Baseline (zero-init calibrator)
            base_cal = SignalCalibrator(config).to(args.device)
            base_cal.eval()
            with torch.no_grad():
                _, bm = run_sim(base_cal, data["features"][test_sl], data["closes"][test_sl],
                                data["highs"][test_sl], data["lows"][test_sl], data["opens"][test_sl],
                                maker_fee=fee, binary=True, decision_lag=args.decision_lag)
            print(f"  baseline sort={bm['sortino']:+.2f} ret={bm['return']:+.4f}")

            r = train_one(sym, data, config, args.epochs, args.lr, args.weight_decay,
                          fee, args.temperature, args.decision_lag, args.device,
                          args.return_weight, train_sl, val_sl, test_sl, save_dir)
            r["baseline"] = bm
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Sym':<10} {'Base':>8} {'Test':>8} {'Ret':>10} {'Ep':>4} {'Delta':>8}")
    print("-" * 50)
    for r in results:
        bs = r["baseline"]["sortino"]
        ts = r["test_metrics"]["sortino"]
        tr = r["test_metrics"]["return"]
        print(f"{r['symbol']:<10} {bs:+8.2f} {ts:+8.2f} {tr:+10.4f} {r['best_epoch']:4d} {ts-bs:+8.2f}")


if __name__ == "__main__":
    main()
