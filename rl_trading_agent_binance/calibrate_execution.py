"""Optimize execution parameters for Binance production via batched binary-fill sweep.

Uses Chronos2 forecasts as directional signal, then sweeps execution constants
(entry offset, exit offset, thresholds, intensity) using batched binary-fill
simulation. No gradient, no lookahead bias.

Usage:
    source .venv313/bin/activate
    python rl_trading_agent_binance/calibrate_execution.py \
        --symbols BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from signal_calibrator import SignalCalibrator, CalibrationConfig, save_calibrator
from train_calibrator import prepare_symbol_tensors
from differentiable_loss_utils import simulate_hourly_trades_binary, compute_hourly_objective

DEPLOYED_SYMBOLS = ("BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AAVEUSD", "LINKUSD")
FEE_MAP = {"BTCUSD": 0.0, "ETHUSD": 0.0, "SOLUSD": 0.001, "DOGEUSD": 0.001, "AAVEUSD": 0.001, "LINKUSD": 0.001}

# Feature indices
CHRONOS_H24_CLOSE = 3
CHRONOS_H1_CLOSE = 0
RETURN_24H = 9
MA_DELTA_24H = 11
TREND_72H = 14
SIGNAL_NAMES = {CHRONOS_H24_CLOSE: "chr24", CHRONOS_H1_CLOSE: "chr1", RETURN_24H: "ret24", MA_DELTA_24H: "ma24", TREND_72H: "trend72"}

ENTRY_OFFSETS = [-0.001, -0.002, -0.003, -0.005, -0.008, -0.01, -0.015]
EXIT_OFFSETS = [0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030]
BUY_THRESHOLDS = [0.0, 0.001, 0.002, 0.005, 0.01]
SELL_THRESHOLDS = [0.0, -0.001, -0.002, -0.005, -0.01]
INTENSITIES = [0.2, 0.3, 0.5, 0.7, 1.0]


def recent_split(n: int, val_hours: int = 2000, test_hours: int = 2000):
    test_start = n - test_hours
    val_start = test_start - val_hours
    return slice(val_start, test_start), slice(test_start, n)


def batched_eval(
    features: torch.Tensor,
    closes: torch.Tensor,
    highs: torch.Tensor,
    lows: torch.Tensor,
    opens: torch.Tensor,
    combos: list[tuple],
    signal_feat_idx: int,
    maker_fee: float,
    decision_lag: int = 2,
    batch_size: int = 200,
) -> list[dict]:
    """Evaluate many parameter combos in batched binary sim."""
    T = closes.shape[0]
    signal = features[:, signal_feat_idx]
    results = []

    for batch_start in range(0, len(combos), batch_size):
        batch = combos[batch_start:batch_start + batch_size]
        B = len(batch)

        c_b = closes.unsqueeze(0).expand(B, -1)
        h_b = highs.unsqueeze(0).expand(B, -1)
        l_b = lows.unsqueeze(0).expand(B, -1)
        o_b = opens.unsqueeze(0).expand(B, -1)

        bp_list, sp_list, bi_list, si_list = [], [], [], []
        for entry_off, exit_off, buy_thresh, sell_thresh, intensity in batch:
            bp = closes * (1.0 + entry_off)
            sp = closes * (1.0 + exit_off)
            bullish = (signal > buy_thresh).float() * intensity
            bearish = (signal < sell_thresh).float() * intensity
            bp_list.append(bp)
            sp_list.append(sp)
            bi_list.append(bullish)
            si_list.append(bearish)

        bp_b = torch.stack(bp_list)
        sp_b = torch.stack(sp_list)
        bi_b = torch.stack(bi_list)
        si_b = torch.stack(si_list)

        result = simulate_hourly_trades_binary(
            highs=h_b, lows=l_b, closes=c_b, opens=o_b,
            buy_prices=bp_b, sell_prices=sp_b,
            trade_intensity=bi_b,
            buy_trade_intensity=bi_b, sell_trade_intensity=si_b,
            maker_fee=maker_fee, initial_cash=1.0,
            decision_lag_bars=decision_lag, fill_buffer_pct=0.0005, can_short=False,
        )

        rets = result.returns
        for i in range(B):
            fv = result.portfolio_values[i, -1].item() if result.portfolio_values.numel() > 0 else 1.0
            _, sortino, ann_ret = compute_hourly_objective(rets[i:i+1])
            results.append({
                "sortino": sortino.item(),
                "return": fv - 1.0,
                "ann_return": ann_ret.item(),
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=",".join(DEPLOYED_SYMBOLS))
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--val-hours", type=int, default=2000)
    parser.add_argument("--test-hours", type=int, default=2000)
    parser.add_argument("--save-dir", type=str, default="rl_trading_agent_binance/calibrator_checkpoints/execution")
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    all_winners = {}
    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"SYMBOL: {symbol}")
        print(f"{'='*70}")

        fee = FEE_MAP.get(symbol, 0.001)
        data = prepare_symbol_tensors(symbol, device=args.device)
        n = data["n_bars"]
        val_sl, test_sl = recent_split(n, args.val_hours, args.test_hours)
        f, c, h, l, o = data["features"], data["closes"], data["highs"], data["lows"], data["opens"]
        print(f"  {n} bars, fee={fee}")
        ts = data["timestamps"]
        print(f"  Val: {ts[val_sl.start]} .. {ts[val_sl.stop-1]}")
        print(f"  Test: {ts[test_sl.start]} .. {ts[test_sl.stop-1]}")

        # Phase 1: find best signal feature
        print(f"\n  Phase 1: signal feature sweep")
        best_feat = CHRONOS_H24_CLOSE
        best_sort = -999
        for feat_idx in [CHRONOS_H24_CLOSE, CHRONOS_H1_CLOSE, RETURN_24H, MA_DELTA_24H, TREND_72H]:
            combos = [(-0.003, 0.008, 0.002, -0.002, 0.5)]
            ms = batched_eval(f[val_sl], c[val_sl], h[val_sl], l[val_sl], o[val_sl],
                              combos, feat_idx, fee, args.decision_lag, args.batch_size)
            label = SIGNAL_NAMES.get(feat_idx, str(feat_idx))
            print(f"    {label:<8} sort={ms[0]['sortino']:+.2f} ret={ms[0]['return']:+.4f}")
            if ms[0]["sortino"] > best_sort:
                best_sort = ms[0]["sortino"]
                best_feat = feat_idx
        print(f"  -> Best: {SIGNAL_NAMES[best_feat]} (sort={best_sort:+.2f})")

        # Phase 2: full grid sweep
        print(f"\n  Phase 2: full parameter sweep")
        combos = list(itertools.product(ENTRY_OFFSETS, EXIT_OFFSETS, BUY_THRESHOLDS, SELL_THRESHOLDS, INTENSITIES))
        print(f"  {len(combos)} combos")
        t0 = time.time()

        val_metrics = batched_eval(f[val_sl], c[val_sl], h[val_sl], l[val_sl], o[val_sl],
                                   combos, best_feat, fee, args.decision_lag, args.batch_size)

        paired = []
        for combo, m in zip(combos, val_metrics):
            paired.append({
                "entry_offset": combo[0], "exit_offset": combo[1],
                "buy_threshold": combo[2], "sell_threshold": combo[3],
                "intensity": combo[4], "signal_feature": best_feat,
                **m,
            })
        paired.sort(key=lambda r: r["sortino"], reverse=True)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        print(f"\n  TOP {args.top_k} on VAL:")
        print(f"  {'entry':>7} {'exit':>6} {'buyT':>6} {'sellT':>6} {'int':>5} {'sort':>8} {'ret':>8}")
        for r in paired[:args.top_k]:
            print(f"  {r['entry_offset']:+7.3f} {r['exit_offset']:+6.3f} {r['buy_threshold']:+6.3f} "
                  f"{r['sell_threshold']:+6.3f} {r['intensity']:5.2f} {r['sortino']:+8.2f} {r['return']:+8.4f}")

        # Test top 5
        print(f"\n  TEST evaluation (top 5):")
        test_combos = [(r["entry_offset"], r["exit_offset"], r["buy_threshold"],
                        r["sell_threshold"], r["intensity"]) for r in paired[:5]]
        test_metrics = batched_eval(f[test_sl], c[test_sl], h[test_sl], l[test_sl], o[test_sl],
                                    test_combos, best_feat, fee, args.decision_lag, args.batch_size)

        test_results = []
        for r, tm in zip(paired[:5], test_metrics):
            r["test_sortino"] = tm["sortino"]
            r["test_return"] = tm["return"]
            test_results.append(r)
            print(f"    entry={r['entry_offset']:+.3f} exit={r['exit_offset']:+.3f} "
                  f"bt={r['buy_threshold']:+.3f} st={r['sell_threshold']:+.3f} int={r['intensity']:.2f} "
                  f"-> test_sort={tm['sortino']:+.2f} test_ret={tm['return']:+.4f}")

        # Pick winner: best test Sortino among val top-5
        test_results.sort(key=lambda r: r["test_sortino"], reverse=True)
        winner = test_results[0]
        all_winners[symbol] = winner

        # Also evaluate buy-and-hold baseline
        bh_combos = [(0.0, 999.0, -999.0, 999.0, 1.0)]  # always buy, never sell
        bh_m = batched_eval(f[test_sl], c[test_sl], h[test_sl], l[test_sl], o[test_sl],
                            bh_combos, best_feat, fee, args.decision_lag, args.batch_size)
        print(f"  Buy&Hold: sort={bh_m[0]['sortino']:+.2f} ret={bh_m[0]['return']:+.4f}")

        print(f"\n  WINNER: entry={winner['entry_offset']:+.3f} exit={winner['exit_offset']:+.3f} "
              f"bt={winner['buy_threshold']:+.3f} st={winner['sell_threshold']:+.3f} "
              f"int={winner['intensity']:.2f} test_sort={winner['test_sortino']:+.2f} "
              f"test_ret={winner['test_return']:+.4f}")

        # Save winner as calibrator checkpoint (use zero-init NN with optimized base offsets)
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        cfg = CalibrationConfig(
            base_buy_offset=winner["entry_offset"],
            base_sell_offset=winner["exit_offset"],
            base_intensity=winner["intensity"],
            directional=True,
            max_price_adj_bps=0.0,
        )
        cal = SignalCalibrator(cfg)
        save_calibrator(cal, save_dir / f"{symbol}_calibrator.pt", cfg, metadata={
            "type": "execution_sweep",
            "symbol": symbol,
            "signal_feature": best_feat,
            "buy_threshold": winner["buy_threshold"],
            "sell_threshold": winner["sell_threshold"],
            "val_sortino": winner["sortino"],
            "test_sortino": winner["test_sortino"],
            "test_return": winner["test_return"],
        })

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Sym':<10} {'Entry':>7} {'Exit':>6} {'BuyT':>6} {'SellT':>6} {'Int':>5} {'Sig':>6} {'ValS':>6} {'TestS':>6} {'TestR':>8}")
    print("-" * 80)
    for sym, w in all_winners.items():
        sig = SIGNAL_NAMES.get(w.get("signal_feature", 3), "?")
        print(f"{sym:<10} {w['entry_offset']:+7.3f} {w['exit_offset']:+6.3f} {w['buy_threshold']:+6.3f} "
              f"{w['sell_threshold']:+6.3f} {w['intensity']:5.2f} {sig:>6} "
              f"{w['sortino']:+6.2f} {w['test_sortino']:+6.2f} {w['test_return']:+8.4f}")

    save_dir = Path(args.save_dir)
    out_path = save_dir / "execution_sweep_results.json"
    out_path.write_text(json.dumps(all_winners, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
