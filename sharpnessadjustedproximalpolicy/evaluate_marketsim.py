#!/usr/bin/env python3
"""Realistic marketsim evaluation for SAP checkpoints.

Uses chunk-based inference (non-overlapping seq_len windows) with continuous
carry-over simulation. This matches how the model was trained and produces
realistic P&L numbers comparable to pufferlib holdout eval.

Usage:
    python -m sharpnessadjustedproximalpolicy.evaluate_marketsim \
        --symbol DOGEUSD

    python -m sharpnessadjustedproximalpolicy.evaluate_marketsim \
        --symbols DOGEUSD BTCUSD ARBUSD --top-k 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Disable triton compilation errors
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.model import PolicyConfig, build_policy
from differentiable_loss_utils import simulate_hourly_trades_binary
from src.torch_load_utils import torch_load_compat


@dataclass
class MarketsimResult:
    symbol: str
    checkpoint: str
    epoch: int
    total_hours: int
    total_return_pct: float
    sortino: float
    max_drawdown_pct: float
    num_trades: int
    win_rate: float
    time_in_position_pct: float
    sharpe: float


def load_model(ckpt_path: Path):
    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    state_dict = ckpt.get("state_dict", {})
    feature_cols = ckpt.get("feature_columns", [])

    # Infer input_dim from embed weight
    input_dim = 0
    for key in ("embed.weight", "_orig_mod.embed.weight"):
        if key in state_dict:
            input_dim = state_dict[key].shape[1]
            break

    # Infer horizons from feature columns
    horizons = set()
    for col in feature_cols:
        if col.startswith("chronos_") and "_h" in col:
            try:
                horizons.add(int(col.split("_h")[-1]))
            except ValueError:
                pass
    horizons = tuple(sorted(horizons)) if horizons else (1, 24)

    pc = PolicyConfig(
        input_dim=input_dim,
        hidden_dim=config.get("transformer_dim", 256),
        num_heads=config.get("transformer_heads", 8),
        num_layers=config.get("transformer_layers", 4),
        model_arch=config.get("model_arch", "classic"),
        max_len=max(config.get("sequence_length", 72), 32),
        use_flex_attention=False,
    )

    model = build_policy(pc)
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    return model, config, feature_cols, horizons


def chunk_based_eval(
    ckpt_path: Path,
    symbol: str,
    lag: int = 2,
    n_windows: int = 0,
    window_hours: int = 720,
) -> list[MarketsimResult]:
    """Chunk-based inference with continuous carry-over simulation.

    The model processes non-overlapping seq_len chunks (matching training),
    then all actions are concatenated and run through a single binary-fill sim.
    """
    model, config, feature_cols, horizons = load_model(ckpt_path)
    seq_len = config.get("sequence_length", 72)

    ds_cfg = DatasetConfig(
        symbol=symbol,
        data_root=Path("trainingdatahourly") / "crypto",
        forecast_cache_root=Path("binanceneural") / "forecast_cache",
        forecast_horizons=horizons,
        feature_columns=feature_cols if feature_cols else None,
        sequence_length=seq_len,
        validation_days=70,
        cache_only=True,
    )
    dm = BinanceHourlyDataModule(ds_cfg)
    vds = dm.val_dataset
    T = len(vds.frame)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    scale = config.get("trade_amount_scale", 100.0)
    fee = config.get("maker_fee", 0.001)
    max_lev = config.get("max_leverage", 1.0)
    fill_buf = config.get("fill_buffer_pct", 0.0005)
    margin_rate = config.get("margin_annual_rate", 0.0625)

    # Collect actions from non-overlapping chunks
    all_bp, all_sp, all_ti, all_bi, all_si = [], [], [], [], []
    with torch.inference_mode():
        for start in range(0, T - seq_len + 1, seq_len):
            batch = vds[start]
            feat = batch["features"].unsqueeze(0).to(device)
            ref = batch["reference_close"].unsqueeze(0).to(device)
            ch_h = batch["chronos_high"].unsqueeze(0).to(device)
            ch_l = batch["chronos_low"].unsqueeze(0).to(device)
            outputs = model(feat)
            actions = model.decode_actions(outputs, reference_close=ref, chronos_high=ch_h, chronos_low=ch_l)
            all_bp.append(actions["buy_price"].squeeze(0))
            all_sp.append(actions["sell_price"].squeeze(0))
            all_ti.append(actions["trade_amount"].squeeze(0) / scale)
            all_bi.append(actions["buy_amount"].squeeze(0) / scale)
            all_si.append(actions["sell_amount"].squeeze(0) / scale)

    bp = torch.cat(all_bp).unsqueeze(0)
    sp = torch.cat(all_sp).unsqueeze(0)
    ti = torch.cat(all_ti).unsqueeze(0)
    bi = torch.cat(all_bi).unsqueeze(0)
    si = torch.cat(all_si).unsqueeze(0)
    total_bars = bp.shape[1]

    highs = torch.from_numpy(vds.highs[:total_bars]).unsqueeze(0).to(device)
    lows = torch.from_numpy(vds.lows[:total_bars]).unsqueeze(0).to(device)
    closes = torch.from_numpy(vds.closes[:total_bars]).unsqueeze(0).to(device)
    opens = torch.from_numpy(vds.opens[:total_bars]).unsqueeze(0).to(device)

    def run_window(s: int, e: int) -> MarketsimResult:
        sim = simulate_hourly_trades_binary(
            highs=highs[:, s:e], lows=lows[:, s:e], closes=closes[:, s:e], opens=opens[:, s:e],
            buy_prices=bp[:, s:e], sell_prices=sp[:, s:e],
            trade_intensity=ti[:, s:e], buy_trade_intensity=bi[:, s:e], sell_trade_intensity=si[:, s:e],
            maker_fee=fee, initial_cash=1.0, can_short=False, can_long=True,
            max_leverage=max_lev, fill_buffer_pct=fill_buf,
            margin_annual_rate=margin_rate, decision_lag_bars=lag,
        )
        pv = sim.portfolio_values.float().cpu().squeeze()
        rets = sim.returns.float().cpu().squeeze()
        inv = sim.inventory_path.float().cpu().squeeze()
        n = e - s

        total_ret = (pv[-1] / pv[0] - 1).item() * 100
        mean_r = rets.mean().item()
        down_std = (rets.clamp(max=0.0).square().mean() + 1e-10).sqrt().item()
        total_std = (rets.var() + 1e-10).sqrt().item()
        ann = 8760.0 / max(n, 1)
        sortino = (mean_r / max(down_std, 1e-10)) * (ann ** 0.5)
        sharpe = (mean_r / max(total_std, 1e-10)) * (ann ** 0.5)
        cummax = pv.cummax(dim=0).values
        max_dd = ((pv - cummax) / cummax.clamp(min=1e-10)).min().item() * 100

        # Trade counting
        n_trades = 0
        wins = 0
        was_flat = True
        entry_val = 0.0
        for t_i in range(len(inv)):
            is_flat = abs(inv[t_i].item()) < 1e-8
            if was_flat and not is_flat:
                entry_val = pv[min(t_i, len(pv) - 1)].item()
            elif not was_flat and is_flat:
                n_trades += 1
                if pv[min(t_i, len(pv) - 1)].item() > entry_val:
                    wins += 1
            was_flat = is_flat

        tip = (inv.abs() > 1e-8).float().mean().item() * 100

        return MarketsimResult(
            symbol=symbol, checkpoint=str(ckpt_path), epoch=ckpt.get("epoch", 0),
            total_hours=n, total_return_pct=round(total_ret, 4),
            sortino=round(sortino, 4), max_drawdown_pct=round(max_dd, 4),
            num_trades=n_trades, win_rate=round(wins / max(n_trades, 1), 4),
            time_in_position_pct=round(tip, 1), sharpe=round(sharpe, 4),
        )

    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)

    if n_windows <= 0:
        return [run_window(0, total_bars)]

    results = []
    if total_bars <= window_hours:
        return [run_window(0, total_bars)]
    stride = max(1, (total_bars - window_hours) // max(n_windows - 1, 1))
    for i in range(n_windows):
        s = i * stride
        e = min(s + window_hours, total_bars)
        if s >= total_bars:
            break
        results.append(run_window(s, e))
    return results


def find_best_checkpoints(ckpt_dir: Path, top_k: int = 1) -> list[Path]:
    history_path = ckpt_dir / "sap_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        ranked = sorted(history, key=lambda h: h.get("val_sortino", -999), reverse=True)
        best = [h["epoch"] for h in ranked[:top_k]]
        return [ckpt_dir / f"epoch_{ep:03d}.pt" for ep in best if (ckpt_dir / f"epoch_{ep:03d}.pt").exists()]
    return sorted(ckpt_dir.glob("epoch_*.pt"))[:top_k]


def summarize_windows(results: list[MarketsimResult]) -> dict:
    if not results:
        return {}
    rets = [r.total_return_pct for r in results]
    return {
        "n_windows": len(results),
        "median_return": round(float(np.median(rets)), 2),
        "mean_return": round(float(np.mean(rets)), 2),
        "p10_return": round(float(np.percentile(rets, 10)), 2),
        "positive_pct": round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1),
        "median_sortino": round(float(np.median([r.sortino for r in results])), 2),
        "median_dd": round(float(np.median([r.max_drawdown_pct for r in results])), 2),
        "worst_dd": round(float(min(r.max_drawdown_pct for r in results)), 2),
        "median_trades": int(np.median([r.num_trades for r in results])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--symbol", default="DOGEUSD")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--n-windows", type=int, default=10)
    parser.add_argument("--window-hours", type=int, default=720)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    symbols = args.symbols or [args.symbol]
    ckpt_root = Path("sharpnessadjustedproximalpolicy/checkpoints")

    all_results = {}
    for sym in symbols:
        print(f"\n{'='*60}\n{sym}\n{'='*60}", flush=True)

        if args.checkpoint:
            paths = [Path(args.checkpoint)]
        else:
            run_dirs = sorted(ckpt_root.glob(f"*{sym}*"))
            paths = []
            for rd in run_dirs:
                if rd.is_dir():
                    paths.extend(find_best_checkpoints(rd, 1))
            # Deduplicate and keep top-k by epoch number variety
            seen = set()
            unique = []
            for p in paths:
                key = p.parent.name
                if key not in seen:
                    seen.add(key)
                    unique.append(p)
            paths = unique[:args.top_k]

        if not paths:
            print(f"  No checkpoints found", flush=True)
            continue

        sym_results = []
        for p in paths:
            if not p.exists():
                continue
            run_name = p.parent.name.replace(f"_{sym}_", " ").split("sap_")[-1].split(" ")[0]
            print(f"\n  {run_name} / {p.name}...", flush=True)
            t0 = time.time()
            try:
                results = chunk_based_eval(p, sym, lag=args.lag, n_windows=args.n_windows, window_hours=args.window_hours)
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                import traceback; traceback.print_exc()
                continue
            wall = time.time() - t0

            if args.n_windows > 0 and len(results) > 1:
                s = summarize_windows(results)
                print(
                    f"    med_ret={s['median_return']:.1f}% pos={s['positive_pct']:.0f}% "
                    f"sort={s['median_sortino']:.2f} dd={s['median_dd']:.1f}% "
                    f"trades={s['median_trades']} ({wall:.0f}s)", flush=True)
                sym_results.append({"run": run_name, "checkpoint": str(p), **s})
            else:
                r = results[0]
                print(
                    f"    ret={r.total_return_pct:.1f}% sort={r.sortino:.2f} "
                    f"dd={r.max_drawdown_pct:.1f}% trades={r.num_trades} "
                    f"wr={r.win_rate:.0%} tip={r.time_in_position_pct:.0f}% ({wall:.0f}s)", flush=True)
                sym_results.append(vars(r))

        all_results[sym] = sym_results

    out = Path("sharpnessadjustedproximalpolicy") / f"marketsim_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved: {out}", flush=True)

    # Summary
    print(f"\n{'Symbol':<10} {'Config':<35} {'Return':>8} {'Sortino':>8} {'MaxDD':>8} {'Trades':>7}")
    print("-" * 85)
    for sym, res_list in all_results.items():
        for r in res_list:
            name = r.get("run", Path(r.get("checkpoint", "")).parent.name[:30])
            if "median_return" in r:
                print(f"{sym:<10} {name:<35} {r['median_return']:>7.1f}% {r['median_sortino']:>8.2f} {r['median_dd']:>7.1f}% {r['median_trades']:>7}")
            else:
                print(f"{sym:<10} {name:<35} {r['total_return_pct']:>7.1f}% {r['sortino']:>8.2f} {r['max_drawdown_pct']:>7.1f}% {r['num_trades']:>7}")


if __name__ == "__main__":
    main()
