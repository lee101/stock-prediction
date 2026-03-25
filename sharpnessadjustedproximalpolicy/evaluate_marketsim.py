#!/usr/bin/env python3
"""Realistic marketsim evaluation for SAP checkpoints.

Runs continuous binary-fill simulation over the FULL validation period
(not 72-bar chunks). This gives comparable numbers to pufferlib holdout eval.

Usage:
    python -m sharpnessadjustedproximalpolicy.evaluate_marketsim \
        --checkpoint path/to/epoch_005.pt --symbol DOGEUSD

    python -m sharpnessadjustedproximalpolicy.evaluate_marketsim \
        --checkpoint-dir path/to/run/ --symbol DOGEUSD --top-k 3

    # Multi-symbol portfolio eval
    python -m sharpnessadjustedproximalpolicy.evaluate_marketsim \
        --checkpoint-dir sharpnessadjustedproximalpolicy/checkpoints \
        --symbols DOGEUSD BTCUSD ETHUSD --portfolio
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

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
    avg_hold_hours: float
    sharpe: float
    calmar: float
    final_equity: float


def load_model_from_checkpoint(ckpt_path: Path):
    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    state_dict = ckpt.get("state_dict", {})
    # Infer input_dim from embed weight shape
    input_dim = 0
    for key in ("embed.weight", "_orig_mod.embed.weight"):
        if key in state_dict:
            input_dim = state_dict[key].shape[1]
            break
    pc = PolicyConfig(
        input_dim=input_dim,
        hidden_dim=config.get("transformer_dim", 256),
        num_heads=config.get("transformer_heads", 8),
        num_layers=config.get("transformer_layers", 4),
        model_arch=config.get("model_arch", "classic"),
        max_len=max(config.get("sequence_length", 72), 32),
        use_flex_attention=False,
    )
    return ckpt, config, pc


def continuous_eval(
    ckpt_path: Path,
    symbol: str,
    lag: int = 2,
    n_windows: int = 0,
    window_hours: int = 720,
) -> list[MarketsimResult]:
    """Run continuous binary-fill sim over full val period.

    If n_windows > 0, splits val into sliding windows and reports per-window.
    Otherwise reports single result over entire val period.
    """
    ckpt, config, pc = load_model_from_checkpoint(ckpt_path)

    seq_len = config.get("sequence_length", 72)

    # Infer forecast horizons from checkpoint feature_columns
    feature_cols = ckpt.get("feature_columns", [])
    inferred_horizons = set()
    for col in feature_cols:
        if col.startswith("chronos_") and "_h" in col:
            h = col.split("_h")[-1]
            try:
                inferred_horizons.add(int(h))
            except ValueError:
                pass
    if inferred_horizons:
        horizons = tuple(sorted(inferred_horizons))
    else:
        horizons = tuple(config.get("forecast_horizons", [1, 24]))
        if isinstance(horizons[0], list):
            horizons = tuple(horizons[0])

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

    # Verify feature dim matches checkpoint
    actual_dim = len(dm.feature_columns)
    if pc.input_dim > 0 and actual_dim != pc.input_dim:
        print(f"    WARN: feature dim mismatch: checkpoint={pc.input_dim} data={actual_dim}", flush=True)
        pc = PolicyConfig(
            input_dim=actual_dim,
            hidden_dim=pc.hidden_dim,
            num_heads=pc.num_heads,
            num_layers=pc.num_layers,
            model_arch=pc.model_arch,
            max_len=pc.max_len,
            use_flex_attention=False,
        )
    elif pc.input_dim == 0:
        pc = PolicyConfig(
            input_dim=actual_dim,
            hidden_dim=pc.hidden_dim,
            num_heads=pc.num_heads,
            num_layers=pc.num_layers,
            model_arch=pc.model_arch,
            max_len=pc.max_len,
            use_flex_attention=False,
        )

    model = build_policy(pc)
    state_dict = ckpt.get("state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get full val data as continuous arrays
    vds = dm.val_dataset
    T = len(vds.frame)
    features_all = torch.from_numpy(vds.features).to(device)
    opens_all = torch.from_numpy(vds.opens).to(device)
    highs_all = torch.from_numpy(vds.highs).to(device)
    lows_all = torch.from_numpy(vds.lows).to(device)
    closes_all = torch.from_numpy(vds.closes).to(device)
    ref_close_all = torch.from_numpy(vds.reference_close).to(device)
    ch_high_all = torch.from_numpy(vds.chronos_high).to(device)
    ch_low_all = torch.from_numpy(vds.chronos_low).to(device)

    fee = config.get("maker_fee", 0.001)
    scale = config.get("trade_amount_scale", 100.0)
    max_lev = config.get("max_leverage", 1.0)
    fill_buf = config.get("fill_buffer_pct", 0.0005)
    margin_rate = config.get("margin_annual_rate", 0.0625)

    # Rolling inference: for each hour, use last seq_len hours of data
    # Collect actions hour-by-hour
    all_buy_prices = []
    all_sell_prices = []
    all_trade_intensity = []
    all_buy_intensity = []
    all_sell_intensity = []

    with torch.inference_mode():
        # Process in chunks for efficiency
        # For hours < seq_len, pad with first available data
        for t in range(T):
            start = max(0, t - seq_len + 1)
            end = t + 1
            pad_len = seq_len - (end - start)

            feat_window = features_all[start:end]
            if pad_len > 0:
                feat_window = torch.cat([feat_window[:1].expand(pad_len, -1), feat_window], dim=0)

            ref_window = ref_close_all[start:end]
            ch_h_window = ch_high_all[start:end]
            ch_l_window = ch_low_all[start:end]
            if pad_len > 0:
                ref_window = torch.cat([ref_window[:1].expand(pad_len), ref_window], dim=0)
                ch_h_window = torch.cat([ch_h_window[:1].expand(pad_len), ch_h_window], dim=0)
                ch_l_window = torch.cat([ch_l_window[:1].expand(pad_len), ch_l_window], dim=0)

            feat_batch = feat_window.unsqueeze(0)
            ref_batch = ref_window.unsqueeze(0)
            ch_h_batch = ch_h_window.unsqueeze(0)
            ch_l_batch = ch_l_window.unsqueeze(0)

            outputs = model(feat_batch)
            actions = model.decode_actions(
                outputs,
                reference_close=ref_batch,
                chronos_high=ch_h_batch,
                chronos_low=ch_l_batch,
            )

            # Use LAST position's action (the current hour prediction)
            all_buy_prices.append(actions["buy_price"][0, -1])
            all_sell_prices.append(actions["sell_price"][0, -1])
            all_trade_intensity.append(actions["trade_amount"][0, -1] / scale)
            all_buy_intensity.append(actions["buy_amount"][0, -1] / scale)
            all_sell_intensity.append(actions["sell_amount"][0, -1] / scale)

    buy_prices = torch.stack(all_buy_prices).unsqueeze(0)
    sell_prices = torch.stack(all_sell_prices).unsqueeze(0)
    trade_int = torch.stack(all_trade_intensity).unsqueeze(0)
    buy_int = torch.stack(all_buy_intensity).unsqueeze(0)
    sell_int = torch.stack(all_sell_intensity).unsqueeze(0)

    highs_sim = highs_all.unsqueeze(0)
    lows_sim = lows_all.unsqueeze(0)
    closes_sim = closes_all.unsqueeze(0)
    opens_sim = opens_all.unsqueeze(0)

    def run_sim_window(start_idx: int, end_idx: int) -> MarketsimResult:
        s, e = start_idx, end_idx
        sim = simulate_hourly_trades_binary(
            highs=highs_sim[:, s:e],
            lows=lows_sim[:, s:e],
            closes=closes_sim[:, s:e],
            opens=opens_sim[:, s:e],
            buy_prices=buy_prices[:, s:e],
            sell_prices=sell_prices[:, s:e],
            trade_intensity=trade_int[:, s:e],
            buy_trade_intensity=buy_int[:, s:e],
            sell_trade_intensity=sell_int[:, s:e],
            maker_fee=fee,
            initial_cash=1.0,
            can_short=False,
            can_long=True,
            max_leverage=max_lev,
            fill_buffer_pct=fill_buf,
            margin_annual_rate=margin_rate,
            decision_lag_bars=lag,
        )
        returns = sim.returns.float().cpu().squeeze(0)
        pv = sim.portfolio_values.float().cpu().squeeze(0)
        inv = sim.inventory.float().cpu().squeeze(0) if hasattr(sim, "inventory") else None

        total_ret = (pv[-1] / pv[0] - 1).item() * 100
        final_eq = pv[-1].item()

        # Sortino (not annualized -- raw period sortino for comparison)
        mean_r = returns.mean().item()
        downside = returns.clamp(max=0.0)
        down_std = (downside.square().mean() + 1e-10).sqrt().item()
        n_hours = e - s
        # Annualize properly based on actual window length
        ann_factor = 8760.0 / max(n_hours, 1)
        sortino_ann = (mean_r / max(down_std, 1e-10)) * (ann_factor ** 0.5)

        # Sharpe
        total_std = (returns.var() + 1e-10).sqrt().item()
        sharpe_ann = (mean_r / max(total_std, 1e-10)) * (ann_factor ** 0.5)

        # Max drawdown
        cummax = pv.cummax(dim=0).values
        dd = (pv - cummax) / cummax.clamp(min=1e-10)
        max_dd = dd.min().item() * 100

        # Trade counting from inventory_path
        num_trades = 0
        win_trades = 0
        total_hold = 0
        inv_path = sim.inventory_path.float().cpu().squeeze(0) if hasattr(sim, "inventory_path") else None
        if inv_path is not None and inv_path.ndim >= 1 and inv_path.shape[0] > 1:
            was_flat = True
            entry_val = 0.0
            entry_t = 0
            for t_i in range(inv_path.shape[0]):
                is_flat = abs(inv_path[t_i].item()) < 1e-8
                if was_flat and not is_flat:
                    entry_val = pv[min(t_i, len(pv)-1)].item()
                    entry_t = t_i
                elif not was_flat and is_flat:
                    num_trades += 1
                    total_hold += t_i - entry_t
                    if pv[min(t_i, len(pv)-1)].item() > entry_val:
                        win_trades += 1
                was_flat = is_flat

        wr = win_trades / max(num_trades, 1)
        avg_hold = total_hold / max(num_trades, 1)
        calmar = (total_ret / 100 * ann_factor) / max(abs(max_dd / 100), 1e-10) if max_dd != 0 else 0

        return MarketsimResult(
            symbol=symbol,
            checkpoint=str(ckpt_path),
            epoch=ckpt.get("epoch", 0),
            total_hours=n_hours,
            total_return_pct=round(total_ret, 4),
            sortino=round(sortino_ann, 4),
            max_drawdown_pct=round(max_dd, 4),
            num_trades=num_trades,
            win_rate=round(wr, 4),
            avg_hold_hours=round(avg_hold, 1),
            sharpe=round(sharpe_ann, 4),
            calmar=round(calmar, 4),
            final_equity=round(final_eq, 6),
        )

    if n_windows <= 0:
        return [run_sim_window(0, T)]

    # Sliding windows
    results = []
    if T <= window_hours:
        return [run_sim_window(0, T)]
    stride = max(1, (T - window_hours) // max(n_windows - 1, 1))
    for i in range(n_windows):
        start = i * stride
        end = min(start + window_hours, T)
        if start >= T:
            break
        results.append(run_sim_window(start, end))
    return results


def find_best_checkpoints(ckpt_dir: Path, symbol: str, top_k: int = 3) -> list[Path]:
    """Find top-k checkpoints by val sortino from sap_history.json."""
    history_path = ckpt_dir / "sap_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        ranked = sorted(history, key=lambda h: h.get("val_sortino", -999), reverse=True)
        best_epochs = [h["epoch"] for h in ranked[:top_k]]
        return [ckpt_dir / f"epoch_{ep:03d}.pt" for ep in best_epochs if (ckpt_dir / f"epoch_{ep:03d}.pt").exists()]
    # Fallback: use all checkpoints
    return sorted(ckpt_dir.glob("epoch_*.pt"))[:top_k]


def summarize_windows(results: list[MarketsimResult]) -> dict:
    """Compute aggregate stats across sliding windows."""
    if not results:
        return {}
    rets = [r.total_return_pct for r in results]
    sorts = [r.sortino for r in results]
    dds = [r.max_drawdown_pct for r in results]
    return {
        "n_windows": len(results),
        "median_return": round(float(np.median(rets)), 4),
        "mean_return": round(float(np.mean(rets)), 4),
        "p10_return": round(float(np.percentile(rets, 10)), 4),
        "p90_return": round(float(np.percentile(rets, 90)), 4),
        "positive_pct": round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1),
        "median_sortino": round(float(np.median(sorts)), 4),
        "median_max_dd": round(float(np.median(dds)), 4),
        "worst_dd": round(float(min(dds)), 4),
        "median_trades": int(np.median([r.num_trades for r in results])),
        "median_wr": round(float(np.median([r.win_rate for r in results])), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--symbol", default="DOGEUSD")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--n-windows", type=int, default=10, help="0=full period, >0=sliding windows")
    parser.add_argument("--window-hours", type=int, default=720, help="hours per window (720=30d)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--portfolio", action="store_true")
    args = parser.parse_args()

    symbols = args.symbols or [args.symbol]

    all_results = {}
    for sym in symbols:
        print(f"\n{'='*60}", flush=True)
        print(f"Evaluating {sym}", flush=True)
        print(f"{'='*60}", flush=True)

        if args.checkpoint:
            paths = [Path(args.checkpoint)]
        elif args.checkpoint_dir:
            ckpt_dir = Path(args.checkpoint_dir)
            if ckpt_dir.is_dir() and any(ckpt_dir.glob("epoch_*.pt")):
                paths = find_best_checkpoints(ckpt_dir, sym, args.top_k)
            else:
                # Search for this symbol's best run
                pattern = f"*{sym}*"
                run_dirs = sorted(ckpt_dir.glob(pattern))
                paths = []
                for rd in run_dirs:
                    if rd.is_dir():
                        paths.extend(find_best_checkpoints(rd, sym, 1))
                paths = paths[:args.top_k]
        else:
            ckpt_root = Path("sharpnessadjustedproximalpolicy/checkpoints")
            pattern = f"*{sym}*"
            run_dirs = sorted(ckpt_root.glob(pattern))
            paths = []
            for rd in run_dirs:
                if rd.is_dir():
                    paths.extend(find_best_checkpoints(rd, sym, 1))
            paths = paths[:args.top_k]

        if not paths:
            print(f"  No checkpoints found for {sym}", flush=True)
            continue

        sym_results = []
        for p in paths:
            if not p.exists():
                continue
            print(f"\n  {p.parent.name}/{p.name}...", flush=True)
            t0 = time.time()
            try:
                results = continuous_eval(
                    p, sym, lag=args.lag,
                    n_windows=args.n_windows,
                    window_hours=args.window_hours,
                )
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                continue
            wall = time.time() - t0

            if args.n_windows > 0 and len(results) > 1:
                summary = summarize_windows(results)
                print(
                    f"    ep{results[0].epoch}: "
                    f"med_ret={summary['median_return']:.2f}% "
                    f"pos={summary['positive_pct']:.0f}% "
                    f"med_sort={summary['median_sortino']:.2f} "
                    f"med_dd={summary['median_max_dd']:.2f}% "
                    f"trades={summary['median_trades']} "
                    f"({wall:.1f}s)",
                    flush=True,
                )
                sym_results.append({
                    "checkpoint": str(p),
                    "epoch": results[0].epoch,
                    "summary": summary,
                    "windows": [vars(r) for r in results],
                })
            else:
                r = results[0]
                print(
                    f"    ep{r.epoch}: "
                    f"ret={r.total_return_pct:.2f}% "
                    f"sort={r.sortino:.2f} "
                    f"dd={r.max_drawdown_pct:.2f}% "
                    f"trades={r.num_trades} "
                    f"wr={r.win_rate:.0%} "
                    f"({wall:.1f}s)",
                    flush=True,
                )
                sym_results.append(vars(r))

        all_results[sym] = sym_results

    # Save results
    out = Path("sharpnessadjustedproximalpolicy") / f"marketsim_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {out}", flush=True)

    # Summary table
    print(f"\n{'Symbol':<12} {'Checkpoint':<50} {'Return':>8} {'Sortino':>8} {'MaxDD':>8} {'Trades':>7}")
    print("-" * 100)
    for sym, sym_res in all_results.items():
        for r in sym_res:
            if "summary" in r:
                s = r["summary"]
                ckpt = Path(r["checkpoint"]).parent.name[:45]
                print(f"{sym:<12} {ckpt:<50} {s['median_return']:>7.2f}% {s['median_sortino']:>8.2f} {s['median_max_dd']:>7.2f}% {s['median_trades']:>7}")
            else:
                ckpt = Path(r["checkpoint"]).parent.name[:45]
                print(f"{sym:<12} {ckpt:<50} {r['total_return_pct']:>7.2f}% {r['sortino']:>8.2f} {r['max_drawdown_pct']:>7.2f}% {r['num_trades']:>7}")


if __name__ == "__main__":
    main()
