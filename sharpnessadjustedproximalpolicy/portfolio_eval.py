#!/usr/bin/env python3
"""Portfolio-level marketsim evaluation for SAPP per-symbol models.

Loads best per-symbol checkpoints, runs chunk-based inference on each,
combines into a diversified portfolio, evaluates with binary fills.

Usage:
    python -m sharpnessadjustedproximalpolicy.portfolio_eval
    python -m sharpnessadjustedproximalpolicy.portfolio_eval --alloc inverse_vol --n-windows 20
    python -m sharpnessadjustedproximalpolicy.portfolio_eval --warp  # weight-averaged checkpoints
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.model import PolicyConfig, build_policy
from differentiable_loss_utils import simulate_hourly_trades_binary
from src.torch_load_utils import torch_load_compat


EXCLUDE_SYMBOLS = {"AVAXUSD", "AAVEUSD"}  # negative or insufficient data
# Skip FDUSD duplicates -- keep the USD pair
FDUSD_DUPES = {"BTCFDUSD", "ETHFDUSD"}


@dataclass
class SymbolResult:
    symbol: str
    checkpoint: str
    epoch: int
    total_hours: int
    total_return_pct: float
    sortino: float
    max_drawdown_pct: float
    num_trades: int
    win_rate: float
    portfolio_values: np.ndarray  # raw PV curve for portfolio combination


@dataclass
class PortfolioResult:
    alloc_method: str
    symbols: list[str]
    n_windows: int
    window_hours: int
    per_window: list[dict]
    median_return: float
    mean_return: float
    p10_return: float
    p05_return: float
    worst_return: float
    positive_pct: float
    median_sortino: float
    mean_sortino: float
    median_dd: float
    worst_dd: float
    median_trades: int


def load_model_from_ckpt(ckpt_path: Path):
    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    state_dict = ckpt.get("state_dict", {})
    feature_cols = ckpt.get("feature_columns", [])

    input_dim = 0
    for key in ("embed.weight", "_orig_mod.embed.weight"):
        if key in state_dict:
            input_dim = state_dict[key].shape[1]
            break

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
        moe_num_experts=config.get("moe_num_experts", 0),
    )
    model = build_policy(pc)
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, config, feature_cols, horizons


def warp_checkpoints(ckpt_paths: list[Path], top_k: int = 3) -> dict:
    """WARP: Weight Averaged Rewarded Policies. Average top-K checkpoint state dicts."""
    if len(ckpt_paths) <= 1:
        ckpt = torch_load_compat(ckpt_paths[0], map_location="cpu", weights_only=False)
        return ckpt

    # Load all, sort by val score, average top-K
    loaded = []
    for p in ckpt_paths:
        c = torch_load_compat(p, map_location="cpu", weights_only=False)
        score = c.get("metrics", {}).get("score", c.get("metrics", {}).get("sortino", -999))
        loaded.append((score, c))
    loaded.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in loaded[:top_k]]

    # Average state dicts
    avg_sd = {}
    ref_sd = top[0]["state_dict"]
    for key in ref_sd:
        tensors = [c["state_dict"][key].float() for c in top if key in c["state_dict"]]
        if tensors:
            avg_sd[key] = (sum(tensors) / len(tensors)).to(ref_sd[key].dtype)

    result = dict(top[0])
    result["state_dict"] = avg_sd
    return result


def run_symbol_inference(
    ckpt_path: Path,
    symbol: str,
    device: torch.device,
    warp_paths: Optional[list[Path]] = None,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Run chunk-based inference for a single symbol. Returns actions and OHLC arrays."""
    if warp_paths and len(warp_paths) > 1:
        ckpt = warp_checkpoints(warp_paths, top_k=min(3, len(warp_paths)))
        config = ckpt.get("config", {})
        state_dict = ckpt.get("state_dict", {})
        feature_cols = ckpt.get("feature_columns", [])
        input_dim = 0
        for key in ("embed.weight", "_orig_mod.embed.weight"):
            if key in state_dict:
                input_dim = state_dict[key].shape[1]
                break
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
            moe_num_experts=config.get("moe_num_experts", 0),
        )
        model = build_policy(pc)
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(sd, strict=False)
        model.eval()
    else:
        model, config, feature_cols, horizons = load_model_from_ckpt(ckpt_path)

    seq_len = config.get("sequence_length", 72)
    scale = config.get("trade_amount_scale", 100.0)

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

    model = model.to(device)
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
            all_bp.append(actions["buy_price"].squeeze(0).cpu())
            all_sp.append(actions["sell_price"].squeeze(0).cpu())
            all_ti.append((actions["trade_amount"].squeeze(0) / scale).cpu())
            all_bi.append((actions["buy_amount"].squeeze(0) / scale).cpu())
            all_si.append((actions["sell_amount"].squeeze(0) / scale).cpu())

    bp = torch.cat(all_bp).numpy()
    sp = torch.cat(all_sp).numpy()
    ti = torch.cat(all_ti).numpy()
    bi = torch.cat(all_bi).numpy()
    si = torch.cat(all_si).numpy()
    total_bars = len(bp)

    highs = vds.highs[:total_bars]
    lows = vds.lows[:total_bars]
    closes = vds.closes[:total_bars]
    opens = vds.opens[:total_bars]

    del model
    torch.cuda.empty_cache()

    return config, highs, lows, closes, opens, bp, sp, ti, bi, si, total_bars


def run_sim_window(
    highs, lows, closes, opens, bp, sp, ti, bi, si,
    s: int, e: int, config: dict, device: torch.device,
    leverage_override: float = 0.0,
    fee_override: float = -1.0,
    can_short: bool = False,
) -> tuple[np.ndarray, float, float, float, int, float]:
    """Run binary-fill sim on a window, return PV curve and metrics."""
    fee = fee_override if fee_override >= 0 else config.get("maker_fee", 0.001)
    max_lev = leverage_override if leverage_override > 0 else config.get("max_leverage", 1.0)
    fill_buf = config.get("fill_buffer_pct", 0.0005)
    margin_rate = config.get("margin_annual_rate", 0.0625)
    lag = 2

    sim = simulate_hourly_trades_binary(
        highs=torch.from_numpy(highs[s:e]).unsqueeze(0).to(device),
        lows=torch.from_numpy(lows[s:e]).unsqueeze(0).to(device),
        closes=torch.from_numpy(closes[s:e]).unsqueeze(0).to(device),
        opens=torch.from_numpy(opens[s:e]).unsqueeze(0).to(device),
        buy_prices=torch.from_numpy(bp[s:e]).unsqueeze(0).to(device),
        sell_prices=torch.from_numpy(sp[s:e]).unsqueeze(0).to(device),
        trade_intensity=torch.from_numpy(ti[s:e]).unsqueeze(0).to(device),
        buy_trade_intensity=torch.from_numpy(bi[s:e]).unsqueeze(0).to(device),
        sell_trade_intensity=torch.from_numpy(si[s:e]).unsqueeze(0).to(device),
        maker_fee=fee, initial_cash=1.0, can_short=can_short, can_long=True,
        max_leverage=max_lev, fill_buffer_pct=fill_buf,
        margin_annual_rate=margin_rate, decision_lag_bars=lag,
    )
    pv = sim.portfolio_values.float().cpu().squeeze().numpy()
    rets = sim.returns.float().cpu().squeeze().numpy()
    inv = sim.inventory_path.float().cpu().squeeze().numpy()

    total_ret = (pv[-1] / pv[0] - 1) * 100
    mean_r = rets.mean()
    down_std = np.sqrt((np.minimum(rets, 0) ** 2).mean() + 1e-10)
    n = e - s
    ann = 8760.0 / max(n, 1)
    sortino = (mean_r / max(down_std, 1e-10)) * (ann ** 0.5)
    cummax = np.maximum.accumulate(pv)
    max_dd = ((pv - cummax) / np.maximum(cummax, 1e-10)).min() * 100

    # Count trades
    n_trades = 0
    was_flat = True
    for t_i in range(len(inv)):
        is_flat = abs(inv[t_i]) < 1e-8
        if was_flat and not is_flat:
            pass
        elif not was_flat and is_flat:
            n_trades += 1
        was_flat = is_flat

    return pv / pv[0], total_ret, sortino, max_dd, n_trades, float(np.mean(np.abs(inv) > 1e-8) * 100)


def combine_portfolio(
    pv_curves: dict[str, np.ndarray],
    method: str = "equal",
    vol_window: int = 168,
) -> np.ndarray:
    """Combine per-symbol normalized PV curves into portfolio PV.

    Methods:
        equal: equal weight allocation
        inverse_vol: inverse realized volatility weighting
        sqrt_sortino: weight by sqrt of per-symbol sortino
    """
    symbols = list(pv_curves.keys())
    n = len(symbols)
    min_len = min(len(v) for v in pv_curves.values())
    aligned = np.stack([pv_curves[s][:min_len] for s in symbols])  # (n_sym, T)

    if method == "equal":
        weights = np.ones(n) / n
    elif method == "inverse_vol":
        # Use rolling volatility of returns
        rets = np.diff(aligned, axis=1) / np.maximum(aligned[:, :-1], 1e-10)
        window = min(vol_window, rets.shape[1])
        vols = np.array([np.std(rets[i, -window:]) for i in range(n)])
        inv_vol = 1.0 / np.maximum(vols, 1e-10)
        weights = inv_vol / inv_vol.sum()
    elif method == "sqrt_sortino":
        # weight by sqrt(sortino) where sortino > 0
        rets = np.diff(aligned, axis=1) / np.maximum(aligned[:, :-1], 1e-10)
        mean_rets = rets.mean(axis=1)
        down_std = np.sqrt((np.minimum(rets, 0) ** 2).mean(axis=1) + 1e-10)
        sortinos = mean_rets / np.maximum(down_std, 1e-10)
        pos_sort = np.maximum(sortinos, 0.01)
        w = np.sqrt(pos_sort)
        weights = w / w.sum()
    else:
        weights = np.ones(n) / n

    # Portfolio PV = weighted sum of per-symbol PV curves (each starts at 1.0)
    portfolio_pv = np.zeros(min_len)
    for i, s in enumerate(symbols):
        portfolio_pv += weights[i] * aligned[i]

    return portfolio_pv


def compute_portfolio_metrics(pv: np.ndarray) -> dict:
    """Compute standard metrics from a portfolio PV curve."""
    rets = np.diff(pv) / np.maximum(pv[:-1], 1e-10)
    total_ret = (pv[-1] / pv[0] - 1) * 100
    mean_r = rets.mean()
    down_std = np.sqrt((np.minimum(rets, 0) ** 2).mean() + 1e-10)
    total_std = np.sqrt(rets.var() + 1e-10)
    n = len(pv)
    ann = 8760.0 / max(n, 1)
    sortino = (mean_r / max(down_std, 1e-10)) * (ann ** 0.5)
    sharpe = (mean_r / max(total_std, 1e-10)) * (ann ** 0.5)
    cummax = np.maximum.accumulate(pv)
    max_dd = ((pv - cummax) / np.maximum(cummax, 1e-10)).min() * 100
    return {
        "total_return_pct": round(total_ret, 3),
        "sortino": round(sortino, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd, 3),
        "hours": n,
    }


def get_best_checkpoints(leaderboard_path: Path) -> dict[str, dict]:
    """Parse leaderboard CSV to get best checkpoint per symbol."""
    import csv
    best = {}
    with open(leaderboard_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row["symbol"]
            if sym in EXCLUDE_SYMBOLS or sym in FDUSD_DUPES:
                continue
            if row.get("error"):
                continue
            try:
                sort = float(row["val_sortino"])
            except (ValueError, KeyError):
                continue
            if sort <= 0:
                continue
            if sym not in best or sort > best[sym]["sortino"]:
                best[sym] = {
                    "sortino": sort,
                    "config": row["config"],
                    "epoch": int(row["best_epoch"]),
                    "val_return": float(row.get("val_return", 0)),
                }
    return best


def find_checkpoint_path(symbol: str, config_name: str, epoch: int) -> Optional[Path]:
    """Find the checkpoint directory for a symbol/config combo."""
    ckpt_root = Path("sharpnessadjustedproximalpolicy/checkpoints")
    candidates = sorted(ckpt_root.glob(f"sap_{config_name}_{symbol}_*"))
    for d in reversed(candidates):  # newest first
        ckpt = d / f"epoch_{epoch:03d}.pt"
        if ckpt.exists():
            return ckpt
    return None


def find_warp_paths(symbol: str, ckpt_root: Path) -> list[Path]:
    """Find all checkpoints for a symbol across configs for WARP averaging."""
    paths = []
    for d in sorted(ckpt_root.glob(f"*_{symbol}_*")):
        if not d.is_dir():
            continue
        history_path = d / "sap_history.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text())
        if not history:
            continue
        best_ep = max(history, key=lambda h: h.get("val_sortino", -999))
        ep = best_ep["epoch"]
        ckpt = d / f"epoch_{ep:03d}.pt"
        if ckpt.exists() and best_ep.get("val_sortino", -999) > 0:
            paths.append(ckpt)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alloc", default="equal", choices=["equal", "inverse_vol", "sqrt_sortino"])
    parser.add_argument("--n-windows", type=int, default=10)
    parser.add_argument("--window-hours", type=int, default=720)
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--warp", action="store_true", help="Use WARP weight averaging")
    parser.add_argument("--warp-k", type=int, default=3)
    parser.add_argument("--min-sortino", type=float, default=50.0, help="Min val sortino to include symbol")
    parser.add_argument("--leaderboard", default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    # Risk tuning
    parser.add_argument("--leverage", type=float, nargs="+", default=[1.0], help="Leverage levels to sweep")
    parser.add_argument("--fee", type=float, default=-1.0, help="Override fee rate (-1 = use checkpoint)")
    parser.add_argument("--can-short", action="store_true", help="Allow short positions")
    parser.add_argument("--top-n", type=int, default=0, help="Only use top N symbols by sortino")
    args = parser.parse_args()

    # Find best leaderboard
    sap_dir = Path("sharpnessadjustedproximalpolicy")
    if args.leaderboard:
        lb_path = Path(args.leaderboard)
    else:
        lbs = sorted(sap_dir.glob("scaled_leaderboard_*.csv"))
        if not lbs:
            lbs = sorted(sap_dir.glob("allpairs_leaderboard_*.csv"))
        lb_path = lbs[-1]

    print(f"Leaderboard: {lb_path.name}", flush=True)
    best_ckpts = get_best_checkpoints(lb_path)

    # Filter by min sortino
    if args.symbols:
        symbols = [s for s in args.symbols if s in best_ckpts]
    else:
        symbols = [s for s, v in best_ckpts.items() if v["sortino"] >= args.min_sortino]
    symbols.sort(key=lambda s: best_ckpts[s]["sortino"], reverse=True)

    print(f"\nPortfolio ({len(symbols)} symbols):", flush=True)
    for s in symbols:
        v = best_ckpts[s]
        print(f"  {s:<10} sort={v['sortino']:>7.1f} ret={v['val_return']:>6.3f} cfg={v['config']} ep={v['epoch']}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_root = Path("sharpnessadjustedproximalpolicy/checkpoints")

    # Run inference for each symbol
    sym_data = {}
    for sym in symbols:
        v = best_ckpts[sym]
        ckpt_path = find_checkpoint_path(sym, v["config"], v["epoch"])
        if not ckpt_path:
            print(f"  SKIP {sym}: no checkpoint found", flush=True)
            continue

        warp_paths = None
        if args.warp:
            warp_paths = find_warp_paths(sym, ckpt_root)
            if len(warp_paths) > 1:
                print(f"  WARP {sym}: averaging {len(warp_paths)} checkpoints (top-{args.warp_k})", flush=True)

        print(f"  Loading {sym}...", end="", flush=True)
        t0 = time.time()
        try:
            config, highs, lows, closes, opens, bp, sp, ti, bi, si, total_bars = run_symbol_inference(
                ckpt_path, sym, device, warp_paths=warp_paths,
            )
            sym_data[sym] = {
                "config": config,
                "highs": highs, "lows": lows, "closes": closes, "opens": opens,
                "bp": bp, "sp": sp, "ti": ti, "bi": bi, "si": si,
                "total_bars": total_bars,
            }
            print(f" {total_bars} bars ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            continue

    if not sym_data:
        print("No symbols loaded")
        return

    # Find common window length (min across all symbols)
    min_bars = min(d["total_bars"] for d in sym_data.values())
    print(f"\nCommon bars: {min_bars} ({min_bars/24:.0f}d)", flush=True)

    # Generate windows
    window_h = args.window_hours
    if min_bars <= window_h:
        windows = [(0, min_bars)]
    else:
        n_w = args.n_windows
        stride = max(1, (min_bars - window_h) // max(n_w - 1, 1))
        windows = []
        for i in range(n_w):
            s = i * stride
            e = min(s + window_h, min_bars)
            if s >= min_bars:
                break
            windows.append((s, e))

    print(f"Windows: {len(windows)} x {window_h}h", flush=True)

    # Run per-symbol sims and combine into portfolio for each window
    alloc_methods = [args.alloc] if args.alloc != "equal" else ["equal", "inverse_vol", "sqrt_sortino"]
    leverage_levels = args.leverage

    all_results = {}
    for lev in leverage_levels:
        for method in alloc_methods:
            label = f"{method}_lev{lev:.0f}x" if lev != 1.0 else method
            window_results = []
            for wi, (s, e) in enumerate(windows):
                pv_curves = {}
                sym_metrics = {}
                for sym, data in sym_data.items():
                    pv_norm, ret, sort, dd, trades, tip = run_sim_window(
                        data["highs"], data["lows"], data["closes"], data["opens"],
                        data["bp"], data["sp"], data["ti"], data["bi"], data["si"],
                        s, e, data["config"], device,
                        leverage_override=lev,
                        fee_override=args.fee,
                        can_short=args.can_short,
                    )
                    pv_curves[sym] = pv_norm
                    sym_metrics[sym] = {"return": round(ret, 2), "sortino": round(sort, 2), "dd": round(dd, 2), "trades": trades}

                # Combine into portfolio
                portfolio_pv = combine_portfolio(pv_curves, method=method)
                pm = compute_portfolio_metrics(portfolio_pv)
            pm["per_symbol"] = sym_metrics
            pm["window"] = (s, e)
            window_results.append(pm)

            # Aggregate
            rets = [w["total_return_pct"] for w in window_results]
            sorts = [w["sortino"] for w in window_results]
            dds = [w["max_drawdown_pct"] for w in window_results]

            result = PortfolioResult(
                alloc_method=label,
                symbols=list(sym_data.keys()),
                n_windows=len(windows),
                window_hours=window_h,
                per_window=window_results,
                median_return=round(float(np.median(rets)), 2),
                mean_return=round(float(np.mean(rets)), 2),
                p10_return=round(float(np.percentile(rets, 10)), 2),
                p05_return=round(float(np.percentile(rets, 5)), 2),
                worst_return=round(float(min(rets)), 2),
                positive_pct=round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1),
                median_sortino=round(float(np.median(sorts)), 2),
                mean_sortino=round(float(np.mean(sorts)), 2),
                median_dd=round(float(np.median(dds)), 2),
                worst_dd=round(float(min(dds)), 2),
                median_trades=int(np.median([sum(w["per_symbol"][s]["trades"] for s in w["per_symbol"]) for w in window_results])),
            )
            all_results[label] = result

            print(f"\n{'='*70}", flush=True)
            lev_str = f" @ {lev:.0f}x" if lev != 1.0 else ""
            short_str = " +SHORT" if args.can_short else ""
            print(f"PORTFOLIO [{method.upper()}{lev_str}{short_str}] - {len(sym_data)} symbols, {len(windows)} windows x {window_h}h", flush=True)
            print(f"{'='*70}", flush=True)
            print(f"  Median return: {result.median_return:>7.2f}%", flush=True)
            print(f"  Mean return:   {result.mean_return:>7.2f}%", flush=True)
            print(f"  P10 return:    {result.p10_return:>7.2f}%", flush=True)
            print(f"  Worst return:  {result.worst_return:>7.2f}%", flush=True)
            print(f"  Positive:      {result.positive_pct:>7.1f}%", flush=True)
            print(f"  Median Sortino:{result.median_sortino:>7.2f}", flush=True)
            print(f"  Mean Sortino:  {result.mean_sortino:>7.2f}", flush=True)
            print(f"  Median DD:     {result.median_dd:>7.2f}%", flush=True)
            print(f"  Worst DD:      {result.worst_dd:>7.2f}%", flush=True)
            print(f"  Median trades: {result.median_trades:>7d}", flush=True)

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    warp_str = "_warp" if args.warp else ""
    out_path = sap_dir / f"portfolio_eval_{ts}{warp_str}.json"
    serializable = {}
    for method, r in all_results.items():
        serializable[method] = {
            "alloc_method": r.alloc_method,
            "symbols": r.symbols,
            "n_windows": r.n_windows,
            "window_hours": r.window_hours,
            "median_return": r.median_return,
            "mean_return": r.mean_return,
            "p10_return": r.p10_return,
            "p05_return": r.p05_return,
            "worst_return": r.worst_return,
            "positive_pct": r.positive_pct,
            "median_sortino": r.median_sortino,
            "mean_sortino": r.mean_sortino,
            "median_dd": r.median_dd,
            "worst_dd": r.worst_dd,
            "median_trades": r.median_trades,
            "per_window": r.per_window,
        }
    out_path.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"\nSaved: {out_path}", flush=True)

    # Benchmark comparison
    print(f"\n{'='*70}", flush=True)
    print("BENCHMARK COMPARISON", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Method':<25} {'Med Ret':>8} {'Mean Ret':>9} {'P10':>8} {'Med Sort':>9} {'Med DD':>8}", flush=True)
    print("-" * 70, flush=True)
    for method, r in all_results.items():
        print(f"SAPP {method:<20} {r.median_return:>7.2f}% {r.mean_return:>8.2f}% {r.p10_return:>7.2f}% {r.median_sortino:>9.2f} {r.median_dd:>7.2f}%", flush=True)
    print(f"Worksteal robust         +3.95% avg   +3.06% worst   Sort=0.85   DD=-4.75%", flush=True)
    print(f"Pufferlib c15_s78       +141.4% med                  Sort=4.52", flush=True)


if __name__ == "__main__":
    main()
