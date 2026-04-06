"""Sweep leverage 1x-5x on cached LLM signals. No API calls needed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_hourly_trader.backtest import RESULTS_DIR, load_bars, load_forecasts
from llm_hourly_trader.config import SYMBOL_UNIVERSE, BacktestConfig, SymbolConfig


def simulate_leveraged(
    bars_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    config: BacktestConfig,
    symbol_configs: dict[str, SymbolConfig],
    leverage: float = 1.0,
    margin_rate_annual: float = 0.10,
    margin_fee_override: float | None = 0.001,
) -> dict:
    """Simulate with leverage. Leverage > 1x = borrowed funds, incurs margin interest.
    margin_fee_override: if set, all symbols pay this fee rate (margin mode = 10bps)."""
    cash = config.initial_cash
    positions: dict[str, dict] = {}
    equity_history: list[float] = []
    trades: list[dict] = []

    def _fee(sym: str) -> float:
        if margin_fee_override is not None and leverage > 1.0 + 1e-9:
            return margin_fee_override
        return symbol_configs[sym].maker_fee if sym in symbol_configs else 0.001

    merged = bars_df.merge(actions_df, on=["timestamp", "symbol"], how="inner", suffixes=("", "_act"))
    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    hourly_margin_rate = margin_rate_annual / 8760

    for ts, chunk in merged.groupby("timestamp", sort=True):
        # Charge margin interest on borrowed portion
        for _sym, pos in list(positions.items()):
            if pos["qty"] <= 0:
                continue
            pos_value = pos["qty"] * pos["cost_basis"]
            borrowed = pos_value - pos.get("equity_used", pos_value)
            if borrowed > 0:
                interest = borrowed * hourly_margin_rate
                cash -= interest
                pos["total_interest"] = pos.get("total_interest", 0) + interest

        # Close at max hold
        for _, row in chunk.iterrows():
            sym = row["symbol"]
            pos = positions.get(sym)
            if pos is None or pos["qty"] <= 0:
                continue
            held_hours = (ts - pos["open_time"]).total_seconds() / 3600.0
            if held_hours >= config.max_hold_hours:
                close_price = float(row["close"])
                fee_rate = _fee(sym)
                pnl = (close_price - pos["cost_basis"]) * pos["qty"]
                fee = pos["qty"] * close_price * fee_rate
                interest = pos.get("total_interest", 0)
                cash += pos["equity_used"] + pnl - fee - interest
                trades.append(
                    {
                        "timestamp": str(ts),
                        "symbol": sym,
                        "side": "close",
                        "price": close_price,
                        "quantity": pos["qty"],
                        "realized_pnl": pnl,
                        "interest": interest,
                        "fee": fee,
                        "reason": "max_hold",
                    }
                )
                del positions[sym]

        # Take-profit exits
        for _, row in chunk.iterrows():
            sym = row["symbol"]
            pos = positions.get(sym)
            if pos is None or pos["qty"] <= 0:
                continue
            sell_price = float(row.get("sell_price", 0) or 0)
            if sell_price <= 0:
                continue
            high = float(row["high"])
            fee_rate = _fee(sym)
            if high >= sell_price:
                pnl = (sell_price - pos["cost_basis"]) * pos["qty"]
                fee = pos["qty"] * sell_price * fee_rate
                interest = pos.get("total_interest", 0)
                cash += pos["equity_used"] + pnl - fee - interest
                trades.append(
                    {
                        "timestamp": str(ts),
                        "symbol": sym,
                        "side": "sell",
                        "price": sell_price,
                        "quantity": pos["qty"],
                        "realized_pnl": pnl,
                        "interest": interest,
                        "fee": fee,
                        "reason": "take_profit",
                    }
                )
                del positions[sym]

        # Entries with leverage
        for _, row in chunk.iterrows():
            sym = row["symbol"]
            if sym in positions:
                continue
            direction = str(row.get("direction", "hold")).lower().strip()
            if direction != "long":
                continue
            sym_cfg = symbol_configs.get(sym)
            if sym_cfg is None:
                continue

            buy_price = float(row.get("buy_price", 0) or 0)
            confidence = float(row.get("confidence", 0) or 0)
            if buy_price <= 0 or confidence <= 0:
                continue
            low = float(row["low"])
            fee_rate = _fee(sym)

            if low <= buy_price:
                equity_alloc = cash * config.max_position_pct
                notional = equity_alloc * leverage
                qty = notional / (buy_price * (1 + fee_rate))
                if qty <= 0:
                    continue
                cost = qty * buy_price * (1 + fee_rate)
                equity_used = cost / leverage
                cash -= equity_used
                positions[sym] = {
                    "qty": qty,
                    "cost_basis": buy_price,
                    "open_time": ts,
                    "equity_used": equity_used,
                    "total_interest": 0,
                }
                trades.append(
                    {
                        "timestamp": str(ts),
                        "symbol": sym,
                        "side": "buy",
                        "price": buy_price,
                        "quantity": qty,
                        "realized_pnl": 0,
                        "interest": 0,
                        "fee": qty * buy_price * fee_rate,
                        "reason": "entry",
                    }
                )

        # Equity = cash + sum(equity_used + unrealized_pnl) per position
        unrealized = 0
        for s, p in positions.items():
            if s in chunk["symbol"].values and p["qty"] > 0:
                cur_price = float(chunk[chunk["symbol"] == s].iloc[0]["close"])
                pnl = (cur_price - p["cost_basis"]) * p["qty"]
                unrealized += p["equity_used"] + pnl - p.get("total_interest", 0)
        equity_history.append(cash + unrealized)

    # Metrics
    equity = np.array(equity_history, dtype=float)
    if len(equity) < 2:
        return None
    total_return = (equity[-1] - equity[0]) / equity[0]
    returns = np.diff(equity) / np.clip(equity[:-1], 1e-8, None)
    downside = returns[returns < 0]
    ds_std = downside.std() if len(downside) else 0
    sortino = returns.mean() / ds_std * np.sqrt(8760) if ds_std > 0 else 0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = abs(dd.min()) * 100

    entries = sum(1 for t in trades if t["side"] == "buy")
    total_pnl = sum(t.get("realized_pnl", 0) for t in trades)
    total_fees = sum(t.get("fee", 0) for t in trades)
    total_interest = sum(t.get("interest", 0) for t in trades)

    return {
        "total_return_pct": total_return * 100,
        "sortino": sortino,
        "max_drawdown_pct": max_dd,
        "final_equity": equity[-1],
        "entries": entries,
        "realized_pnl": total_pnl,
        "fees": total_fees,
        "interest": total_interest,
        "trades": trades,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD", "SOLUSD"])
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--leverages", nargs="+", type=float, default=[1, 2, 3, 4, 5])
    p.add_argument("--margin-rate", type=float, default=0.10, help="Annual margin interest rate")
    p.add_argument("--variant", default="optimization_long")
    p.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    args = p.parse_args()

    cache_tag = f"prompt-v3-{args.variant}-{args.model}"

    print(f"\n{'=' * 70}")
    print(f"LEVERAGE SWEEP: {args.leverages}")
    print(f"Symbols: {args.symbols}, Days: {args.days}")
    print(f"Variant: {args.variant}, Model: {args.model}")
    print(f"Margin rate: {args.margin_rate * 100:.1f}% annual")
    print(f"{'=' * 70}\n")

    # Load cached signals
    from datetime import timedelta

    import torch
    from llm_hourly_trader.cache import get_cached
    from llm_hourly_trader.gemini_wrapper import TradePlan

    # Load bars + forecasts
    ab, af1 = {}, {}
    sc = {}
    for s in args.symbols:
        ab[s] = load_bars(s)
        af1[s] = load_forecasts(s, "h1")
        sc[s] = SYMBOL_UNIVERSE.get(s, SymbolConfig(s, "crypto"))

    # Need to rebuild actions from cache. Reuse v3 logic to get cached signals.
    # Import v3 helpers
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prompt_experiment_v3 import (
        BINANCE6_SYMBOLS,
        MLPPolicy,
        TradingSignal,
        _read_forecast,
        _read_hourly_prices,
        build_prompt,
        compute_mktd_features,
        get_forecast_at,
    )

    # Load RL model for prompt building (needed for cache key match)
    ckpt_path = None
    for cp in [
        REPO / "rl-trainingbinance/checkpoints/binance6_ppo_v1_h1024_100M/best.pt",
        REPO / "rl-trainingbinance/checkpoints/autoresearch_ema.pt",
    ]:
        if cp.exists():
            ckpt_path = cp
            break

    if not ckpt_path:
        print("ERROR: No checkpoint found")
        return

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    ns = len(BINANCE6_SYMBOLS)

    # Check if this is autoresearch (TradingPolicy) or old (MLPPolicy)
    has_obs_norm = "obs_norm.running_mean" in sd
    if has_obs_norm:
        obs_size = sd["obs_norm.running_mean"].shape[0]
        hidden = sd["encoder.0.weight"].shape[0]
        na = sd["actor.2.weight"].shape[0]
        from rl_signal import TradingPolicy

        rl = TradingPolicy(obs_size, na, hidden)
    else:
        obs_size = ns * 16 + 5 + ns
        hidden = sd["encoder.0.weight"].shape[0]
        na = sd["actor.2.weight"].shape[0]
        rl = MLPPolicy(obs_size, na, hidden)
    rl.load_state_dict(sd)
    rl.eval()

    # Build feature arrays
    DR = REPO / "trainingdatahourly"
    FR = REPO / "binanceneural" / "forecast_cache"
    abf = {}
    for b in BINANCE6_SYMBOLS:
        try:
            pp = _read_hourly_prices(b, DR)
            f1 = _read_forecast(b, FR, 1) if (FR / f"{b}_h1.parquet").exists() else pd.DataFrame()
            f24 = _read_forecast(b, FR, 24) if (FR / f"{b}_h24.parquet").exists() else pd.DataFrame()
            abf[b] = compute_mktd_features(pp, f1, f24)
        except Exception:
            abf[b] = None

    fe = [af1[s]["timestamp"].max() for s in args.symbols if not af1[s].empty]
    end_ts = min(fe)
    start_ts = end_ts - timedelta(days=args.days)

    def gfa(ts):
        f = np.zeros((ns, 16), dtype=np.float32)
        tf = pd.Timestamp(ts).floor("h")
        for i, b in enumerate(BINANCE6_SYMBOLS):
            m = abf.get(b)
            if m is None:
                continue
            if tf in m.index:
                f[i] = m.loc[tf].values[:16].astype(np.float32)
            else:
                bef = m.index[m.index <= tf]
                if len(bef) > 0:
                    f[i] = m.iloc[m.index.get_loc(bef[-1])].values[:16].astype(np.float32)
        return f

    def grs(fa):
        if has_obs_norm:
            o = np.zeros(obs_size, dtype=np.float32)
            o[: ns * 16] = fa.flatten()
            o[ns * 16] = 1.0
            o[ns * 16 + 4] = 0.5
        else:
            o = np.zeros(obs_size, dtype=np.float32)
            o[: ns * 16] = fa.flatten()
            o[ns * 16] = 1.0
            o[ns * 16 + 4] = 0.5
        ot = torch.from_numpy(o).unsqueeze(0)
        with torch.no_grad():
            lg, v = rl(ot)
            pr = torch.softmax(lg, -1)
            a = lg.argmax(-1).item()
            cf = pr[0, a].item()
            vl = v.item()
        if a == 0:
            return TradingSignal("flat", None, None, cf, vl, 0, 0)
        ai = a - 1
        sh = ai >= ns
        if sh:
            ai -= ns
        sy = BINANCE6_SYMBOLS[ai] if ai < ns else "?"
        return TradingSignal(f"{'short' if sh else 'long'}_{sy}", sy, "short" if sh else "long", cf, vl, 1, 0)

    # Build actions from cached prompts
    print("Building actions from cached LLM signals...")
    ar, br = [], []
    ps = dict.fromkeys(args.symbols)
    po = dict.fromkeys(args.symbols, "N/A")
    hits, misses = 0, 0
    for s in args.symbols:
        sb = ab[s]
        w = sb[(sb["timestamp"] >= start_ts) & (sb["timestamp"] <= end_ts)].copy()
        af24_s = load_forecasts(s, "h24")
        for i, (_, bar) in enumerate(w.iterrows()):
            ts_bar = bar["timestamp"]
            h = sb[sb["timestamp"] <= ts_bar].tail(72)
            if len(h) < 5:
                ar.append(
                    {
                        "timestamp": ts_bar,
                        "symbol": s,
                        "buy_price": 0,
                        "sell_price": 0,
                        "direction": "hold",
                        "confidence": 0,
                    }
                )
                br.append(bar.to_dict())
                continue
            f1 = get_forecast_at(af1[s], ts_bar)
            f24 = get_forecast_at(af24_s, ts_bar)
            fa = gfa(ts_bar)
            rs = grs(fa)
            prompt = build_prompt(args.variant, s, h.tail(72).to_dict("records"), f1, f24, rs, ps[s], po[s])
            c = get_cached(cache_tag, prompt)
            if c:
                plan = TradePlan(**c)
                hits += 1
            else:
                plan = TradePlan("hold", 0, 0, 0, "no cache")
                misses += 1
            if plan.direction == "short":
                plan = TradePlan("hold", 0, 0, 0, "short filtered")
            ar.append(
                {
                    "timestamp": ts_bar,
                    "symbol": s,
                    "buy_price": plan.buy_price,
                    "sell_price": plan.sell_price,
                    "direction": plan.direction,
                    "confidence": plan.confidence,
                }
            )
            br.append(bar.to_dict())
            ps[s] = rs
            if i > 0:
                pc = float(w.iloc[i - 1]["close"])
                cc = float(bar["close"])
                po[s] = f"{(cc - pc) / pc * 100:+.2f}% (${pc:.2f}->${cc:.2f})"

    print(f"  Cache: {hits} hits, {misses} misses")
    if misses > hits * 0.5:
        print(f"  WARNING: {misses} cache misses - results may differ from original experiment")

    bd = pd.DataFrame(br)
    ad = pd.DataFrame(ar)
    bd["timestamp"] = pd.to_datetime(bd["timestamp"], utc=True)
    ad["timestamp"] = pd.to_datetime(ad["timestamp"], utc=True)

    cfg_base = BacktestConfig(initial_cash=10_000.0, max_hold_hours=6, max_position_pct=0.25, model=cache_tag)

    # Sweep
    results = {}
    print(
        f"\n{'Leverage':<10} {'Return':>10} {'Sortino':>10} {'MaxDD':>10} {'Trades':>8} {'PnL':>12} {'Fees':>10} {'Interest':>10}"
    )
    print("-" * 90)
    for lev in args.leverages:
        r = simulate_leveraged(bd, ad, cfg_base, sc, leverage=lev, margin_rate_annual=args.margin_rate)
        if r is None:
            print(f"{lev:.0f}x        (no data)")
            continue
        label = f"{lev:.0f}x"
        results[label] = r
        print(
            f"{label:<10} {r['total_return_pct']:>+9.2f}% {r['sortino']:>10.2f} {r['max_drawdown_pct']:>9.2f}% {r['entries']:>8d} ${r['realized_pnl']:>+10.2f} ${r['fees']:>9.2f} ${r['interest']:>9.2f}"
        )

    # Per-pair breakdown for best
    if results:
        best_key = max(results, key=lambda k: results[k]["sortino"])
        best = results[best_key]
        print(f"\nBest by Sortino: {best_key}")

        from prompt_experiment_v3 import per_pair_pnl

        pp = per_pair_pnl(best["trades"])
        print("  Per-pair:")
        for sym in sorted(pp.keys()):
            ps_data = pp[sym]
            wr = ps_data["wins"] / max(1, ps_data["wins"] + ps_data["losses"]) * 100
            print(
                f"    {sym:>8}: PnL=${ps_data['pnl']:+8.2f}  fees=${ps_data['fees']:6.2f}  entries={ps_data['entries']:3d}  W/L={ps_data['wins']}/{ps_data['losses']} ({wr:.0f}%)"
            )

    # Save
    out = RESULTS_DIR / "leverage_sweep.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_data = {k: {kk: vv for kk, vv in v.items() if kk != "trades"} for k, v in results.items()}
    with open(out, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
