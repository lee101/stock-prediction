"""Backtest RL vs Gemini vs RL+Gemini with REAL RL model inference.

Instead of SMA proxy, loads actual RL checkpoint and runs inference on
observations built from CSV price data. For Gemini modes, makes real API calls
with Chronos2 forecasts.

Runs through HourlyTrader simulator for realistic fill modeling.

Usage:
  python -m unified_orchestrator.backtest_rl_gemini_real \
      --checkpoint pufferlib_market/checkpoints/autoresearch/slip_5bps/best.pt \
      --symbols BTCUSD ETHUSD SOLUSD --days 7
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from newnanoalpacahourlyexp.marketsimulator.hourly_trader import (
    HourlyTraderMarketSimulator,
    HourlyTraderSimulationConfig,
)
from unified_orchestrator.rl_gemini_bridge import (
    RLSignal,
    build_hybrid_prompt,
    _softmax,
)

DATA_DIR = REPO / "trainingdatahourly" / "crypto"
FORECAST_DIR = REPO / "binanceneural" / "forecast_cache"


# ── Data loading ─────────────────────────────────────────────────────

def load_bars(symbol: str) -> pd.DataFrame:
    csv = DATA_DIR / f"{symbol}.csv"
    if not csv.exists():
        csv = REPO / f"binance_spot_hourly/{symbol.replace('USD', 'USDT')}.csv"
    df = pd.read_csv(csv)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("timestamp").dropna(subset=["close"]).reset_index(drop=True)


def load_forecast(symbol: str, horizon: str) -> pd.DataFrame:
    path = FORECAST_DIR / horizon / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_forecast_at(fc_df: pd.DataFrame, ts) -> dict | None:
    if fc_df.empty:
        return None
    match = fc_df[fc_df["timestamp"] <= ts]
    if match.empty:
        return None
    row = match.iloc[-1]
    return {
        "predicted_close_p50": float(row.get("predicted_close_p50", 0)),
        "predicted_close_p10": float(row.get("predicted_close_p10", 0)),
        "predicted_close_p90": float(row.get("predicted_close_p90", 0)),
        "predicted_high_p50": float(row.get("predicted_high_p50", 0)),
        "predicted_low_p50": float(row.get("predicted_low_p50", 0)),
    }


# ── RL observation builder (matching C env format) ───────────────────

def build_obs_from_bars(
    bars_df: pd.DataFrame,
    idx: int,
    num_symbols: int,
    features_per_sym: int = 16,
    lookback: int = 1,
) -> np.ndarray:
    """Build an observation vector matching the C env format from bar data.

    The C env observation is: [features(S*F)] + [portfolio(5+S)]
    Features per symbol (16): normalized OHLCV + technical indicators
    Portfolio: [cash_ratio, equity_ratio, unrealized_pnl, num_positions, step_ratio] + [position_per_sym]
    """
    obs = np.zeros(num_symbols * features_per_sym + 5 + num_symbols, dtype=np.float32)

    # Use t-1 observation (causal lag)
    t_obs = max(0, idx - lookback)

    if t_obs < 24:
        return obs  # Not enough history for features

    # Build features for each "symbol" (we pack all into symbol 0 for single-symbol mode,
    # or distribute across symbols)
    row = bars_df.iloc[t_obs]
    close = row["close"]
    if close == 0 or np.isnan(close):
        return obs

    # Simple normalized features matching what the C env computes
    # The actual C env uses: returns, log-returns, volatility, RSI, etc.
    hist = bars_df.iloc[max(0, t_obs - 24):t_obs + 1]
    closes = hist["close"].values

    if len(closes) < 2:
        return obs

    # Feature 0-3: normalized OHLC (relative to close)
    obs[0] = row["open"] / close - 1.0
    obs[1] = row["high"] / close - 1.0
    obs[2] = row["low"] / close - 1.0
    obs[3] = 0.0  # close/close - 1 = 0

    # Feature 4: volume (normalized, dummy)
    obs[4] = 0.0

    # Feature 5-8: returns at different horizons
    if len(closes) >= 2:
        obs[5] = closes[-1] / closes[-2] - 1.0
    if len(closes) >= 4:
        obs[6] = closes[-1] / closes[-4] - 1.0
    if len(closes) >= 12:
        obs[7] = closes[-1] / closes[-12] - 1.0
    if len(closes) >= 24:
        obs[8] = closes[-1] / closes[-24] - 1.0

    # Feature 9-10: volatility
    if len(closes) >= 12:
        rets = np.diff(np.log(closes[-12:]))
        obs[9] = np.std(rets) if len(rets) > 1 else 0
    if len(closes) >= 24:
        rets = np.diff(np.log(closes[-24:]))
        obs[10] = np.std(rets) if len(rets) > 1 else 0

    # Feature 11-13: SMA ratios
    if len(closes) >= 12:
        obs[11] = close / closes[-12:].mean() - 1.0
    if len(closes) >= 24:
        obs[12] = close / closes[-24:].mean() - 1.0

    # Feature 13-15: high/low range
    if len(hist) >= 12:
        obs[13] = (hist["high"].iloc[-12:].max() - close) / close
        obs[14] = (close - hist["low"].iloc[-12:].min()) / close
    obs[15] = (row["high"] - row["low"]) / close if close > 0 else 0

    # Portfolio state: start with full cash
    base_idx = num_symbols * features_per_sym
    obs[base_idx] = 1.0      # cash_ratio
    obs[base_idx + 1] = 1.0  # equity_ratio
    obs[base_idx + 2] = 0.0  # unrealized_pnl
    obs[base_idx + 3] = 0.0  # num_positions
    obs[base_idx + 4] = float(idx) / max(1, len(bars_df))  # step_ratio

    return obs


# ── RL policy loader ────────────────────────────────────────────────

def load_rl_policy(checkpoint_path: str, obs_size: int, num_actions: int,
                   hidden_size: int = 1024, arch: str = "mlp",
                   device: str = "cpu"):
    """Load RL policy from checkpoint."""
    from pufferlib_market.evaluate import TradingPolicy, ResidualTradingPolicy

    if arch == "resmlp":
        policy = ResidualTradingPolicy(obs_size, num_actions, hidden=hidden_size)
    else:
        policy = TradingPolicy(obs_size, num_actions, hidden=hidden_size)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.to(device)
    policy.eval()
    return policy


def get_rl_signal_from_obs(policy, obs: np.ndarray, num_symbols: int,
                           device: str = "cpu") -> tuple[str, float]:
    """Run RL policy on observation and return (direction, confidence)."""
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        logits, value = policy(obs_t)
        logits_np = logits.squeeze(0).cpu().numpy()

    probs = _softmax(logits_np)
    action = int(logits_np.argmax())

    flat_prob = probs[0]

    if action == 0:
        return "flat", float(flat_prob)

    # Determine which symbol and direction
    per_sym = 1  # assuming 1 alloc bin, 1 level bin
    side_block = num_symbols * per_sym

    if action <= side_block:
        # Long action
        direction = "long"
        sym_idx = (action - 1) // per_sym
        prob = float(probs[action])
    else:
        # Short action
        direction = "short"
        sym_idx = (action - 1 - side_block) // per_sym
        prob = float(probs[action])

    confidence = prob / (prob + flat_prob + 1e-8)
    return direction, confidence


# ── Main backtest ───────────────────────────────────────────────────

def run_backtest(
    symbols: list[str],
    checkpoint_path: str = "",
    days: int = 7,
    initial_cash: float = 10_000.0,
    modes: list[str] | None = None,
    model: str = "gemini-2.5-flash",
    thinking_level: str = "HIGH",
    hidden_size: int = 1024,
    arch: str = "mlp",
    gemini_sample_rate: int = 4,  # Only call Gemini every N bars to save cost
) -> dict:
    """Run backtest comparing modes.

    Modes:
      - rl_only: RL model signal → fixed spread prices
      - gemini_only: Gemini prompt with Chronos2 forecasts (sampled)
      - rl_gemini: RL signal direction + Gemini price refinement (sampled)
    """
    if modes is None:
        modes = ["rl_only", "gemini_only", "rl_gemini"]

    # Load data
    print("Loading data...")
    all_bars = {}
    all_fc_h1 = {}
    all_fc_h24 = {}
    for sym in symbols:
        try:
            all_bars[sym] = load_bars(sym)
            all_fc_h1[sym] = load_forecast(sym, "h1")
            all_fc_h24[sym] = load_forecast(sym, "h24")
            print(f"  {sym}: {len(all_bars[sym])} bars, "
                  f"fc_h1={len(all_fc_h1[sym])}, fc_h24={len(all_fc_h24[sym])}")
        except Exception as e:
            print(f"  {sym}: SKIP ({e})")

    usable = [s for s in symbols if s in all_bars]
    if not usable:
        print("No usable symbols")
        return {}

    # Determine time window - use last N days
    end_ts = min(all_bars[s]["timestamp"].max() for s in usable)
    start_ts = end_ts - pd.Timedelta(days=days)
    print(f"\nBacktest window: {start_ts} to {end_ts} ({days}d)")

    # Load RL policy if checkpoint provided
    rl_policy = None
    num_symbols_rl = len(usable)
    obs_size_rl = num_symbols_rl * 16 + 5 + num_symbols_rl
    num_actions_rl = 1 + 2 * num_symbols_rl

    if checkpoint_path and Path(checkpoint_path).exists():
        # Need to figure out the right obs/action sizes from the checkpoint
        ckpt_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # The checkpoint was trained on a different number of symbols
        # Infer from model weights
        first_weight = ckpt_meta["model"]["encoder.0.weight"]
        trained_obs_size = first_weight.shape[1]
        last_weight = ckpt_meta["model"]["actor.2.weight"]
        trained_num_actions = last_weight.shape[0]

        print(f"\nRL checkpoint: obs_size={trained_obs_size}, "
              f"num_actions={trained_num_actions}")

        # We'll build observations matching the trained size
        # Infer num_symbols: obs_size = S*16 + 5 + S = S*17 + 5
        trained_num_symbols = (trained_obs_size - 5) // 17
        print(f"  Trained on {trained_num_symbols} symbols, "
              f"evaluating on {len(usable)} symbols")

        try:
            rl_policy = load_rl_policy(
                checkpoint_path, trained_obs_size, trained_num_actions,
                hidden_size=hidden_size, arch=arch,
            )
            obs_size_rl = trained_obs_size
            num_actions_rl = trained_num_actions
            num_symbols_rl = trained_num_symbols
            print(f"  Policy loaded OK")
        except Exception as e:
            print(f"  Policy load failed: {e}")
    else:
        print("\nNo RL checkpoint — using momentum proxy for RL signals")

    results = {}

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"MODE: {mode.upper()}")
        print(f"{'=' * 60}")

        all_bar_dfs = []
        all_action_dfs = []
        api_calls = 0
        t0 = time.time()

        for sym in usable:
            bars = all_bars[sym]
            window = bars[(bars["timestamp"] >= start_ts) & (bars["timestamp"] <= end_ts)].copy()
            if len(window) < 24:
                continue

            all_bar_dfs.append(window[["timestamp", "open", "high", "low", "close"]].assign(symbol=sym))

            closes_all = bars["close"].values
            actions = []
            last_gemini_plan = None

            for i, (_, bar) in enumerate(window.iterrows()):
                ts = bar["timestamp"]
                close = float(bar["close"])
                bar_idx = bars.index.get_loc(bar.name)

                # Get RL signal
                if rl_policy is not None:
                    obs = build_obs_from_bars(bars, bar_idx, num_symbols_rl)
                    direction, confidence = get_rl_signal_from_obs(
                        rl_policy, obs, num_symbols_rl)
                else:
                    # Momentum proxy
                    if bar_idx >= 48:
                        sma_12 = closes_all[max(0, bar_idx-12):bar_idx].mean()
                        sma_48 = closes_all[max(0, bar_idx-48):bar_idx].mean()
                        if sma_12 > sma_48:
                            gap = (sma_12 - sma_48) / sma_48
                            direction = "long"
                            confidence = min(0.95, 0.5 + gap * 50)
                        else:
                            direction = "flat"
                            confidence = 0.5
                    else:
                        direction = "flat"
                        confidence = 0.5

                if mode == "rl_only":
                    if direction == "long" and confidence > 0.55:
                        buy_price = close * 0.998
                        sell_price = close * 1.008
                    elif direction == "short" and confidence > 0.55:
                        buy_price = 0
                        sell_price = close * 1.002
                    else:
                        buy_price = 0
                        sell_price = 0

                elif mode == "gemini_only":
                    # Call Gemini with Chronos2 forecasts (sampled)
                    if i % gemini_sample_rate == 0:
                        hist_slice = bars[bars["timestamp"] <= ts].tail(25)
                        fc_1h = get_forecast_at(all_fc_h1[sym], ts)
                        fc_24h = get_forecast_at(all_fc_h24[sym], ts)

                        from llm_hourly_trader.gemini_wrapper import build_prompt
                        try:
                            prompt = build_prompt(
                                symbol=sym,
                                history_rows=hist_slice.to_dict("records"),
                                forecast_1h=fc_1h, forecast_24h=fc_24h,
                                current_position="flat",
                                cash=initial_cash, equity=initial_cash,
                                allowed_directions=["long"],
                                asset_class="crypto", maker_fee=0.0008,
                            )
                            from llm_hourly_trader.providers import call_llm
                            plan = call_llm(prompt, model=model,
                                          thinking_level=thinking_level)
                            api_calls += 1
                            last_gemini_plan = plan
                            buy_price = plan.buy_price if plan.direction == "long" else 0
                            sell_price = plan.sell_price if plan.sell_price > 0 else 0
                        except Exception as e:
                            print(f"    Gemini error: {e}")
                            buy_price = 0
                            sell_price = 0
                    else:
                        # Reuse last Gemini plan, adjusted for current price
                        if last_gemini_plan and last_gemini_plan.direction == "long":
                            buy_price = close * 0.998
                            sell_price = close * 1.008
                        else:
                            buy_price = 0
                            sell_price = 0

                elif mode == "rl_gemini":
                    # RL direction + Gemini price refinement
                    if direction == "flat" or confidence < 0.5:
                        buy_price = 0
                        sell_price = 0
                    elif i % gemini_sample_rate == 0:
                        # Call Gemini to refine prices
                        hist_slice = bars[bars["timestamp"] <= ts].tail(25)
                        fc_1h = get_forecast_at(all_fc_h1[sym], ts)
                        fc_24h = get_forecast_at(all_fc_h24[sym], ts)

                        rl_sig = RLSignal(0, sym, direction, confidence, 0.0, 1.0)
                        try:
                            prompt = build_hybrid_prompt(
                                symbol=sym, rl_signal=rl_sig,
                                history_rows=hist_slice.to_dict("records"),
                                current_price=close,
                                forecast_1h=fc_1h, forecast_24h=fc_24h,
                            )
                            from llm_hourly_trader.providers import call_llm
                            plan = call_llm(prompt, model=model,
                                          thinking_level=thinking_level)
                            api_calls += 1
                            last_gemini_plan = plan
                            buy_price = plan.buy_price if plan.direction in ("long", "short") else 0
                            sell_price = plan.sell_price if plan.sell_price > 0 else 0
                        except Exception as e:
                            # Fallback to RL-only prices
                            buy_price = close * 0.998
                            sell_price = close * 1.008
                    else:
                        # Between Gemini calls, use RL direction with fixed spreads
                        if direction == "long":
                            buy_price = close * 0.998
                            sell_price = close * 1.008
                        else:
                            buy_price = 0
                            sell_price = close * 0.998

                actions.append({
                    "timestamp": ts, "symbol": sym,
                    "buy_price": buy_price, "sell_price": sell_price,
                    "buy_amount": 100 if buy_price > 0 else 0,
                    "sell_amount": 100 if buy_price == 0 and sell_price > 0 else 0,
                })

                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {sym}: {i+1}/{len(window)} bars, "
                          f"{api_calls} API calls, {elapsed:.0f}s")

            all_action_dfs.append(pd.DataFrame(actions))

        if not all_bar_dfs:
            results[mode] = {"error": "no data"}
            continue

        bars_df = pd.concat(all_bar_dfs, ignore_index=True)
        actions_df = pd.concat(all_action_dfs, ignore_index=True)

        elapsed = time.time() - t0
        print(f"  Generated {len(actions_df)} actions in {elapsed:.0f}s "
              f"({api_calls} API calls)")

        # Run simulator
        cfg = HourlyTraderSimulationConfig(
            initial_cash=initial_cash,
            allocation_pct=1.0 / len(usable),
            max_leverage=1.0,
            enforce_market_hours=False,
            allow_short=False,
            decision_lag_bars=1,
            fill_buffer_bps=5.0,
            partial_fill_on_touch=True,
        )

        try:
            sim = HourlyTraderMarketSimulator(cfg)
            result = sim.run(bars_df, actions_df)
            total_return = (result.equity_curve.iloc[-1] / initial_cash - 1) * 100

            # Compute max drawdown from equity curve
            eq = result.equity_curve.values
            peak = np.maximum.accumulate(eq)
            dd = (eq - peak) / peak * 100
            max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

            # Compute win rate from fills (pair entries with exits)
            wins = 0
            total_trades = 0
            entry_prices = {}
            for fill in result.fills:
                sym = fill.symbol
                if fill.kind == "entry":
                    entry_prices[sym] = fill.price
                elif fill.kind == "exit" and sym in entry_prices:
                    total_trades += 1
                    if fill.price > entry_prices[sym]:
                        wins += 1
                    del entry_prices[sym]
            win_rate = wins / total_trades if total_trades > 0 else 0.0

            results[mode] = {
                "return_pct": total_return,
                "final_equity": float(result.equity_curve.iloc[-1]),
                "fills": len(result.fills),
                "trades": total_trades,
                "sortino": result.metrics.get("sortino", 0),
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "api_calls": api_calls,
                "elapsed_s": elapsed,
            }
            print(f"  Return: {total_return:+.2f}%")
            print(f"  Sortino: {results[mode]['sortino']:.2f}")
            print(f"  Fills: {len(result.fills)}, Trades: {total_trades}")
            print(f"  Win rate: {win_rate:.1%}")
            print(f"  Max DD: {max_dd:.2f}%")
        except Exception as e:
            print(f"  Sim error: {e}")
            import traceback
            traceback.print_exc()
            results[mode] = {"error": str(e)}

    # Summary
    print(f"\n{'=' * 60}")
    print("BACKTEST COMPARISON")
    print(f"{'=' * 60}")
    print(f"Period: {start_ts.date()} to {end_ts.date()} ({days}d)")
    print(f"Symbols: {', '.join(usable)}")
    print(f"Gemini model: {model} (sampled every {gemini_sample_rate}h)")
    print()

    for mode, data in results.items():
        if "error" in data:
            print(f"  {mode:15s}: ERROR - {data['error']}")
        else:
            print(f"  {mode:15s}: {data['return_pct']:+.2f}% | "
                  f"Sortino={data['sortino']:.2f} | "
                  f"DD={data['max_drawdown']:.2f}% | "
                  f"WR={data.get('win_rate', 0):.1%} | "
                  f"{data['fills']} fills | "
                  f"{data['api_calls']} API | "
                  f"{data['elapsed_s']:.0f}s")

    # Determine winner
    valid = {k: v for k, v in results.items() if "error" not in v}
    if len(valid) >= 2:
        best_mode = max(valid, key=lambda k: valid[k].get("sortino", -999))
        best_ret = max(valid, key=lambda k: valid[k]["return_pct"])
        print(f"\n  Best Sortino: {best_mode} ({valid[best_mode]['sortino']:.2f})")
        print(f"  Best Return:  {best_ret} ({valid[best_ret]['return_pct']:+.2f}%)")

        if "rl_gemini" in valid and "rl_only" in valid:
            delta_ret = valid["rl_gemini"]["return_pct"] - valid["rl_only"]["return_pct"]
            delta_sort = valid["rl_gemini"].get("sortino", 0) - valid["rl_only"].get("sortino", 0)
            print(f"\n  RL+Gemini vs RL-only:")
            print(f"    Return delta: {delta_ret:+.2f}%")
            print(f"    Sortino delta: {delta_sort:+.2f}")
            if delta_sort > 0 and delta_ret > 0:
                print(f"    VERDICT: RL+Gemini WINS")
            elif delta_sort < 0 and delta_ret < 0:
                print(f"    VERDICT: RL-only WINS")
            else:
                print(f"    VERDICT: MIXED (check metrics)")

    print(f"{'=' * 60}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Backtest RL vs Gemini vs RL+Gemini")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD", "SOLUSD"])
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--thinking-level", default="HIGH")
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--arch", default="mlp")
    parser.add_argument("--gemini-sample-rate", type=int, default=4,
                        help="Call Gemini every N bars (reduce API cost)")
    parser.add_argument("--modes", nargs="+",
                        default=["rl_only", "gemini_only", "rl_gemini"])
    args = parser.parse_args()

    results = run_backtest(
        symbols=args.symbols,
        checkpoint_path=args.checkpoint,
        days=args.days,
        initial_cash=args.cash,
        modes=args.modes,
        model=args.model,
        thinking_level=args.thinking_level,
        hidden_size=args.hidden_size,
        arch=args.arch,
        gemini_sample_rate=args.gemini_sample_rate,
    )

    # Save results
    import json
    out_path = REPO / "unified_orchestrator" / "backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
