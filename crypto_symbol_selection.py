"""Crypto symbol selection and max-hold-hours sweep.

Runs the LLM hourly backtest to rank candidate crypto symbols by Sortino ratio
and profit/loss, and finds the optimal hold-time exit parameter.

Modes
-----
per_symbol  : Run each candidate symbol in isolation. Ranks by Sortino.
hold_sweep  : Run the current baseline symbols (BTC/ETH/SOL/LTC) with
              different max_hold_hours to find the optimal exit timing.
combined    : Run all candidate symbols together (shared cash), show per-symbol
              PnL breakdown and rank. Most realistic for live portfolio sizing.
all         : Run all three modes in sequence.

Usage
-----
cd /nvme0n1-disk/code/stock-prediction
python scripts/crypto_symbol_selection.py --mode per_symbol --days 7
python scripts/crypto_symbol_selection.py --mode hold_sweep --days 7
python scripts/crypto_symbol_selection.py --mode combined --days 7 --candidates BTCUSD ETHUSD SOLUSD LTCUSD AVAXUSD DOGEUSD
python scripts/crypto_symbol_selection.py --mode all --days 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from llm_hourly_trader.backtest import run_backtest
from llm_hourly_trader.config import BacktestConfig

# Candidate pool — all have bar + forecast data available
CANDIDATE_CRYPTO = ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD", "DOGEUSD"]

# Current production set (pre-selection)
BASELINE_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD"]

# Hold hours to sweep
HOLD_HOURS_SWEEP = [2, 4, 6, 8, 12, 24]

RESULTS_DIR = REPO / "llm_hourly_trader" / "results"


def _make_config(
    max_hold_hours: int = 6,
    max_position_pct: float = 0.40,
    days: int = 7,
    model: str = "gemini-3.1-flash-lite-preview",
) -> BacktestConfig:
    return BacktestConfig(
        initial_cash=10_000.0,
        max_hold_hours=max_hold_hours,
        max_position_pct=max_position_pct,
        rate_limit_seconds=4.2,
        model=model,
        prompt_variant="default",
        parallel_workers=1,
    )


def _win_rate(trades: list[dict], symbol: str | None = None) -> float:
    """Fraction of completed round-trips that were profitable."""
    exits = [
        t for t in trades
        if t["side"] in ("sell", "cover", "close")
        and (symbol is None or t["symbol"] == symbol)
    ]
    if not exits:
        return 0.0
    wins = sum(1 for t in exits if t["realized_pnl"] > 0)
    return wins / len(exits)


def run_per_symbol(candidates: list[str], days: int, max_position_pct: float) -> None:
    """Run each symbol in isolation and rank by Sortino."""
    print(f"\n{'='*70}")
    print(f"PER-SYMBOL SELECTION  ({days}d, pos={max_position_pct*100:.0f}%)")
    print(f"{'='*70}")
    print(f"Candidates: {candidates}")
    print(
        "Note: Each symbol uses its own forecast window (fair individual comparison).\n"
        "LTC/AVAX data ends Feb-06; DOGE ends Feb-28; BTC/ETH/SOL end Mar-09.\n"
    )

    rows: list[dict] = []
    for sym in candidates:
        print(f"\n{'─'*50}")
        print(f"  Running {sym} in isolation ({days}d)...")
        config = _make_config(max_hold_hours=6, max_position_pct=max_position_pct, days=days)
        try:
            result = run_backtest(symbols=[sym], days=days, config=config)
        except Exception as e:
            print(f"  {sym}: FAILED — {e}")
            continue
        if "error" in result:
            print(f"  {sym}: SKIPPED — {result['error']}")
            continue

        # Load trades for win-rate
        model_tag = config.model.replace("/", "_").replace(".", "")
        tag = f"{model_tag}_{config.prompt_variant}_{days}d_{sym}"
        trades_path = RESULTS_DIR / f"{tag}_trades.json"
        win_rate = 0.0
        if trades_path.exists():
            with open(trades_path) as f:
                trades = json.load(f)
            win_rate = _win_rate(trades, sym)

        rows.append({
            "symbol": sym,
            "total_return_pct": result["total_return_pct"],
            "sortino": result["sortino"],
            "max_drawdown_pct": result["max_drawdown_pct"],
            "entries": result["entries"],
            "exits": result["exits"],
            "realized_pnl": result["realized_pnl"],
            "win_rate": win_rate,
            "window": result.get("window", "?"),
        })

    if not rows:
        print("\nNo results.")
        return

    # Rank by Sortino
    rows.sort(key=lambda r: r["sortino"], reverse=True)

    print(f"\n{'='*70}")
    print(f"RANKING BY SORTINO ({days}d)")
    print(f"{'='*70}")
    header = f"{'Symbol':10s} {'Return%':>9s} {'Sortino':>9s} {'MaxDD%':>8s} {'WinRate':>9s} {'PnL$':>9s} {'Trades':>7s}"
    print(header)
    print("─" * len(header))
    for r in rows:
        print(
            f"{r['symbol']:10s} "
            f"{r['total_return_pct']:+9.2f}% "
            f"{r['sortino']:9.2f} "
            f"{r['max_drawdown_pct']:8.2f}% "
            f"{r['win_rate']:9.1%} "
            f"${r['realized_pnl']:+8.2f} "
            f"{r['entries']+r['exits']:7d}"
        )

    # Recommendation: top symbols where Sortino > 1.0 and return > 0
    recommended = [r["symbol"] for r in rows if r["sortino"] > 1.0 and r["total_return_pct"] > 0]
    print(f"\nRECOMMENDED (Sortino > 1.0, return > 0): {recommended or '(none passed filters)'}")
    print(f"Current CRYPTO_SYMBOLS in orchestrator: {BASELINE_SYMBOLS}")

    # Save ranking
    out = RESULTS_DIR / f"symbol_selection_{days}d_per_symbol.json"
    with open(out, "w") as f:
        json.dump({"days": days, "ranking": rows, "recommended": recommended}, f, indent=2)
    print(f"\nSaved: {out}")


def run_hold_sweep(symbols: list[str], days: int, max_position_pct: float) -> None:
    """Sweep max_hold_hours to find optimal exit timing."""
    print(f"\n{'='*70}")
    print(f"MAX-HOLD-HOURS SWEEP  (symbols={symbols}, {days}d, pos={max_position_pct*100:.0f}%)")
    print(f"{'='*70}")
    print(
        f"Sweeping: {HOLD_HOURS_SWEEP}\n"
        "Note: LLM calls are the same for all hold durations (only simulation changes).\n"
        "First run fetches/caches; subsequent runs are instant simulation reruns.\n"
    )

    rows: list[dict] = []
    for hold_h in HOLD_HOURS_SWEEP:
        print(f"\n{'─'*50}")
        print(f"  max_hold_hours = {hold_h}h ...")
        config = _make_config(max_hold_hours=hold_h, max_position_pct=max_position_pct, days=days)
        try:
            result = run_backtest(symbols=symbols, days=days, config=config)
        except Exception as e:
            print(f"  hold={hold_h}h: FAILED — {e}")
            continue
        if "error" in result:
            print(f"  hold={hold_h}h: SKIPPED — {result['error']}")
            continue

        rows.append({
            "max_hold_hours": hold_h,
            "total_return_pct": result["total_return_pct"],
            "sortino": result["sortino"],
            "max_drawdown_pct": result["max_drawdown_pct"],
            "entries": result["entries"],
            "exits": result["exits"],
            "realized_pnl": result["realized_pnl"],
        })

    if not rows:
        print("\nNo results.")
        return

    best = max(rows, key=lambda r: r["sortino"])

    print(f"\n{'='*70}")
    print(f"MAX-HOLD SWEEP RESULTS (symbols={symbols})")
    print(f"{'='*70}")
    header = f"{'HoldH':>6s} {'Return%':>9s} {'Sortino':>9s} {'MaxDD%':>8s} {'PnL$':>9s} {'Trades':>7s}"
    print(header)
    print("─" * len(header))
    for r in rows:
        marker = " ◄ BEST" if r["max_hold_hours"] == best["max_hold_hours"] else ""
        print(
            f"{r['max_hold_hours']:6d}h "
            f"{r['total_return_pct']:+9.2f}% "
            f"{r['sortino']:9.2f} "
            f"{r['max_drawdown_pct']:8.2f}% "
            f"${r['realized_pnl']:+8.2f} "
            f"{r['entries']+r['exits']:7d}"
            f"{marker}"
        )

    print(f"\nOPTIMAL max_hold_hours = {best['max_hold_hours']}h "
          f"(Sortino={best['sortino']:.2f}, return={best['total_return_pct']:+.2f}%)")
    print(f"Current production setting: 6h (no live enforcement — GTC orders)")

    # Save
    out = RESULTS_DIR / f"hold_sweep_{'_'.join(symbols)}_{days}d.json"
    with open(out, "w") as f:
        json.dump({"days": days, "symbols": symbols, "sweep": rows, "optimal": best}, f, indent=2)
    print(f"\nSaved: {out}")


def run_combined(candidates: list[str], days: int, max_position_pct: float) -> None:
    """Run all candidates together, compare per-symbol PnL contribution."""
    print(f"\n{'='*70}")
    print(f"COMBINED BACKTEST  (candidates={candidates}, {days}d, pos={max_position_pct*100:.0f}%)")
    print(f"{'='*70}")
    print(
        "All symbols share cash pool. Window ends at earliest forecast cutoff.\n"
        "Per-symbol PnL shows each asset's contribution to the portfolio.\n"
    )

    config = _make_config(max_hold_hours=6, max_position_pct=max_position_pct, days=days)
    try:
        result = run_backtest(symbols=candidates, days=days, config=config)
    except Exception as e:
        print(f"Combined run FAILED: {e}")
        return
    if "error" in result:
        print(f"Combined run SKIPPED: {result['error']}")
        return

    per_sym = result.get("per_symbol", {})

    # Load trades for per-symbol win rates
    model_tag = config.model.replace("/", "_").replace(".", "")
    sym_tag = "_".join(result.get("symbols", candidates))
    tag = f"{model_tag}_{config.prompt_variant}_{days}d_{sym_tag}"
    trades_path = RESULTS_DIR / f"{tag}_trades.json"
    sym_win_rates: dict[str, float] = {}
    if trades_path.exists():
        with open(trades_path) as f:
            trades = json.load(f)
        for sym in per_sym:
            sym_win_rates[sym] = _win_rate(trades, sym)

    # Rank per-symbol by realized PnL
    ranked = sorted(per_sym.items(), key=lambda kv: kv[1]["realized_pnl"], reverse=True)

    print(f"\n{'='*70}")
    print(f"PER-SYMBOL CONTRIBUTION (combined portfolio)")
    print(f"{'='*70}")
    header = f"{'Symbol':10s} {'PnL$':>9s} {'WinRate':>9s} {'Entries':>8s} {'Exits':>7s} {'Fees$':>7s}"
    print(header)
    print("─" * len(header))
    for sym, stats in ranked:
        wr = sym_win_rates.get(sym, 0.0)
        print(
            f"{sym:10s} "
            f"${stats['realized_pnl']:+8.2f} "
            f"{wr:9.1%} "
            f"{stats['entries']:8d} "
            f"{stats['exits']:7d} "
            f"${stats['fees']:6.2f}"
        )

    positive_syms = [sym for sym, s in ranked if s["realized_pnl"] > 0]
    print(f"\nPositive-PnL symbols: {positive_syms}")
    print(f"Portfolio Sortino: {result['sortino']:.2f} | Return: {result['total_return_pct']:+.2f}%")

    out = RESULTS_DIR / f"combined_selection_{'_'.join(candidates)}_{days}d.json"
    with open(out, "w") as f:
        json.dump({
            "days": days,
            "candidates": candidates,
            "portfolio_return_pct": result["total_return_pct"],
            "portfolio_sortino": result["sortino"],
            "per_symbol": {
                sym: {**stats, "win_rate": sym_win_rates.get(sym, 0.0)}
                for sym, stats in per_sym.items()
            },
            "recommended": positive_syms,
        }, f, indent=2)
    print(f"\nSaved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto symbol selection and hold-time sweep")
    parser.add_argument(
        "--mode",
        choices=["per_symbol", "hold_sweep", "combined", "all"],
        default="all",
    )
    parser.add_argument("--days", type=int, default=7, help="Backtest window in days")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=CANDIDATE_CRYPTO,
        help="Candidate symbols (default: all 6)",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        default=BASELINE_SYMBOLS,
        help="Baseline symbols for hold sweep (default: BTC/ETH/SOL/LTC)",
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.40,
        help="Max position size per symbol (default: 0.40 = 40%%)",
    )
    args = parser.parse_args()

    if args.mode in ("per_symbol", "all"):
        run_per_symbol(args.candidates, args.days, args.max_position_pct)

    if args.mode in ("hold_sweep", "all"):
        run_hold_sweep(args.baseline, args.days, args.max_position_pct)

    if args.mode in ("combined", "all"):
        run_combined(args.candidates, args.days, args.max_position_pct)

    print("\nDone. Check llm_hourly_trader/results/ for saved outputs.")


if __name__ == "__main__":
    main()
