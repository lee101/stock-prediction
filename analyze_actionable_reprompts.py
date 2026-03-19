from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from llm_hourly_trader.providers import (
    _passes_reprompt_filters,
    _plan_has_entry,
    _plan_is_actionable,
    _should_run_reprompt,
    call_llm,
)
from unified_orchestrator import backtest_hybrid as bh


def classify_plan(plan) -> str:
    buy_price = float(getattr(plan, "buy_price", 0.0) or 0.0)
    sell_price = float(getattr(plan, "sell_price", 0.0) or 0.0)
    if buy_price > 0.0 and sell_price > 0.0:
        return "entry_with_exit"
    if buy_price > 0.0:
        return "entry_only"
    if sell_price > 0.0:
        return "exit_only"
    return "flat_hold"


def analyze_window(
    symbols: list[str],
    *,
    model: str,
    start_ts: str,
    end_ts: str,
    reprompt_policy: str = "actionable",
    review_max_confidence: float | None = None,
) -> dict[str, object]:
    start_bound = pd.Timestamp(start_ts, tz="UTC")
    end_bound = pd.Timestamp(end_ts, tz="UTC")
    if start_bound > end_bound:
        raise ValueError(f"start_ts must be <= end_ts, got {start_bound} > {end_bound}")

    per_symbol: dict[str, dict[str, object]] = {}
    total_decisions = 0
    total_actionable = 0
    total_entry_only_reviews = 0
    total_would_review = 0

    for sym in symbols:
        bars = bh.load_bars(sym)
        fc_h1 = bh.load_forecasts(sym, "h1")
        fc_h24 = bh.load_forecasts(sym, "h24")
        window = bars[(bars["timestamp"] >= start_bound) & (bars["timestamp"] <= end_bound)].copy()
        if len(window) < 12:
            continue

        position = bh.PositionState()
        prev_outcome = None
        cash = 10_000.0
        equity = 10_000.0
        active_plan = None
        last_decision_key = None

        decisions = 0
        actionable = 0
        entry_only_reviews = 0
        would_review = 0
        categories = {
            "entry_with_exit": 0,
            "entry_only": 0,
            "exit_only": 0,
            "flat_hold": 0,
        }

        for _, bar in window.iterrows():
            ts = bar["timestamp"]
            close = float(bar["close"])
            decision_key = bh._decision_bucket(ts, "daily")
            refresh_plan = active_plan is None or decision_key != last_decision_key

            if position.direction == "long":
                position.unrealized_pnl_pct = (close - position.entry_price) / position.entry_price * 100.0
                equity = cash + (close / position.entry_price) * 100.0
            else:
                equity = cash

            if refresh_plan:
                hist_slice = bars[bars["timestamp"] <= ts].tail(25)
                prompt = bh._build_context_header(ts, position, prev_outcome, cash, equity) + bh.build_prompt(
                    symbol=sym,
                    history_rows=hist_slice.to_dict("records"),
                    forecast_1h=bh.get_forecast_at(fc_h1, ts),
                    forecast_24h=bh.get_forecast_at(fc_h24, ts),
                    current_position=position.direction,
                    cash=cash,
                    equity=equity,
                    allowed_directions=["long"],
                    asset_class="crypto",
                    maker_fee=0.0008,
                )
                active_plan = call_llm(prompt, model=model, cache_only=True)
                decisions += 1
                actionable += int(_plan_is_actionable(active_plan))
                entry_only_reviews += int(_plan_has_entry(active_plan))
                would_review += int(
                    _should_run_reprompt(active_plan, reprompt_policy)
                    and _passes_reprompt_filters(
                        active_plan,
                        review_max_confidence=review_max_confidence,
                    )
                )
                categories[classify_plan(active_plan)] += 1
                last_decision_key = decision_key

            prev_outcome = bh._update_position(position, active_plan, close, close)

        flat_holds = decisions - actionable
        per_symbol[sym] = {
            "decisions": decisions,
            "actionable": actionable,
            "flat_holds": flat_holds,
            "actionable_pct": (actionable / decisions) if decisions else 0.0,
            "entry_only_reviews": entry_only_reviews,
            "entry_only_pct": (entry_only_reviews / decisions) if decisions else 0.0,
            "would_review": would_review,
            "would_review_pct": (would_review / decisions) if decisions else 0.0,
            "categories": categories,
        }
        total_decisions += decisions
        total_actionable += actionable
        total_entry_only_reviews += entry_only_reviews
        total_would_review += would_review

    return {
        "model": model,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "symbols": symbols,
        "reprompt_policy": reprompt_policy,
        "review_max_confidence": review_max_confidence,
        "per_symbol": per_symbol,
        "total": {
            "decisions": total_decisions,
            "actionable": total_actionable,
            "flat_holds": total_decisions - total_actionable,
            "actionable_pct": (total_actionable / total_decisions) if total_decisions else 0.0,
            "entry_only_reviews": total_entry_only_reviews,
            "entry_only_pct": (total_entry_only_reviews / total_decisions) if total_decisions else 0.0,
            "would_review": total_would_review,
            "would_review_pct": (total_would_review / total_decisions) if total_decisions else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate how many cached daily first-pass prompts would trigger an actionable review pass.",
    )
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--reprompt-policy", choices=["always", "actionable", "entry_only"], default="actionable")
    parser.add_argument("--review-max-confidence", type=float, default=None)
    parser.add_argument("--start-ts", required=True)
    parser.add_argument("--end-ts", required=True)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    result = analyze_window(
        args.symbols,
        model=args.model,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        reprompt_policy=args.reprompt_policy,
        review_max_confidence=args.review_max_confidence,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
