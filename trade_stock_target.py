from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from loguru import logger

from marketsimulator import alpaca_wrapper_mock as broker
from marketsimulator.environment import activate_simulation
from marketsimulator.state import SimulationState

from gpt5_queries import query_to_gpt5_async


@dataclass
class Allocation:
    weight: float
    side: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark portfolio balancing strategies inside the simulator.",
    )
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "NVDA"], help="Symbols to evaluate.")
    parser.add_argument("--steps", type=int, default=16, help="Number of rebalance steps to simulate.")
    parser.add_argument("--step-size", type=int, default=1, help="Simulation steps to advance between rebalances.")
    parser.add_argument("--initial-cash", type=float, default=100_000.0, help="Initial simulator cash balance.")
    parser.add_argument("--max-positions", type=int, default=4, help="Maximum portfolio size per rebalance.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["top1", "top2", "top3", "top4", "equal_25", "gpt5"],
        help="Strategies to benchmark (subset of: top1, top2, top3, top4, equal_25, gpt5).",
    )
    parser.add_argument(
        "--forecast-rows",
        type=int,
        default=8,
        help="Number of forecast rows per symbol to include in GPT prompts.",
    )
    parser.add_argument("--skip-gpt", action="store_true", help="Skip GPT-5 allocation benchmarking.")
    parser.add_argument(
        "--gpt-reasoning",
        choices=["minimal", "low", "medium", "high"],
        default="low",
        help="Reasoning effort to request for GPT-5 allocation.",
    )
    parser.add_argument("--gpt-timeout", type=int, default=90, help="Timeout (seconds) for GPT-5 allocation calls.")
    parser.add_argument(
        "--gpt-max-output",
        type=int,
        default=2048,
        help="Maximum output tokens for GPT-5 allocation responses.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/simulator_balancing"),
        help="Directory to store run summaries.",
    )
    return parser.parse_args()


def _select_top(
    picks: Dict[str, Dict],
    count: int,
) -> Dict[str, Dict]:
    ordered = sorted(
        picks.items(),
        key=lambda item: item[1].get("composite_score", 0),
        reverse=True,
    )
    selected = dict(ordered[:count])
    return selected


def allocation_top_k_equal(k: int):
    def allocator(
        picks: Dict[str, Dict],
        _analysis: Dict[str, Dict],
        _state: SimulationState,
    ) -> Dict[str, Allocation]:
        if not picks:
            return {}
        selected = _select_top(picks, k)
        if not selected:
            return {}
        weight = 1.0 / len(selected)
        return {
            symbol: Allocation(weight=weight, side=data.get("side", "buy"))
            for symbol, data in selected.items()
        }

    return allocator


def allocation_equal_25(
    picks: Dict[str, Dict],
    _analysis: Dict[str, Dict],
    _state: SimulationState,
) -> Dict[str, Allocation]:
    if not picks:
        return {}
    selected = _select_top(picks, min(4, len(picks)))
    if not selected:
        return {}
    weight = 0.25 if len(selected) >= 4 else 1.0 / len(selected)
    return {
        symbol: Allocation(weight=weight, side=data.get("side", "buy"))
        for symbol, data in selected.items()
    }


def _gather_forecast_context(
    picks: Dict[str, Dict],
    analysis: Dict[str, Dict],
    max_rows: int,
) -> Dict[str, Dict]:
    context: Dict[str, Dict] = {}
    for symbol, data in analysis.items():
        predictions = data.get("predictions")
        if isinstance(predictions, pd.DataFrame):
            trimmed = predictions.head(max_rows).copy()
            trimmed = trimmed[
                [
                    col
                    for col in [
                        "date",
                        "close",
                        "predicted_close",
                        "predicted_high",
                        "predicted_low",
                        "simple_strategy_return",
                        "all_signals_strategy_return",
                        "entry_takeprofit_return",
                        "highlow_return",
                    ]
                    if col in trimmed.columns
                ]
            ]
            rows = trimmed.to_dict(orient="records")
        else:
            rows = []

        context[symbol] = {
            "side": data.get("side"),
            "avg_return": data.get("avg_return"),
            "strategy": data.get("strategy"),
            "predicted_movement": data.get("predicted_movement"),
            "directional_edge": data.get("directional_edge"),
            "edge_strength": data.get("edge_strength"),
            "expected_move_pct": data.get("expected_move_pct"),
            "unprofit_shutdown_return": data.get("unprofit_shutdown_return"),
            "predicted_high": data.get("predicted_high"),
            "predicted_low": data.get("predicted_low"),
            "predictions_preview": rows,
            "in_portfolio": symbol in picks,
        }
    return context


def _parse_gpt_allocation_response(response: str) -> Dict[str, Allocation]:
    if not response:
        return {}

    def _extract_json(text: str) -> Optional[str]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return text[start : end + 1]

    json_candidate = _extract_json(response)
    if not json_candidate:
        logger.warning("GPT-5 response did not contain JSON payload. Raw response:\n%s", response)
        return {}
    try:
        payload = json.loads(json_candidate)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse GPT-5 allocation JSON (%s). Raw segment: %s", exc, json_candidate)
        return {}

    allocations_raw: Iterable[Dict] = payload.get("allocations", [])
    parsed: Dict[str, Allocation] = {}
    for item in allocations_raw:
        symbol = str(item.get("symbol", "")).upper()
        try:
            weight = float(item.get("weight", 0))
        except (TypeError, ValueError):
            continue
        side = str(item.get("side", "buy")).lower()
        if symbol and weight >= 0:
            parsed[symbol] = Allocation(weight=weight, side=side if side in {"buy", "sell"} else "buy")
    return parsed


def allocation_gpt5(
    picks: Dict[str, Dict],
    analysis: Dict[str, Dict],
    state: SimulationState,
    *,
    max_rows: int,
    reasoning_effort: str,
    timeout: int,
    max_output_tokens: int,
) -> Dict[str, Allocation]:
    if not picks:
        return {}

    context = _gather_forecast_context(picks, analysis, max_rows=max_rows)
    summary = {
        symbol: {
            "strategy": data.get("strategy"),
            "avg_return": data.get("avg_return"),
            "side": data.get("side"),
        }
        for symbol, data in picks.items()
    }

    prompt = (
        "You are helping allocate capital across trading strategies. "
        "Each symbol already has a direction ('buy' or 'sell') determined by the forecast pipeline. "
        "You must return a JSON object with an 'allocations' array. "
        "Each allocation entry should contain 'symbol', 'weight', and 'side'. "
        "Weights must be non-negative fractions that sum to 1.0 when combined across all entries you return. "
        "Only include symbols listed in the provided context. "
        "Do not invent new symbols. "
        "If you believe a symbol should receive zero weight, omit it from the allocations array. "
        "Keep reasoning concise and ensure the final JSON is strictly valid."
        "\n\nContext:\n"
        + json.dumps(
            {
                "picks": summary,
                "analysis": context,
                "current_equity": state.equity,
                "cash": state.cash,
            },
            indent=2,
        )
    )

    system_message = (
        "You are a portfolio balancing assistant. "
        "Respect the provided trade direction for each symbol. "
        "Return machine-readable JSON with allocation weights."
    )

    try:
        response_text = asyncio.run(
            query_to_gpt5_async(
                prompt,
                system_message=system_message,
                extra_data={
                    "reasoning_effort": reasoning_effort,
                    "lock_reasoning_effort": True,
                    "max_output_tokens": max_output_tokens,
                    "timeout": timeout,
                },
                model="gpt-5-mini",
            )
        )
    except Exception as exc:
        logger.error("GPT-5 allocation request failed: %s", exc)
        return {}

    allocations = _parse_gpt_allocation_response(response_text)
    if not allocations:
        logger.warning("GPT-5 allocation empty; falling back to equal weighting.")
        return {}
    total_weight = sum(alloc.weight for alloc in allocations.values())
    if not total_weight or not np.isfinite(total_weight):
        logger.warning("GPT-5 allocation weights invalid (%s); falling back to equal weighting.", total_weight)
        return {}
    normalised: Dict[str, Allocation] = {}
    for symbol, alloc in allocations.items():
        weight = alloc.weight / total_weight
        side = alloc.side
        normalised[symbol] = Allocation(weight=weight, side=side)
    return normalised


def apply_allocation(state: SimulationState, allocations: Dict[str, Allocation]) -> None:
    # Flatten previous exposure
    for symbol in list(state.positions.keys()):
        state.close_position(symbol)
    state.update_market_prices()
    broker.re_setup_vars()

    equity = state.equity
    if equity <= 0:
        logger.warning("State equity <= 0; skipping allocation.")
        return

    orders: List[Dict[str, float]] = []
    for symbol, alloc in allocations.items():
        series = state.prices.get(symbol)
        if not series:
            logger.warning("No price series available for %s; skipping allocation entry.", symbol)
            continue
        price = series.price("Close")
        notional = max(alloc.weight, 0) * equity
        if price <= 0 or notional <= 0:
            continue
        qty = notional / price
        orders.append(
            {
                "symbol": symbol,
                "qty": qty,
                "side": alloc.side,
                "price": price,
            }
        )

    if not orders:
        logger.info("No orders generated for allocation step; holding cash.")
        return

    broker.execute_portfolio_orders(orders)
    broker.re_setup_vars()
    state.update_market_prices()


def run_balancing_strategy(
    name: str,
    allocator,
    args: argparse.Namespace,
) -> Dict:
    logger.info("Running strategy '%s'", name)
    with activate_simulation(
        symbols=args.symbols,
        initial_cash=args.initial_cash,
        use_mock_analytics=False,
    ) as controller:
        from trade_stock_e2e import analyze_symbols, build_portfolio  # defer until after simulator patches

        state = controller.state
        snapshots: List[Dict] = []
        for step in range(args.steps):
            timestamp = controller.current_time()
            analysis = analyze_symbols(args.symbols)
            if not analysis:
                logger.warning("No analysis results at step %d; skipping allocation.", step)
                controller.advance_steps(args.step_size)
                state.update_market_prices()
                snapshots.append(
                    {
                        "step": step,
                        "timestamp": str(timestamp),
                        "equity": state.equity,
                        "cash": state.cash,
                        "allocations": {},
                    }
                )
                continue

            picks = build_portfolio(
                analysis,
                min_positions=1,
                max_positions=args.max_positions,
                max_expanded=args.max_positions,
            )

            allocations = allocator(picks, analysis, state)
            if allocations:
                apply_allocation(state, allocations)
            else:
                logger.info("Allocator returned no allocations; closing positions and remaining in cash.")
                apply_allocation(state, {})

            state.update_market_prices()
            snapshots.append(
                {
                    "step": step,
                    "timestamp": str(timestamp),
                    "equity": state.equity,
                    "cash": state.cash,
                    "allocations": {
                        symbol: {
                            "weight": alloc.weight,
                            "side": alloc.side,
                        }
                        for symbol, alloc in allocations.items()
                    },
                }
            )

            controller.advance_steps(args.step_size)

        # Final state summary
        state.update_market_prices()
        final_equity = state.equity
        trades = len(state.trade_log)
        result = {
            "strategy": name,
            "final_equity": final_equity,
            "total_return": final_equity - args.initial_cash,
            "total_return_pct": (final_equity - args.initial_cash) / args.initial_cash if args.initial_cash else 0.0,
            "fees_paid": state.fees_paid,
            "trades_executed": trades,
            "snapshots": snapshots,
        }
    return result


def summarize_results(results: List[Dict]) -> None:
    if not results:
        logger.warning("No results to summarize.")
        return
    logger.info("\n=== Portfolio Balancing Benchmark ===")
    header = f"{'Strategy':<12} {'Final Equity':>14} {'Return ($)':>12} {'Return (%)':>11} {'Fees':>10} {'Trades':>8}"
    logger.info(header)
    for entry in results:
        logger.info(
            f"{entry['strategy']:<12} "
            f"{entry['final_equity']:>14,.2f} "
            f"{entry['total_return']:>12,.2f} "
            f"{entry['total_return_pct']*100:>10.2f}% "
            f"{entry['fees_paid']:>10,.2f} "
            f"{entry['trades_executed']:>8}"
        )


def ensure_results_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_results_dir(args.results_dir)

    available_allocators = {
        "top1": allocation_top_k_equal(1),
        "top2": allocation_top_k_equal(2),
        "top3": allocation_top_k_equal(3),
        "top4": allocation_top_k_equal(4),
        "equal_25": allocation_equal_25,
    }

    if not args.skip_gpt:
        available_allocators["gpt5"] = lambda picks, analysis, state: allocation_gpt5(
            picks,
            analysis,
            state,
            max_rows=args.forecast_rows,
            reasoning_effort=args.gpt_reasoning,
            timeout=args.gpt_timeout,
            max_output_tokens=args.gpt_max_output,
        )

    selected_strategies = []
    for name in args.strategies:
        key = name.lower()
        if key == "gpt5" and args.skip_gpt:
            logger.info("Skipping GPT-5 strategy as requested.")
            continue
        allocator = available_allocators.get(key)
        if allocator is None:
            logger.warning("Unknown strategy '%s'; skipping.", name)
            continue
        selected_strategies.append((key, allocator))

    if not selected_strategies:
        raise SystemExit("No valid strategies selected for benchmarking.")

    results: List[Dict] = []
    for name, allocator in selected_strategies:
        result = run_balancing_strategy(name, allocator, args)
        results.append(result)
        output_file = args.results_dir / f"{name}_summary.json"
        output_file.write_text(json.dumps(result, indent=2))
        logger.info("Saved strategy summary to %s", output_file)

    summarize_results(results)


if __name__ == "__main__":
    main()
