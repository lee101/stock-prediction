"""Neural DeepSeek planning with max-diff execution, calibration, and annual metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping, MutableMapping, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency in test environments
    from backtest_test3_inline import calibrate_signal  # type: ignore
except Exception:  # pragma: no cover - fallback when module unavailable
    def calibrate_signal(predictions: np.ndarray, actual_returns: np.ndarray) -> Tuple[float, float]:
        matched = min(len(predictions), len(actual_returns))
        if matched > 1:
            slope, intercept = np.polyfit(predictions[:matched], actual_returns[:matched], 1)
            return float(slope), float(intercept)
        return 1.0, 0.0
from stockagent.agentsimulator.data_models import AccountSnapshot, TradingPlan
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagentcombined.forecaster import CombinedForecastGenerator
from stockagentdeepseek_neural.agent import generate_deepseek_neural_plan
from stockagentdeepseek_neural.forecaster import (
    NeuralForecast,
    build_neural_forecasts,
)
from stockagentdeepseek_maxdiff.simulator import MaxDiffResult, MaxDiffSimulator
from src.fixtures import crypto_symbols


def _has_crypto(plan: TradingPlan) -> bool:
    return any(instr.symbol in crypto_symbols for instr in plan.instructions)


def _has_equities(plan: TradingPlan) -> bool:
    return any(instr.symbol not in crypto_symbols for instr in plan.instructions)


@dataclass(slots=True)
class DeepSeekCombinedMaxDiffResult:
    plan: TradingPlan
    raw_response: str
    forecasts: Mapping[str, NeuralForecast]
    simulation: MaxDiffResult
    summary: Mapping[str, float]
    calibration: Mapping[str, float]


def simulate_deepseek_combined_maxdiff_plan(
    *,
    market_data: MarketDataBundle,
    account_snapshot: AccountSnapshot,
    target_date: date,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
    deepseek_kwargs: Mapping[str, object] | None = None,
    forecasts: Mapping[str, NeuralForecast] | None = None,
    simulator: MaxDiffSimulator | None = None,
    calibration_window: int = 14,
    generator: CombinedForecastGenerator | None = None,
) -> DeepSeekCombinedMaxDiffResult:
    """
    Generate a neural DeepSeek plan, execute it with the MaxDiff simulator, and capture calibration metrics.
    """

    working_generator = generator or CombinedForecastGenerator()
    if forecasts is None:
        forecasts = build_neural_forecasts(
            symbols=symbols or market_data.bars.keys(),
            market_data=market_data,
            prediction_length=1,
            generator=working_generator,
        )

    plan, raw_text, resolved_forecasts = generate_deepseek_neural_plan(
        market_data=market_data,
        account_snapshot=account_snapshot,
        target_date=target_date,
        symbols=symbols,
        include_market_history=include_market_history,
        deepseek_kwargs=deepseek_kwargs,
        forecasts=forecasts,
    )

    simulator_instance = simulator or MaxDiffSimulator(market_data=market_data)
    result = simulator_instance.run([plan])

    starting_nav = float(account_snapshot.cash or 0.0)
    if starting_nav == 0:
        starting_nav = float(account_snapshot.equity or 0.0)
    if starting_nav == 0:
        starting_nav = 1.0

    daily_return_pct = result.net_pnl / starting_nav

    summary: MutableMapping[str, float] = {
        "realized_pnl": result.realized_pnl,
        "fees": result.total_fees,
        "net_pnl": result.net_pnl,
        "ending_cash": result.ending_cash,
        "ending_equity": result.ending_equity,
        "daily_return_pct": daily_return_pct,
    }

    calibration: MutableMapping[str, float] = {}

    plan_symbols = {instruction.symbol for instruction in plan.instructions}
    if any(symbol not in crypto_symbols for symbol in plan_symbols):
        summary["annual_return_equity_pct"] = daily_return_pct * 252
    if any(symbol in crypto_symbols for symbol in plan_symbols):
        summary["annual_return_crypto_pct"] = daily_return_pct * 365

    if calibration_window > 1 and resolved_forecasts:
        for symbol in plan_symbols:
            if symbol not in resolved_forecasts:
                continue
            slope, intercept, raw_move, calibrated_move = _calibrate_symbol(
                generator=working_generator,
                bundle=market_data,
                symbol=symbol,
                target_date=target_date,
                window=calibration_window,
                forecast=resolved_forecasts[symbol],
            )
            calibration[f"{symbol}_calibration_slope"] = slope
            calibration[f"{symbol}_calibration_intercept"] = intercept
            calibration[f"{symbol}_raw_expected_move_pct"] = raw_move
            calibration[f"{symbol}_calibrated_expected_move_pct"] = calibrated_move

    return DeepSeekCombinedMaxDiffResult(
        plan=plan,
        raw_response=raw_text,
        forecasts=resolved_forecasts,
        simulation=result,
        summary=summary,
        calibration=calibration,
    )


def _calibrate_symbol(
    *,
    generator: CombinedForecastGenerator,
    bundle: MarketDataBundle,
    symbol: str,
    target_date: date,
    window: int,
    forecast: NeuralForecast,
) -> Tuple[float, float, float, float]:
    frame = bundle.get_symbol_bars(symbol)
    if frame.empty:
        return 1.0, 0.0, 0.0, 0.0
    frame = frame.sort_index()

    predictions: list[float] = []
    actuals: list[float] = []

    total_rows = len(frame)
    # Only run forecasts for the tail of the series that feeds the calibration window.
    if window > 0:
        start_idx = max(1, total_rows - window - 1)
    else:
        start_idx = 1
    if start_idx >= total_rows:
        start_idx = max(1, total_rows - 1)

    for idx in range(start_idx, total_rows):
        hist = frame.iloc[:idx]
        if hist.empty:
            continue
        prev_close = float(hist.iloc[-1]["close"])
        try:
            combined = generator.generate_for_symbol(
                symbol,
                prediction_length=1,
                historical_frame=hist,
            )
        except Exception:
            continue
        predicted_close = float(combined.combined.get("close", prev_close))
        predictions.append((predicted_close - prev_close) / prev_close if prev_close else 0.0)

        current_close = float(frame.iloc[idx]["close"])
        actuals.append((current_close - prev_close) / prev_close if prev_close else 0.0)

    if len(predictions) > window:
        predictions = predictions[-window:]
        actuals = actuals[-window:]

    if len(predictions) < 2:
        slope, intercept = 1.0, 0.0
    else:
        slope, intercept = calibrate_signal(
            np.array(predictions, dtype=np.float64),
            np.array(actuals, dtype=np.float64),
        )

    if symbol in bundle.bars and not bundle.bars[symbol].empty:
        last_close = float(bundle.bars[symbol].iloc[-1]["close"])
    else:
        last_close = 0.0
    predicted_close = float(forecast.combined.get("close", last_close))
    raw_move = (predicted_close - last_close) / last_close if last_close else 0.0
    calibrated_move = float(slope * raw_move + intercept)

    return float(slope), float(intercept), raw_move, calibrated_move


__all__ = ["DeepSeekCombinedMaxDiffResult", "simulate_deepseek_combined_maxdiff_plan"]
