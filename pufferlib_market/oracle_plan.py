from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from pufferlib_market.evaluate_multiperiod import load_policy, make_policy_fn
from pufferlib_market.hourly_replay import (
    INITIAL_CASH,
    MktdData,
    P_CLOSE,
    P_HIGH,
    P_LOW,
    Position,
    _apply_short_borrow_cost,
    _build_obs,
    _close_position,
    _compute_equity,
    _is_tradable,
    _open_long_limit,
    _open_short_limit,
    read_mktd,
)
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity


@dataclass
class OraclePortfolioState:
    cash: float = INITIAL_CASH
    position: Position | None = None
    hold_steps: int = 0


@dataclass(frozen=True)
class OracleScenarioResult:
    fill_buffer_bps: float
    slippage_bps: float
    total_return_pct: float
    max_drawdown_pct: float
    pnl_smoothness: float


@dataclass(frozen=True)
class LegacyActionScore:
    action: int
    action_label: str
    robust_score: float
    mean_return_pct: float
    worst_return_pct: float
    mean_max_drawdown_pct: float
    mean_pnl_smoothness: float
    scenario_count: int
    scenario_returns_pct: tuple[float, ...]


@dataclass(frozen=True)
class OracleStep:
    step: int
    current_position: str
    best_action: int
    best_action_label: str
    best_action_score: float
    near_best_action_labels: tuple[str, ...]
    chosen_action_gap: float
    action_scores: tuple[LegacyActionScore, ...]


@dataclass(frozen=True)
class OracleComparisonSummary:
    step_count: int
    exact_match_rate: float
    near_best_match_rate: float
    mean_regret: float
    median_regret: float
    worst_regret: float


def legacy_action_label(action: int, symbols: Sequence[str]) -> str:
    symbol_count = len(symbols)
    if action <= 0:
        return "flat"
    if action <= symbol_count:
        return f"long:{symbols[action - 1]}"
    short_idx = action - symbol_count - 1
    if 0 <= short_idx < symbol_count:
        return f"short:{symbols[short_idx]}"
    return f"unknown:{action}"


def _position_label(position: Position | None, symbols: Sequence[str]) -> str:
    if position is None:
        return "flat"
    side = "short" if position.is_short else "long"
    return f"{side}:{symbols[position.sym]}"


def enumerate_legacy_actions(num_symbols: int) -> list[int]:
    if num_symbols <= 0:
        raise ValueError("num_symbols must be positive")
    return list(range(0, 1 + 2 * int(num_symbols)))


def _clone_position(position: Position | None) -> Position | None:
    if position is None:
        return None
    return Position(
        sym=int(position.sym),
        is_short=bool(position.is_short),
        qty=float(position.qty),
        entry_price=float(position.entry_price),
    )


def clone_portfolio_state(state: OraclePortfolioState) -> OraclePortfolioState:
    return OraclePortfolioState(
        cash=float(state.cash),
        position=_clone_position(state.position),
        hold_steps=int(state.hold_steps),
    )


def _action_target(action: int, num_symbols: int) -> tuple[str, int | None]:
    if action <= 0:
        return "flat", None
    if action <= num_symbols:
        return "long", action - 1
    return "short", action - num_symbols - 1


def _equity_at_close(data: MktdData, step: int, state: OraclePortfolioState) -> float:
    if state.position is None:
        return float(state.cash)
    close_price = float(data.prices[step, state.position.sym, P_CLOSE])
    return float(_compute_equity(float(state.cash), state.position, close_price))


def apply_legacy_action_step(
    data: MktdData,
    *,
    step: int,
    state: OraclePortfolioState,
    action: int,
    fee_rate: float,
    slippage_bps: float,
    fill_buffer_bps: float,
    max_leverage: float,
    short_borrow_apr: float,
    periods_per_year: float,
) -> tuple[OraclePortfolioState, float, float]:
    symbol_count = data.num_symbols
    effective_fee = float(fee_rate) + max(0.0, float(slippage_bps)) / 10_000.0
    next_state = clone_portfolio_state(state)
    position = next_state.position
    current_sym = position.sym if position is not None else -1
    current_tradable = _is_tradable(data, step, current_sym) if current_sym >= 0 else True

    if position is None:
        equity_before = float(next_state.cash)
    else:
        price_cur = float(data.prices[step, position.sym, P_CLOSE])
        equity_before = float(_compute_equity(next_state.cash, position, price_cur))

    target_side, target_sym = _action_target(action, symbol_count)
    if target_side == "flat":
        if position is not None and current_tradable:
            close_price = float(data.prices[step, position.sym, P_CLOSE])
            next_state.cash, _ = _close_position(next_state.cash, position, close_price, effective_fee)
            next_state.position = None
            next_state.hold_steps = 0
        elif position is not None:
            next_state.hold_steps += 1
    else:
        assert target_sym is not None
        target_pos_matches = (
            position is not None
            and position.sym == target_sym
            and position.is_short == (target_side == "short")
        )
        if target_pos_matches:
            next_state.hold_steps += 1
        else:
            target_tradable = _is_tradable(data, step, target_sym)
            if position is not None and not current_tradable:
                next_state.hold_steps += 1
            elif not target_tradable:
                if position is not None:
                    next_state.hold_steps += 1
            else:
                if position is not None:
                    close_price = float(data.prices[step, position.sym, P_CLOSE])
                    next_state.cash, _ = _close_position(next_state.cash, position, close_price, effective_fee)
                    next_state.position = None
                close_price = float(data.prices[step, target_sym, P_CLOSE])
                low_price = float(data.prices[step, target_sym, P_LOW])
                high_price = float(data.prices[step, target_sym, P_HIGH])
                if target_side == "short":
                    next_state.cash, next_state.position = _open_short_limit(
                        cash=next_state.cash,
                        sym=target_sym,
                        close_price=close_price,
                        low_price=low_price,
                        high_price=high_price,
                        fee_rate=effective_fee,
                        max_leverage=max_leverage,
                        allocation_pct=1.0,
                        level_offset_bps=0.0,
                        fill_buffer_bps=fill_buffer_bps,
                    )
                else:
                    next_state.cash, next_state.position = _open_long_limit(
                        cash=next_state.cash,
                        sym=target_sym,
                        close_price=close_price,
                        low_price=low_price,
                        high_price=high_price,
                        fee_rate=effective_fee,
                        max_leverage=max_leverage,
                        allocation_pct=1.0,
                        level_offset_bps=0.0,
                        fill_buffer_bps=fill_buffer_bps,
                    )
                next_state.hold_steps = 0 if next_state.position is not None else next_state.hold_steps

    next_step = min(step + 1, data.num_timesteps - 1)
    if next_state.position is None:
        equity_after = float(next_state.cash)
    else:
        next_close = float(data.prices[next_step, next_state.position.sym, P_CLOSE])
        next_state.cash, _ = _apply_short_borrow_cost(
            cash=next_state.cash,
            pos=next_state.position,
            price=next_close,
            short_borrow_apr=short_borrow_apr,
            periods_per_year=periods_per_year,
        )
        equity_after = float(_compute_equity(next_state.cash, next_state.position, next_close))
    return next_state, float(equity_before), float(equity_after)


def score_legacy_action(
    data: MktdData,
    *,
    start_step: int,
    state: OraclePortfolioState,
    action: int,
    lookahead_steps: int,
    fee_rate: float,
    fill_buffer_bps_values: Sequence[float],
    slippage_bps_values: Sequence[float],
    max_leverage: float = 1.0,
    short_borrow_apr: float = 0.0,
    periods_per_year: float = 365.0,
) -> LegacyActionScore:
    scenarios: list[OracleScenarioResult] = []
    horizon = max(1, int(lookahead_steps))
    action_label = legacy_action_label(action, data.symbols)

    for fill_buffer_bps in fill_buffer_bps_values:
        for slippage_bps in slippage_bps_values:
            scenario_state = clone_portfolio_state(state)
            initial_equity = _equity_at_close(data, start_step, scenario_state)
            equity_curve = [float(initial_equity)]
            peak_equity = float(initial_equity)
            max_drawdown = 0.0
            last_step = start_step

            for offset in range(horizon):
                current_step = start_step + offset
                if current_step >= data.num_timesteps - 1:
                    break
                scenario_state, _, equity_after = apply_legacy_action_step(
                    data,
                    step=current_step,
                    state=scenario_state,
                    action=action,
                    fee_rate=fee_rate,
                    slippage_bps=slippage_bps,
                    fill_buffer_bps=fill_buffer_bps,
                    max_leverage=max_leverage,
                    short_borrow_apr=short_borrow_apr,
                    periods_per_year=periods_per_year,
                )
                peak_equity = max(peak_equity, float(equity_after))
                if peak_equity > 0.0:
                    max_drawdown = max(max_drawdown, (peak_equity - float(equity_after)) / peak_equity)
                equity_curve.append(float(equity_after))
                last_step = current_step + 1

            if scenario_state.position is not None:
                close_step = min(last_step, data.num_timesteps - 1)
                close_price = float(data.prices[close_step, scenario_state.position.sym, P_CLOSE])
                effective_fee = float(fee_rate) + max(0.0, float(slippage_bps)) / 10_000.0
                scenario_state.cash, _ = _close_position(
                    scenario_state.cash,
                    scenario_state.position,
                    close_price,
                    effective_fee,
                )
                scenario_state.position = None
                final_equity = float(scenario_state.cash)
                peak_equity = max(peak_equity, final_equity)
                if peak_equity > 0.0:
                    max_drawdown = max(max_drawdown, (peak_equity - final_equity) / peak_equity)
                equity_curve.append(final_equity)
            else:
                final_equity = float(scenario_state.cash)

            total_return_pct = 100.0 * ((final_equity / initial_equity) - 1.0) if initial_equity > 0 else 0.0
            pnl_smoothness = compute_pnl_smoothness_from_equity(np.asarray(equity_curve, dtype=np.float64))
            scenarios.append(
                OracleScenarioResult(
                    fill_buffer_bps=float(fill_buffer_bps),
                    slippage_bps=float(slippage_bps),
                    total_return_pct=float(total_return_pct),
                    max_drawdown_pct=100.0 * float(max_drawdown),
                    pnl_smoothness=float(pnl_smoothness),
                )
            )

    if not scenarios:
        raise ValueError("No oracle scenarios were evaluated")

    returns = np.asarray([row.total_return_pct for row in scenarios], dtype=np.float64)
    drawdowns = np.asarray([row.max_drawdown_pct for row in scenarios], dtype=np.float64)
    smoothness = np.asarray([row.pnl_smoothness for row in scenarios], dtype=np.float64)
    robust_score = float(
        np.min(returns)
        + 0.5 * np.mean(returns)
        - 0.25 * np.mean(drawdowns)
        - 25.0 * np.mean(smoothness)
    )
    return LegacyActionScore(
        action=int(action),
        action_label=str(action_label),
        robust_score=robust_score,
        mean_return_pct=float(np.mean(returns)),
        worst_return_pct=float(np.min(returns)),
        mean_max_drawdown_pct=float(np.mean(drawdowns)),
        mean_pnl_smoothness=float(np.mean(smoothness)),
        scenario_count=len(scenarios),
        scenario_returns_pct=tuple(float(x) for x in returns.tolist()),
    )


def score_all_legacy_actions(
    data: MktdData,
    *,
    start_step: int,
    state: OraclePortfolioState,
    lookahead_steps: int,
    fee_rate: float,
    fill_buffer_bps_values: Sequence[float],
    slippage_bps_values: Sequence[float],
    max_leverage: float = 1.0,
    short_borrow_apr: float = 0.0,
    periods_per_year: float = 365.0,
) -> list[LegacyActionScore]:
    scores = [
        score_legacy_action(
            data,
            start_step=start_step,
            state=state,
            action=action,
            lookahead_steps=lookahead_steps,
            fee_rate=fee_rate,
            fill_buffer_bps_values=fill_buffer_bps_values,
            slippage_bps_values=slippage_bps_values,
            max_leverage=max_leverage,
            short_borrow_apr=short_borrow_apr,
            periods_per_year=periods_per_year,
        )
        for action in enumerate_legacy_actions(data.num_symbols)
    ]
    return sorted(scores, key=lambda row: (row.robust_score, row.worst_return_pct), reverse=True)


def build_legacy_oracle_trace(
    data: MktdData,
    *,
    max_steps: int,
    lookahead_steps: int,
    fee_rate: float,
    fill_buffer_bps_values: Sequence[float],
    slippage_bps_values: Sequence[float],
    near_best_score_gap: float = 0.25,
    max_leverage: float = 1.0,
    short_borrow_apr: float = 0.0,
    periods_per_year: float = 365.0,
) -> list[OracleStep]:
    total_steps = min(int(max_steps), max(0, data.num_timesteps - 1))
    if total_steps <= 0:
        return []

    base_fill_buffer = float(fill_buffer_bps_values[0])
    base_slippage = float(slippage_bps_values[0])
    state = OraclePortfolioState()
    steps: list[OracleStep] = []

    for step in range(total_steps):
        scores = score_all_legacy_actions(
            data,
            start_step=step,
            state=state,
            lookahead_steps=lookahead_steps,
            fee_rate=fee_rate,
            fill_buffer_bps_values=fill_buffer_bps_values,
            slippage_bps_values=slippage_bps_values,
            max_leverage=max_leverage,
            short_borrow_apr=short_borrow_apr,
            periods_per_year=periods_per_year,
        )
        best = scores[0]
        near_best = tuple(
            score.action_label
            for score in scores
            if score.robust_score >= best.robust_score - float(near_best_score_gap)
        )
        second_best_score = scores[1].robust_score if len(scores) > 1 else best.robust_score
        steps.append(
            OracleStep(
                step=int(step),
                current_position=_position_label(state.position, data.symbols),
                best_action=int(best.action),
                best_action_label=str(best.action_label),
                best_action_score=float(best.robust_score),
                near_best_action_labels=near_best,
                chosen_action_gap=float(best.robust_score - second_best_score),
                action_scores=tuple(scores),
            )
        )
        state, _, _ = apply_legacy_action_step(
            data,
            step=step,
            state=state,
            action=best.action,
            fee_rate=fee_rate,
            slippage_bps=base_slippage,
            fill_buffer_bps=base_fill_buffer,
            max_leverage=max_leverage,
            short_borrow_apr=short_borrow_apr,
            periods_per_year=periods_per_year,
        )
    return steps


def rollout_policy_actions(
    data: MktdData,
    *,
    policy_fn: Callable[[np.ndarray], int],
    max_steps: int,
    fee_rate: float,
    fill_buffer_bps: float,
    slippage_bps: float,
    max_leverage: float = 1.0,
    short_borrow_apr: float = 0.0,
    periods_per_year: float = 365.0,
) -> list[int]:
    total_steps = min(int(max_steps), max(0, data.num_timesteps - 1))
    state = OraclePortfolioState()
    actions: list[int] = []
    for step in range(total_steps):
        obs = _build_obs(
            data,
            step,
            state.position,
            state.cash,
            state.hold_steps,
            step,
            total_steps,
            portfolio_scale=INITIAL_CASH,
        )
        action = int(policy_fn(obs))
        actions.append(action)
        state, _, _ = apply_legacy_action_step(
            data,
            step=step,
            state=state,
            action=action,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            fill_buffer_bps=fill_buffer_bps,
            max_leverage=max_leverage,
            short_borrow_apr=short_borrow_apr,
            periods_per_year=periods_per_year,
        )
    return actions


def compare_policy_actions_to_oracle(
    oracle_steps: Sequence[OracleStep],
    policy_actions: Sequence[int],
) -> OracleComparisonSummary:
    if len(oracle_steps) != len(policy_actions):
        raise ValueError("oracle_steps and policy_actions must have the same length")
    if not oracle_steps:
        return OracleComparisonSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    regrets: list[float] = []
    exact_matches = 0
    near_best_matches = 0

    for oracle_step, policy_action in zip(oracle_steps, policy_actions, strict=True):
        by_action = {score.action: score for score in oracle_step.action_scores}
        best_score = oracle_step.best_action_score
        policy_score = by_action.get(int(policy_action))
        policy_action_label = policy_score.action_label if policy_score is not None else "missing"
        if int(policy_action) == int(oracle_step.best_action):
            exact_matches += 1
        if policy_action_label in oracle_step.near_best_action_labels:
            near_best_matches += 1
        regrets.append(float(best_score - (policy_score.robust_score if policy_score is not None else best_score)))

    regret_arr = np.asarray(regrets, dtype=np.float64)
    return OracleComparisonSummary(
        step_count=len(oracle_steps),
        exact_match_rate=float(exact_matches / len(oracle_steps)),
        near_best_match_rate=float(near_best_matches / len(oracle_steps)),
        mean_regret=float(np.mean(regret_arr)),
        median_regret=float(np.median(regret_arr)),
        worst_regret=float(np.max(regret_arr)),
    )


def load_checkpoint_policy_fn(
    checkpoint_path: str | Path,
    data: MktdData,
    *,
    deterministic: bool = True,
    device: str = "cpu",
) -> Callable[[np.ndarray], int]:
    torch_device = torch.device(device)
    policy, _, _ = load_policy(
        str(checkpoint_path),
        data.num_symbols,
        device=torch_device,
        features_per_sym=int(data.features.shape[2]),
    )
    return make_policy_fn(
        policy,
        num_symbols=data.num_symbols,
        deterministic=deterministic,
        device=torch_device,
    )


def _parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def _json_ready_steps(steps: Sequence[OracleStep]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in steps:
        row = asdict(step)
        row["action_scores"] = [asdict(score) for score in step.action_scores]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a robust legacy-action oracle trace")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--lookahead-steps", type=int, default=5)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffers", default="3,5,8")
    parser.add_argument("--slippages", default="0,5,8")
    parser.add_argument("--near-best-score-gap", type=float, default=0.25)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.0)
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    data = read_mktd(args.data_path)
    fill_buffers = _parse_float_list(args.fill_buffers)
    slippages = _parse_float_list(args.slippages)
    oracle_steps = build_legacy_oracle_trace(
        data,
        max_steps=args.max_steps,
        lookahead_steps=args.lookahead_steps,
        fee_rate=args.fee_rate,
        fill_buffer_bps_values=fill_buffers,
        slippage_bps_values=slippages,
        near_best_score_gap=args.near_best_score_gap,
        max_leverage=args.max_leverage,
        short_borrow_apr=args.short_borrow_apr,
        periods_per_year=args.periods_per_year,
    )

    output: dict[str, object] = {
        "data_path": str(args.data_path),
        "max_steps": int(args.max_steps),
        "lookahead_steps": int(args.lookahead_steps),
        "fill_buffers": fill_buffers,
        "slippages": slippages,
        "oracle_steps": _json_ready_steps(oracle_steps),
    }

    if args.checkpoint:
        policy_fn = load_checkpoint_policy_fn(args.checkpoint, data)
        policy_actions = rollout_policy_actions(
            data,
            policy_fn=policy_fn,
            max_steps=args.max_steps,
            fee_rate=args.fee_rate,
            fill_buffer_bps=float(fill_buffers[0]),
            slippage_bps=float(slippages[0]),
            max_leverage=args.max_leverage,
            short_borrow_apr=args.short_borrow_apr,
            periods_per_year=args.periods_per_year,
        )
        comparison = compare_policy_actions_to_oracle(oracle_steps, policy_actions)
        output["checkpoint"] = str(args.checkpoint)
        output["policy_actions"] = [
            {"step": idx, "action": int(action), "action_label": legacy_action_label(action, data.symbols)}
            for idx, action in enumerate(policy_actions)
        ]
        output["comparison"] = asdict(comparison)

    text = json.dumps(output, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
