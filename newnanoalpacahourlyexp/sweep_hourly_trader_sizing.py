from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd

from .marketsimulator.hourly_trader import HourlyTraderMarketSimulator, HourlyTraderSimulationConfig
from .run_hourly_trader_sim import _load_initial_state, _parse_symbols


def _parse_float_list(raw: str) -> list[float]:
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return [float(value) for value in values]


def _parse_str_list(raw: str) -> list[str]:
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one string value.")
    return values


def _parse_bool_list(raw: str) -> list[bool]:
    values = []
    for token in _parse_str_list(raw):
        normalized = token.lower()
        if normalized in {"1", "true", "yes", "on"}:
            values.append(True)
        elif normalized in {"0", "false", "no", "off"}:
            values.append(False)
        else:
            raise ValueError(f"Invalid boolean token: {token!r}")
    return values


def _load_replay_inputs(bars_csv: Path, actions_csv: Path, *, symbols: list[str] | None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    bars = pd.read_csv(bars_csv)
    actions = pd.read_csv(actions_csv)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    actions["timestamp"] = pd.to_datetime(actions["timestamp"], utc=True)
    bars["symbol"] = bars["symbol"].astype(str).str.upper().str.replace("/", "", regex=False)
    actions["symbol"] = actions["symbol"].astype(str).str.upper().str.replace("/", "", regex=False)
    if symbols:
        wanted = [symbol.upper() for symbol in symbols]
        bars = bars[bars["symbol"].isin(wanted)].copy()
        actions = actions[actions["symbol"].isin(wanted)].copy()
        symbol_order = wanted
    else:
        symbol_order = list(dict.fromkeys(bars["symbol"].tolist()))
    return bars.reset_index(drop=True), actions.reset_index(drop=True), symbol_order


def _run_trial(
    *,
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    symbols: list[str],
    initial_cash: float,
    initial_positions: dict[str, float] | None,
    initial_open_orders,
    allocation_pct: float,
    allocation_mode: str,
    intensity_scale: float,
    allow_position_adds: bool,
    always_full_exit: bool,
    price_offset_pct: float,
    min_gap_pct: float,
    fill_buffer_bps: float,
    decision_lag_bars: int,
    cancel_ack_delay_bars: int,
    partial_fill_on_touch: bool,
):
    simulator = HourlyTraderMarketSimulator(
        HourlyTraderSimulationConfig(
            initial_cash=float(initial_cash),
            initial_positions=initial_positions,
            initial_open_orders=initial_open_orders,
            allocation_pct=float(allocation_pct),
            allocation_mode=str(allocation_mode),
            intensity_scale=float(intensity_scale),
            allow_position_adds=bool(allow_position_adds),
            always_full_exit=bool(always_full_exit),
            price_offset_pct=float(price_offset_pct),
            min_gap_pct=float(min_gap_pct),
            fill_buffer_bps=float(fill_buffer_bps),
            decision_lag_bars=int(decision_lag_bars),
            cancel_ack_delay_bars=int(cancel_ack_delay_bars),
            partial_fill_on_touch=bool(partial_fill_on_touch),
            symbols=symbols,
        )
    )
    return simulator.run(bars, actions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep hourly trader position-sizing hyperparameters over saved replay inputs.")
    parser.add_argument("--bars-csv", required=True)
    parser.add_argument("--actions-csv", required=True)
    parser.add_argument("--symbols", default=None, help="Optional comma-separated symbol subset/order.")
    parser.add_argument("--initial-state", default=None, help="Optional JSON file from export_alpaca_initial_state.py.")
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--allocation-pct-values", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--allocation-mode-values", default="portfolio")
    parser.add_argument("--intensity-scale-values", default="1.0,1.5,2.0,3.0")
    parser.add_argument("--allow-position-adds-values", default="false,true")
    parser.add_argument("--always-full-exit-values", default="true,false")
    parser.add_argument("--price-offset-pct", type=float, default=0.0)
    parser.add_argument("--min-gap-pct", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--cancel-ack-delay-bars", type=int, default=1)
    parser.add_argument("--partial-fill-on-touch", dest="partial_fill_on_touch", action="store_true")
    parser.add_argument("--no-partial-fill-on-touch", dest="partial_fill_on_touch", action="store_false")
    parser.set_defaults(partial_fill_on_touch=True)
    parser.add_argument("--baseline-allocation-pct", type=float, default=None)
    parser.add_argument("--baseline-allocation-mode", default=None)
    parser.add_argument("--baseline-intensity-scale", type=float, default=None)
    parser.add_argument("--baseline-allow-position-adds", default=None)
    parser.add_argument("--baseline-always-full-exit", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols) if args.symbols else None
    bars, actions, symbol_order = _load_replay_inputs(Path(args.bars_csv), Path(args.actions_csv), symbols=symbols)

    initial_cash = float(args.initial_cash)
    initial_positions = None
    initial_open_orders = None
    if args.initial_state:
        initial_cash, initial_positions, initial_open_orders = _load_initial_state(Path(args.initial_state))

    allocation_pct_values = _parse_float_list(args.allocation_pct_values)
    allocation_mode_values = _parse_str_list(args.allocation_mode_values)
    intensity_scale_values = _parse_float_list(args.intensity_scale_values)
    allow_position_adds_values = _parse_bool_list(args.allow_position_adds_values)
    always_full_exit_values = _parse_bool_list(args.always_full_exit_values)

    baseline_signature = None
    if (
        args.baseline_allocation_pct is not None
        and args.baseline_allocation_mode is not None
        and args.baseline_intensity_scale is not None
        and args.baseline_allow_position_adds is not None
        and args.baseline_always_full_exit is not None
    ):
        baseline_signature = (
            float(args.baseline_allocation_pct),
            str(args.baseline_allocation_mode),
            float(args.baseline_intensity_scale),
            str(args.baseline_allow_position_adds).strip().lower() in {"1", "true", "yes", "on"},
            str(args.baseline_always_full_exit).strip().lower() in {"1", "true", "yes", "on"},
        )

    rows: list[dict[str, object]] = []
    best_result = None
    best_trial = None
    for allocation_pct, allocation_mode, intensity_scale, allow_position_adds, always_full_exit in product(
        allocation_pct_values,
        allocation_mode_values,
        intensity_scale_values,
        allow_position_adds_values,
        always_full_exit_values,
    ):
        result = _run_trial(
            bars=bars,
            actions=actions,
            symbols=symbol_order,
            initial_cash=initial_cash,
            initial_positions=initial_positions,
            initial_open_orders=initial_open_orders,
            allocation_pct=allocation_pct,
            allocation_mode=allocation_mode,
            intensity_scale=intensity_scale,
            allow_position_adds=allow_position_adds,
            always_full_exit=always_full_exit,
            price_offset_pct=float(args.price_offset_pct),
            min_gap_pct=float(args.min_gap_pct),
            fill_buffer_bps=float(args.fill_buffer_bps),
            decision_lag_bars=int(args.decision_lag_bars),
            cancel_ack_delay_bars=int(args.cancel_ack_delay_bars),
            partial_fill_on_touch=bool(args.partial_fill_on_touch),
        )
        final_equity = float(result.equity_curve.iloc[-1]) if not result.equity_curve.empty else float(initial_cash)
        row = {
            "allocation_pct": float(allocation_pct),
            "allocation_mode": str(allocation_mode),
            "intensity_scale": float(intensity_scale),
            "allow_position_adds": bool(allow_position_adds),
            "always_full_exit": bool(always_full_exit),
            "total_return": float(result.metrics.get("total_return", 0.0)),
            "sortino": float(result.metrics.get("sortino", 0.0)),
            "mean_hourly_return": float(result.metrics.get("mean_hourly_return", 0.0)),
            "fill_count": int(len(result.fills)),
            "max_gross_exposure": float(result.per_hour["gross_exposure"].max()) if not result.per_hour.empty else 0.0,
            "final_equity": final_equity,
        }
        if baseline_signature is not None:
            row["is_baseline"] = (
                baseline_signature
                == (
                    float(allocation_pct),
                    str(allocation_mode),
                    float(intensity_scale),
                    bool(allow_position_adds),
                    bool(always_full_exit),
                )
            )
        rows.append(row)
        rank_key = (row["total_return"], row["sortino"], row["final_equity"])
        if best_trial is None or rank_key > best_trial:
            best_trial = rank_key
            best_result = (row, result)

    ranking = pd.DataFrame(rows).sort_values(
        ["total_return", "sortino", "final_equity"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_dir / "ranking.csv", index=False)

    if best_result is None:
        raise RuntimeError("Sizing sweep produced no results.")
    best_row, result = best_result
    (output_dir / "best_config.json").write_text(json.dumps(best_row, indent=2))
    pd.DataFrame([fill.__dict__ for fill in result.fills]).to_csv(output_dir / "best_fills.csv", index=False)
    result.per_hour.to_csv(output_dir / "best_per_hour.csv", index=False)

    print(ranking.head(10).to_string(index=False))
    if "is_baseline" in ranking.columns and ranking["is_baseline"].any():
        baseline_rank = int(ranking.index[ranking["is_baseline"]].tolist()[0]) + 1
        print(f"\nBaseline rank: {baseline_rank}/{len(ranking)}")


if __name__ == "__main__":
    main()
