#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.inference import generate_actions_from_frame
from binanceneural.marketsimulator import SimulationConfig, run_shared_cash_simulation
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from binanceneural.sweep import apply_action_overrides
from differentiable_loss_utils import HOURLY_PERIODS_PER_YEAR
from src.action_frame_cache import load_or_generate_action_frame
from src.robust_trading_metrics import (
    compute_max_drawdown,
    compute_pnl_smoothness_from_equity,
    summarize_scenario_results,
)
from src.torch_load_utils import torch_load_compat


DEFAULT_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]


@dataclass(frozen=True)
class CheckpointCandidate:
    checkpoint_path: Path
    label: str


def parse_csv_list(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def parse_symbols(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        for token in parse_csv_list(str(raw)):
            symbol = token.upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                result.append(symbol)
    if not result:
        raise ValueError("At least one symbol is required.")
    return result


def parse_float_list(value: str) -> list[float]:
    values = [float(token) for token in parse_csv_list(value)]
    if not values:
        raise ValueError(f"Expected at least one float in {value!r}.")
    return values


def parse_int_list(value: str) -> list[int]:
    values = [int(token) for token in parse_csv_list(value)]
    if not values:
        raise ValueError(f"Expected at least one integer in {value!r}.")
    return values


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.startswith("epoch_"):
        try:
            return int(stem.split("_", 1)[1]), stem
        except ValueError:
            pass
    return (0, stem)


def _candidate_label(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def resolve_checkpoint_candidates(
    inputs: Sequence[str],
    *,
    sample_epochs: Sequence[int] | None = None,
    use_all_epochs: bool = False,
) -> list[CheckpointCandidate]:
    candidates: list[CheckpointCandidate] = []
    seen: set[Path] = set()
    requested_epochs = {int(epoch) for epoch in sample_epochs or []}
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_file():
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(CheckpointCandidate(checkpoint_path=resolved, label=_candidate_label(resolved)))
            continue
        if path.is_dir():
            epoch_paths = sorted(path.glob("epoch_*.pt"), key=_checkpoint_sort_key)
            if not epoch_paths:
                raise FileNotFoundError(f"No epoch checkpoints found in {path}.")
            if requested_epochs:
                epoch_paths = [item for item in epoch_paths if _checkpoint_sort_key(item)[0] in requested_epochs]
                if not epoch_paths:
                    raise FileNotFoundError(
                        f"Requested epochs {sorted(requested_epochs)} were not found in {path}."
                    )
            elif not use_all_epochs:
                epoch_paths = [epoch_paths[-1]]
            for item in epoch_paths:
                resolved = item.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append(CheckpointCandidate(checkpoint_path=resolved, label=_candidate_label(resolved)))
            continue
        raise FileNotFoundError(f"Checkpoint path does not exist: {raw}")
    if not candidates:
        raise ValueError("No checkpoint candidates resolved.")
    return candidates


def load_checkpoint_model(checkpoint_path: Path, *, input_dim: int) -> torch.nn.Module:
    payload = torch_load_compat(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)
    cfg = payload.get("config", TrainingConfig())
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    policy_cfg = policy_config_from_payload(cfg, input_dim=input_dim, state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def build_start_state_maps(
    frames_by_symbol: dict[str, pd.DataFrame],
    *,
    initial_cash: float,
    start_symbol: str | None,
    position_fraction: float,
) -> dict[str, Any]:
    if start_symbol is None:
        return {
            "initial_cash": float(initial_cash),
            "initial_inventory_by_symbol": {},
            "initial_cost_basis_by_symbol": {},
        }
    symbol = str(start_symbol).upper()
    if symbol not in frames_by_symbol:
        raise ValueError(f"Cannot seed start state for {symbol}: symbol missing from scenario.")
    frame = frames_by_symbol[symbol]
    if frame.empty:
        raise ValueError(f"Cannot seed start state for {symbol}: no bars available.")
    price = float(frame["close"].iloc[0])
    if price <= 0.0:
        raise ValueError(f"Cannot seed start state for {symbol}: invalid close {price}.")
    fraction = min(max(float(position_fraction), 0.0), 1.0)
    deployed_cash = float(initial_cash) * fraction
    qty = deployed_cash / price if deployed_cash > 0.0 else 0.0
    return {
        "initial_cash": float(initial_cash) - deployed_cash,
        "initial_inventory_by_symbol": {symbol: qty} if qty > 0.0 else {},
        "initial_cost_basis_by_symbol": {symbol: price} if qty > 0.0 else {},
    }


def compute_selection_score(
    summary: dict[str, float],
    *,
    min_trade_count_mean: float,
    require_all_positive: bool,
) -> float:
    trade_shortfall = max(0.0, float(min_trade_count_mean) - float(summary.get("trade_count_mean", 0.0)))
    score = float(summary.get("robust_score", 0.0)) - 0.75 * trade_shortfall
    worst_return = float(summary.get("return_worst_pct", 0.0))
    if require_all_positive and worst_return <= 0.0:
        score -= 100.0 + 10.0 * abs(worst_return)
    return score


def _annualized_return_pct(equity_curve: pd.Series) -> float:
    values = equity_curve.to_numpy(dtype=float)
    if len(values) < 2 or values[0] <= 0.0 or values[-1] <= 0.0:
        return 0.0
    periods = len(values) - 1
    growth = values[-1] / values[0]
    if growth <= 0.0:
        return -100.0
    annualized = growth ** (HOURLY_PERIODS_PER_YEAR / periods) - 1.0
    return float(annualized * 100.0)


def _trade_count(result: Any) -> int:
    return int(
        sum(len(sym_result.trades) for sym_result in getattr(result, "per_symbol", {}).values())
    )


def _extract_bars(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ["timestamp", "symbol", "open", "high", "low", "close"]].copy()


def _extract_actions(frame: pd.DataFrame) -> pd.DataFrame:
    action_cols = [
        "timestamp",
        "symbol",
        "buy_price",
        "sell_price",
        "buy_amount",
        "sell_amount",
        "trade_amount",
    ]
    optional_cols = [col for col in ("hold_hours", "allocation_fraction") if col in frame.columns]
    return frame.loc[:, action_cols + optional_cols].copy()


def build_common_window(
    merged_by_symbol: dict[str, pd.DataFrame],
    *,
    window_hours: int,
) -> dict[str, pd.DataFrame]:
    windowed: dict[str, pd.DataFrame] = {}
    for symbol, merged in merged_by_symbol.items():
        frame = merged.sort_values("timestamp").reset_index(drop=True)
        if window_hours > 0:
            end_ts = pd.to_datetime(frame["timestamp"], utc=True).max()
            start_ts = end_ts - pd.Timedelta(hours=int(window_hours))
            frame = frame.loc[pd.to_datetime(frame["timestamp"], utc=True) >= start_ts].copy()
        if frame.empty:
            return {}
        windowed[symbol] = frame.reset_index(drop=True)

    common_ts: Optional[pd.Index] = None
    for frame in windowed.values():
        timestamps = pd.Index(pd.to_datetime(frame["timestamp"], utc=True))
        common_ts = timestamps if common_ts is None else common_ts.intersection(timestamps)
    if common_ts is None or len(common_ts) < 2:
        return {}

    restricted: dict[str, pd.DataFrame] = {}
    for symbol, frame in windowed.items():
        restricted_frame = frame.loc[frame["timestamp"].isin(common_ts)].copy()
        restricted_frame = restricted_frame.sort_values("timestamp").reset_index(drop=True)
        if restricted_frame.empty:
            return {}
        restricted[symbol] = restricted_frame
    return restricted


def prepare_candidate_frames(
    candidate: CheckpointCandidate,
    data_modules: dict[str, BinanceHourlyDataModule],
    *,
    horizon: int,
    device: torch.device,
    action_cache_root: Path,
) -> dict[str, pd.DataFrame]:
    first_module = next(iter(data_modules.values()))
    model = load_checkpoint_model(candidate.checkpoint_path, input_dim=len(first_module.feature_columns))
    merged_by_symbol: dict[str, pd.DataFrame] = {}
    try:
        for symbol, data_module in data_modules.items():
            frame = data_module.val_dataset.frame.copy().reset_index(drop=True)

            def _generate_actions() -> pd.DataFrame:
                return generate_actions_from_frame(
                    model=model,
                    frame=frame,
                    feature_columns=data_module.feature_columns,
                    normalizer=data_module.normalizer,
                    sequence_length=int(data_module.config.sequence_length),
                    horizon=int(horizon),
                    device=device,
                )

            actions, _cache_hit = load_or_generate_action_frame(
                cache_root=action_cache_root,
                symbol=symbol,
                checkpoint_path=candidate.checkpoint_path,
                frame=frame,
                feature_columns=data_module.feature_columns,
                normalizer=data_module.normalizer,
                sequence_length=int(data_module.config.sequence_length),
                horizon=int(horizon),
                generator=_generate_actions,
            )
            if "symbol" not in actions.columns:
                actions["symbol"] = symbol
            merged = frame.merge(actions, on=["timestamp", "symbol"], how="inner")
            merged = merged.sort_values("timestamp").reset_index(drop=True)
            if merged.empty:
                raise ValueError(f"No merged bars/actions for {symbol} using {candidate.label}.")
            merged_by_symbol[symbol] = merged
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return merged_by_symbol


def evaluate_candidate(
    candidate: CheckpointCandidate,
    data_modules: dict[str, BinanceHourlyDataModule],
    *,
    horizon: int,
    device: torch.device,
    action_cache_root: Path,
    window_hours_list: Sequence[int],
    decision_lags: Sequence[int],
    fill_buffer_bps_list: Sequence[float],
    intensity_scales: Sequence[float],
    price_offsets: Sequence[float],
    initial_cash: float,
    position_fraction: float,
    maker_fee: float,
    max_hold_hours: int | None,
    one_side_per_bar: bool,
    min_trade_count_mean: float,
    require_all_positive: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged_by_symbol = prepare_candidate_frames(
        candidate,
        data_modules,
        horizon=horizon,
        device=device,
        action_cache_root=action_cache_root,
    )
    symbols = list(merged_by_symbol.keys())
    scenario_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for intensity_scale in intensity_scales:
        for price_offset_pct in price_offsets:
            config_name = (
                f"{candidate.label}|i={float(intensity_scale):.3f}|o={float(price_offset_pct):.5f}"
            )
            config_rows: list[dict[str, Any]] = []
            for window_hours in window_hours_list:
                restricted = build_common_window(merged_by_symbol, window_hours=int(window_hours))
                if not restricted:
                    continue
                adjusted = {
                    symbol: apply_action_overrides(
                        frame,
                        intensity_scale=float(intensity_scale),
                        price_offset_pct=float(price_offset_pct),
                    )
                    for symbol, frame in restricted.items()
                }
                combined = pd.concat(adjusted.values(), ignore_index=True).sort_values(["timestamp", "symbol"])
                bars_df = _extract_bars(combined)
                actions_df = _extract_actions(combined)

                for decision_lag_bars in decision_lags:
                    for fill_buffer_bps in fill_buffer_bps_list:
                        for start_symbol in [None, *symbols]:
                            start_state = "flat" if start_symbol is None else start_symbol
                            start_maps = build_start_state_maps(
                                restricted,
                                initial_cash=float(initial_cash),
                                start_symbol=start_symbol,
                                position_fraction=float(position_fraction),
                            )
                            result = run_shared_cash_simulation(
                                bars_df,
                                actions_df,
                                SimulationConfig(
                                    maker_fee=float(maker_fee),
                                    initial_cash=float(start_maps["initial_cash"]),
                                    initial_inventory_by_symbol=start_maps["initial_inventory_by_symbol"],
                                    initial_cost_basis_by_symbol=start_maps["initial_cost_basis_by_symbol"],
                                    max_hold_hours=int(max_hold_hours) if max_hold_hours is not None else None,
                                    fill_buffer_bps=float(fill_buffer_bps),
                                    decision_lag_bars=int(decision_lag_bars),
                                    one_side_per_bar=bool(one_side_per_bar),
                                ),
                            )
                            equity = result.combined_equity
                            row = {
                                "config_name": config_name,
                                "candidate_label": candidate.label,
                                "checkpoint": str(candidate.checkpoint_path),
                                "period": f"{int(window_hours)}h",
                                "window_hours": int(window_hours),
                                "start_state": start_state,
                                "decision_lag_bars": int(decision_lag_bars),
                                "fill_buffer_bps": float(fill_buffer_bps),
                                "intensity_scale": float(intensity_scale),
                                "price_offset_pct": float(price_offset_pct),
                                "return_pct": float(result.metrics.get("total_return", 0.0) * 100.0),
                                "annualized_return_pct": _annualized_return_pct(equity),
                                "sortino": float(result.metrics.get("sortino", 0.0)),
                                "max_drawdown_pct": float(compute_max_drawdown(equity) * 100.0),
                                "pnl_smoothness": float(compute_pnl_smoothness_from_equity(equity)),
                                "trade_count": _trade_count(result),
                                "final_equity": float(equity.iloc[-1]) if not equity.empty else float(initial_cash),
                            }
                            config_rows.append(row)
                            scenario_rows.append(row)
            if not config_rows:
                continue
            summary = summarize_scenario_results(config_rows)
            selection_score = compute_selection_score(
                summary,
                min_trade_count_mean=float(min_trade_count_mean),
                require_all_positive=bool(require_all_positive),
            )
            summary_rows.append(
                {
                    "config_name": config_name,
                    "candidate_label": candidate.label,
                    "checkpoint": str(candidate.checkpoint_path),
                    "intensity_scale": float(intensity_scale),
                    "price_offset_pct": float(price_offset_pct),
                    "selection_score": float(selection_score),
                    **summary,
                }
            )
    if not summary_rows:
        raise ValueError(f"No robustness scenarios were produced for {candidate.label}.")
    return summary_rows, scenario_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust checkpoint sweep for binanceneural hourly models.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint files or directories.")
    parser.add_argument("--sample-epochs", default="", help="Optional comma-separated epoch numbers for directory inputs.")
    parser.add_argument("--all-epochs", action="store_true", help="Evaluate all epoch_*.pt files in directory inputs.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--validation-days", type=int, default=90)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--window-hours", default="168,336,720,1440")
    parser.add_argument("--decision-lag-list", default="1,2")
    parser.add_argument("--fill-buffer-bps-list", default="0,5,10")
    parser.add_argument("--intensity-list", default="1.0")
    parser.add_argument("--price-offset-list", default="0.0")
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--max-hold-hours", type=int, default=24)
    parser.add_argument("--no-one-side-per-bar", action="store_true")
    parser.add_argument("--min-trade-count-mean", type=float, default=6.0)
    parser.add_argument("--require-all-positive", action="store_true")
    parser.add_argument("--data-root", default="trainingdatahourly/crypto")
    parser.add_argument("--forecast-cache-root", default="binanceneural/forecast_cache")
    parser.add_argument("--device", default=None)
    parser.add_argument("--action-cache-root", default="experiments/binanceneural_action_cache")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = parse_symbols(args.symbols)
    sample_epochs = parse_int_list(args.sample_epochs) if str(args.sample_epochs).strip() else []
    candidates = resolve_checkpoint_candidates(
        args.checkpoints,
        sample_epochs=sample_epochs,
        use_all_epochs=bool(args.all_epochs),
    )

    data_modules: dict[str, BinanceHourlyDataModule] = {}
    for symbol in symbols:
        data_modules[symbol] = BinanceHourlyDataModule(
            DatasetConfig(
                symbol=symbol,
                data_root=Path(args.data_root),
                forecast_cache_root=Path(args.forecast_cache_root),
                sequence_length=int(args.sequence_length),
                forecast_horizons=(int(args.horizon),),
                validation_days=int(args.validation_days),
                cache_only=True,
            )
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("experiments") / time.strftime("binanceneural_robustness_%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    window_hours_list = parse_int_list(args.window_hours)
    decision_lags = parse_int_list(args.decision_lag_list)
    fill_buffer_bps_list = parse_float_list(args.fill_buffer_bps_list)
    intensity_scales = parse_float_list(args.intensity_list)
    price_offsets = parse_float_list(args.price_offset_list)

    summary_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_summaries, candidate_scenarios = evaluate_candidate(
            candidate,
            data_modules,
            horizon=int(args.horizon),
            device=device,
            action_cache_root=Path(args.action_cache_root),
            window_hours_list=window_hours_list,
            decision_lags=decision_lags,
            fill_buffer_bps_list=fill_buffer_bps_list,
            intensity_scales=intensity_scales,
            price_offsets=price_offsets,
            initial_cash=float(args.initial_cash),
            position_fraction=float(args.position_fraction),
            maker_fee=float(args.maker_fee),
            max_hold_hours=int(args.max_hold_hours) if args.max_hold_hours is not None else None,
            one_side_per_bar=not bool(args.no_one_side_per_bar),
            min_trade_count_mean=float(args.min_trade_count_mean),
            require_all_positive=bool(args.require_all_positive),
        )
        summary_rows.extend(candidate_summaries)
        scenario_rows.extend(candidate_scenarios)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["selection_score", "return_worst_pct", "return_mean_pct"], ascending=False
    )
    scenario_df = pd.DataFrame(scenario_rows).sort_values(
        ["config_name", "window_hours", "decision_lag_bars", "fill_buffer_bps", "start_state"]
    )

    summary_path = output_dir / "summary.csv"
    scenario_path = output_dir / "scenarios.csv"
    payload_path = output_dir / "summary.json"
    summary_df.to_csv(summary_path, index=False)
    scenario_df.to_csv(scenario_path, index=False)
    payload_path.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "args": {key: str(value) for key, value in vars(args).items()},
                "summary_rows": summary_df.to_dict(orient="records"),
                "scenario_rows": scenario_df.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Summary: {summary_path}")
    print(f"Scenarios: {scenario_path}")
    if not summary_df.empty:
        best = summary_df.iloc[0].to_dict()
        print(
            "Best:",
            best["candidate_label"],
            f"score={float(best['selection_score']):.3f}",
            f"worst_ret={float(best['return_worst_pct']):+.2f}%",
            f"mean_ret={float(best['return_mean_pct']):+.2f}%",
            f"worst_dd={float(best['max_drawdown_worst_pct']):.2f}%",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
