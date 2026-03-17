from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from llm_hourly_trader.gemini_wrapper import build_prompt
from llm_hourly_trader.providers import CacheMissError, call_llm
from src.allocation_refiner import refine_allocation, scale_allocations_to_gross_limit
from src.fees import get_fee_for_symbol
from src.symbol_utils import is_crypto_symbol

BASELINE_REFORECAST_MODE = "baseline"
GEMINI_REFORECAST_MODE = "gemini"
AMOUNT_REFORECAST_MODE = "amount_reforecasting"
GEMINI_AMOUNT_REFORECAST_MODE = "gemini+amount_reforecasting"

_VALID_REFORECAST_MODES = {
    BASELINE_REFORECAST_MODE,
    GEMINI_REFORECAST_MODE,
    AMOUNT_REFORECAST_MODE,
    GEMINI_AMOUNT_REFORECAST_MODE,
}


@dataclass(frozen=True)
class HourlyActionReforecastConfig:
    mode: str = BASELINE_REFORECAST_MODE
    llm_model: str = "gemini-3.1-flash-lite-preview"
    llm_prompt_variant: str = "position_context"
    llm_cache_only: bool = False
    llm_thinking_level: Optional[str] = None
    llm_reasoning_effort: Optional[str] = None
    llm_min_signal_amount_pct: float = 20.0
    llm_max_rows_per_symbol: Optional[int] = None
    max_gross_allocation: float = 1.0

    def normalized_mode(self) -> str:
        return normalize_reforecast_mode(self.mode)


def normalize_reforecast_mode(raw: Optional[str]) -> str:
    token = str(raw or BASELINE_REFORECAST_MODE).strip().lower().replace("_", "-").replace(" ", "")
    if token in {"", "baseline", "none", "off", "raw"}:
        return BASELINE_REFORECAST_MODE
    if token in {"gemini"}:
        return GEMINI_REFORECAST_MODE
    if token in {"amount", "amountreforecasting", "amount-reforecasting", "reforecast-amount"}:
        return AMOUNT_REFORECAST_MODE
    if token in {
        "gemini+amount",
        "gemini+amountreforecasting",
        "gemini+amount-reforecasting",
        "gemini-amount",
        "double",
        "double-reforecasting",
    }:
        return GEMINI_AMOUNT_REFORECAST_MODE
    raise ValueError(f"Unsupported reforecast mode: {raw!r}")


def parse_reforecast_modes(raw: Optional[str]) -> list[str]:
    if raw is None:
        return [BASELINE_REFORECAST_MODE]
    parts = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not parts:
        return [BASELINE_REFORECAST_MODE]
    seen: set[str] = set()
    ordered: list[str] = []
    for token in parts:
        mode = normalize_reforecast_mode(token)
        if mode not in seen:
            ordered.append(mode)
            seen.add(mode)
    return ordered


def apply_hourly_action_reforecasting(
    actions: pd.DataFrame,
    history_by_symbol: Mapping[str, pd.DataFrame],
    *,
    config: HourlyActionReforecastConfig,
    allow_short: bool = False,
    position_context_by_symbol: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> pd.DataFrame:
    normalized_mode = config.normalized_mode()
    if normalized_mode == BASELINE_REFORECAST_MODE or actions.empty:
        return actions.copy()

    out = _normalize_actions_frame(actions)
    history_frames = _prepare_history_frames(history_by_symbol)
    if normalized_mode in {GEMINI_REFORECAST_MODE, GEMINI_AMOUNT_REFORECAST_MODE}:
        out = _apply_gemini_reforecasting(
            out,
            history_frames,
            config=config,
            allow_short=allow_short,
            position_context_by_symbol=position_context_by_symbol,
        )
    if normalized_mode in {AMOUNT_REFORECAST_MODE, GEMINI_AMOUNT_REFORECAST_MODE}:
        out = _apply_amount_reforecasting(
            out,
            history_frames,
            config=config,
            allow_short=allow_short,
        )
    return out


def _normalize_actions_frame(actions: pd.DataFrame) -> pd.DataFrame:
    out = actions.copy()
    if "timestamp" not in out.columns or "symbol" not in out.columns:
        raise ValueError("actions must include timestamp and symbol columns")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["symbol"] = out["symbol"].astype(str).str.upper()
    for column in ("buy_price", "sell_price", "buy_amount", "sell_amount", "trade_amount", "allocation_fraction"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _prepare_history_frames(history_by_symbol: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    prepared: dict[str, pd.DataFrame] = {}
    for raw_symbol, raw_frame in history_by_symbol.items():
        symbol = str(raw_symbol or "").upper()
        if not symbol or raw_frame is None or raw_frame.empty:
            continue
        frame = raw_frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["symbol"] = frame.get("symbol", symbol)
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        if symbol not in set(frame["symbol"].unique().tolist()):
            frame["symbol"] = symbol
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        if "previous_forecast_error_pct" not in frame.columns:
            frame["previous_forecast_error_pct"] = _previous_forecast_error_pct(frame)
        prepared[symbol] = frame
    return prepared


def _previous_forecast_error_pct(frame: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    pred = pd.to_numeric(frame.get("predicted_close_p50_h1"), errors="coerce")
    reference = close.shift(1)
    predicted = pred.shift(1)
    actual_move_pct = (close / reference.replace(0.0, np.nan) - 1.0) * 100.0
    predicted_move_pct = (predicted / reference.replace(0.0, np.nan) - 1.0) * 100.0
    error = actual_move_pct - predicted_move_pct
    return error.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _apply_gemini_reforecasting(
    actions: pd.DataFrame,
    history_frames: Mapping[str, pd.DataFrame],
    *,
    config: HourlyActionReforecastConfig,
    allow_short: bool,
    position_context_by_symbol: Optional[Mapping[str, Mapping[str, object]]],
) -> pd.DataFrame:
    out = actions.copy()
    min_signal_amount = max(0.0, float(config.llm_min_signal_amount_pct))
    for symbol, group in out.groupby("symbol", sort=False):
        history = history_frames.get(symbol)
        if history is None or history.empty:
            continue
        candidate_indices = []
        for idx in group.index.tolist():
            row = out.loc[idx]
            if _signal_amount_pct(row) < min_signal_amount:
                continue
            candidate_indices.append(idx)
        if config.llm_max_rows_per_symbol is not None and config.llm_max_rows_per_symbol > 0:
            candidate_indices = candidate_indices[-int(config.llm_max_rows_per_symbol) :]

        for idx in candidate_indices:
            row = out.loc[idx]
            ts = pd.Timestamp(row["timestamp"])
            hist_slice = history[history["timestamp"] <= ts].tail(24)
            if len(hist_slice) < 8:
                continue

            current_price = _safe_float(hist_slice["close"].iloc[-1])
            if current_price <= 0.0:
                continue

            symbol_position = dict(position_context_by_symbol.get(symbol, {})) if position_context_by_symbol else {}
            qty = _safe_float(symbol_position.get("qty"), default=0.0)
            current_position = "long" if qty > 0.0 else "flat"
            prompt = build_prompt(
                symbol=symbol,
                history_rows=hist_slice[
                    [col for col in ("timestamp", "open", "high", "low", "close", "volume") if col in hist_slice.columns]
                ].to_dict("records"),
                forecast_1h=_extract_forecast(row, horizon=1),
                forecast_24h=_extract_forecast(row, horizon=24),
                current_position=current_position,
                cash=max(0.0, _safe_float(symbol_position.get("cash"), default=10_000.0)),
                equity=max(0.0, _safe_float(symbol_position.get("equity"), default=10_000.0)),
                allowed_directions=_allowed_directions_for_symbol(symbol, allow_short=allow_short),
                asset_class="crypto" if is_crypto_symbol(symbol) else "stock",
                maker_fee=float(get_fee_for_symbol(symbol)),
                variant=str(config.llm_prompt_variant),
                position_info=symbol_position,
            )
            try:
                plan = call_llm(
                    prompt,
                    model=str(config.llm_model),
                    thinking_level=config.llm_thinking_level,
                    reasoning_effort=config.llm_reasoning_effort,
                    cache_only=bool(config.llm_cache_only),
                )
            except CacheMissError:
                continue
            except Exception:
                continue

            base_buy_amount = float(_safe_float(row.get("buy_amount"), default=0.0))
            base_sell_amount = float(_safe_float(row.get("sell_amount"), default=0.0))
            is_short_capable = allow_short and not is_crypto_symbol(symbol)

            if plan.direction == "hold":
                if base_buy_amount >= base_sell_amount:
                    out.at[idx, "buy_amount"] = 0.0
                    if "trade_amount" in out.columns:
                        out.at[idx, "trade_amount"] = float(max(out.at[idx, "buy_amount"], out.at[idx, "sell_amount"]))
                    if "allocation_fraction" in out.columns:
                        out.at[idx, "allocation_fraction"] = 0.0
                elif is_short_capable and base_sell_amount > base_buy_amount:
                    out.at[idx, "sell_amount"] = 0.0
                    if "trade_amount" in out.columns:
                        out.at[idx, "trade_amount"] = float(max(out.at[idx, "buy_amount"], out.at[idx, "sell_amount"]))
                continue

            if plan.direction == "long":
                if plan.buy_price > 0.0:
                    out.at[idx, "buy_price"] = float(plan.buy_price)
                if plan.sell_price > 0.0:
                    sell_price = max(float(plan.sell_price), float(out.at[idx, "buy_price"]) * 1.001)
                    out.at[idx, "sell_price"] = sell_price
                continue

            if plan.direction == "short" and is_short_capable:
                if plan.sell_price > 0.0:
                    out.at[idx, "sell_price"] = float(plan.sell_price)
                if plan.buy_price > 0.0:
                    out.at[idx, "buy_price"] = float(plan.buy_price)
                if base_sell_amount <= base_buy_amount:
                    out.at[idx, "sell_amount"] = float(max(base_sell_amount, base_buy_amount))
                    out.at[idx, "buy_amount"] = 0.0
                    if "trade_amount" in out.columns:
                        out.at[idx, "trade_amount"] = float(max(out.at[idx, "buy_amount"], out.at[idx, "sell_amount"]))
    return out


def _apply_amount_reforecasting(
    actions: pd.DataFrame,
    history_frames: Mapping[str, pd.DataFrame],
    *,
    config: HourlyActionReforecastConfig,
    allow_short: bool,
) -> pd.DataFrame:
    out = actions.copy()
    context = _build_context_frame(history_frames)
    if context.empty:
        return out
    joined = out.merge(context, on=["timestamp", "symbol"], how="left", suffixes=("", "_ctx"))
    if joined.empty:
        return out

    previous_allocations: dict[str, float] = {}
    rows_by_key = {(pd.Timestamp(row.timestamp), str(row.symbol).upper()): idx for idx, row in out.iterrows()}

    for ts, group in joined.groupby("timestamp", sort=True):
        signed_targets: dict[str, float] = {}
        side_meta: dict[str, dict[str, float]] = {}

        for row in group.itertuples(index=False):
            symbol = str(row.symbol).upper()
            current_price = _safe_float(getattr(row, "close", 0.0))
            if current_price <= 0.0:
                signed_targets[symbol] = 0.0
                side_meta[symbol] = {"sell_amount": _safe_float(getattr(row, "sell_amount", 0.0))}
                continue

            buy_amount_pct = float(np.clip(_safe_float(getattr(row, "buy_amount", 0.0)), 0.0, 100.0))
            sell_amount_pct = float(np.clip(_safe_float(getattr(row, "sell_amount", 0.0)), 0.0, 100.0))
            allocation_fraction = _safe_float(
                getattr(row, "allocation_fraction", None),
                default=max(buy_amount_pct, sell_amount_pct) / 100.0,
            )
            allocation_fraction = float(np.clip(allocation_fraction, 0.0, 1.0))

            weighted_delta = _weighted_delta(row)
            agreement = float(np.clip(abs(_safe_float(getattr(row, "forecast_agreement", 0.0))), 0.0, 1.0))
            forecast_confidence = _forecast_confidence_unit(row)
            long_edge = _entry_edge_pct(row, current_price=current_price)
            previous_error = _safe_float(getattr(row, "previous_forecast_error_pct", 0.0))

            short_capable = allow_short and not is_crypto_symbol(symbol)
            prefer_short = short_capable and sell_amount_pct > (buy_amount_pct + 1e-9) and weighted_delta < 0.0
            direction = "short" if prefer_short else "long"
            direction_amount = sell_amount_pct if prefer_short else buy_amount_pct
            base_fraction = max(allocation_fraction, direction_amount / 100.0)

            if base_fraction <= 0.0 and long_edge <= 0.0 and weighted_delta <= 0.0 and not prefer_short:
                target_allocation = 0.0
            else:
                rl_confidence = float(
                    np.clip(
                        0.15 + (0.45 * max(base_fraction, 0.0)) + (0.20 * agreement) + (0.20 * forecast_confidence),
                        0.0,
                        1.0,
                    )
                )
                if prefer_short:
                    rl_gap = (-weighted_delta * 30.0) + max(-long_edge * 60.0, 0.0)
                else:
                    rl_gap = (weighted_delta * 30.0) + max(long_edge * 60.0, 0.0)
                refinement = refine_allocation(
                    asset_class="crypto" if is_crypto_symbol(symbol) else "stock",
                    rl_direction=direction,
                    rl_allocation_pct=base_fraction,
                    rl_confidence=rl_confidence,
                    rl_logit_gap=rl_gap,
                    current_allocation=float(previous_allocations.get(symbol, 0.0)),
                    current_price=current_price,
                    forecast_1h=_extract_forecast(row, horizon=1),
                    forecast_24h=_extract_forecast(row, horizon=24),
                    previous_forecast_error=previous_error,
                )
                target_allocation = float(refinement.target_allocation)

            signed_targets[symbol] = target_allocation
            side_meta[symbol] = {
                "sell_amount": sell_amount_pct,
                "weighted_delta": weighted_delta,
                "long_edge": long_edge,
            }

        scaled_targets = scale_allocations_to_gross_limit(
            signed_targets,
            max_gross=max(float(config.max_gross_allocation), 0.0) or 1.0,
        )

        for symbol, signed_target in scaled_targets.items():
            out_idx = rows_by_key.get((pd.Timestamp(ts), symbol))
            if out_idx is None:
                continue
            abs_target = float(np.clip(abs(float(signed_target)), 0.0, 1.0))
            new_buy_amount = float(np.clip(abs_target * 100.0 if signed_target >= 0.0 else 0.0, 0.0, 100.0))
            base_sell_amount = float(side_meta.get(symbol, {}).get("sell_amount", 0.0))
            weighted_delta = float(side_meta.get(symbol, {}).get("weighted_delta", 0.0))
            long_edge = float(side_meta.get(symbol, {}).get("long_edge", 0.0))
            bearish_pressure = float(np.clip(max(-weighted_delta * 40.0, 0.0) + max(-long_edge * 80.0, 0.0), 0.0, 1.5))
            bullish_relief = float(np.clip(max(weighted_delta * 25.0, 0.0) + max(long_edge * 60.0, 0.0), 0.0, 1.0))
            if signed_target < 0.0:
                new_sell_amount = float(np.clip(abs_target * 100.0, 0.0, 100.0))
                new_buy_amount = 0.0
            else:
                new_sell_amount = float(np.clip(base_sell_amount * (1.0 + 0.8 * bearish_pressure), 0.0, 100.0))
                if bullish_relief > 0.0:
                    new_sell_amount = float(np.clip(new_sell_amount * max(0.25, 1.0 - (0.35 * bullish_relief)), 0.0, 100.0))

            out.at[out_idx, "buy_amount"] = new_buy_amount
            out.at[out_idx, "sell_amount"] = new_sell_amount
            if "trade_amount" in out.columns:
                out.at[out_idx, "trade_amount"] = float(max(new_buy_amount, new_sell_amount))
            if "allocation_fraction" in out.columns:
                out.at[out_idx, "allocation_fraction"] = abs_target
            previous_allocations[symbol] = float(signed_target)

    return out


def _build_context_frame(history_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    context_frames = []
    for symbol, frame in history_frames.items():
        columns = [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "previous_forecast_error_pct",
        ]
        columns.extend(col for col in frame.columns if col.startswith("predicted_"))
        columns.extend(col for col in frame.columns if col.startswith("chronos_"))
        columns.extend(col for col in frame.columns if col.startswith("forecast_"))
        unique_columns = [col for col in dict.fromkeys(columns) if col in frame.columns]
        if not unique_columns:
            continue
        context_frames.append(frame[unique_columns].assign(symbol=symbol))
    if not context_frames:
        return pd.DataFrame(columns=["timestamp", "symbol"])
    context = pd.concat(context_frames, ignore_index=True)
    context["timestamp"] = pd.to_datetime(context["timestamp"], utc=True)
    context["symbol"] = context["symbol"].astype(str).str.upper()
    return context.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _signal_amount_pct(row: Mapping[str, object] | pd.Series) -> float:
    values = [
        _safe_float(row.get("buy_amount", 0.0)),
        _safe_float(row.get("sell_amount", 0.0)),
        _safe_float(row.get("trade_amount", 0.0)),
        _safe_float(row.get("allocation_fraction", 0.0)) * 100.0,
    ]
    return max(values)


def _extract_forecast(row: Mapping[str, object] | pd.Series | object, *, horizon: int) -> Optional[dict[str, float]]:
    suffix = f"_h{int(horizon)}"
    fields = (
        "predicted_close_p10",
        "predicted_close_p50",
        "predicted_close_p90",
        "predicted_high_p50",
        "predicted_low_p50",
    )
    result: dict[str, float] = {}
    for field in fields:
        value = _row_value(row, f"{field}{suffix}")
        if value is None:
            continue
        numeric = _safe_float(value, default=0.0)
        if numeric <= 0.0:
            continue
        result[field] = numeric
    return result or None


def _allowed_directions_for_symbol(symbol: str, *, allow_short: bool) -> list[str]:
    if allow_short and not is_crypto_symbol(symbol):
        return ["long", "short"]
    return ["long"]


def _forecast_confidence_unit(row: object) -> float:
    candidates = (
        _safe_float(_row_value(row, "forecast_confidence_mean"), default=0.0),
        _safe_float(_row_value(row, "forecast_confidence_h1"), default=0.0),
        _safe_float(_row_value(row, "forecast_confidence_h24"), default=0.0),
    )
    raw = max(candidates)
    if raw <= 0.0:
        return 0.0
    return float(np.clip(np.log1p(raw) / 6.0, 0.0, 1.0))


def _weighted_delta(row: object) -> float:
    direct = _row_value(row, "forecast_weighted_delta")
    if direct is not None:
        return _safe_float(direct, default=0.0)
    values = []
    for weight, column in ((0.65, "chronos_close_delta_h1"), (0.35, "chronos_close_delta_h24")):
        value = _row_value(row, column)
        if value is None:
            continue
        values.append((weight, _safe_float(value, default=0.0)))
    if not values:
        return 0.0
    total_weight = sum(weight for weight, _ in values)
    return float(sum(weight * value for weight, value in values) / total_weight)


def _entry_edge_pct(row: object, *, current_price: float) -> float:
    buy_price = _safe_float(_row_value(row, "buy_price"), default=0.0)
    if buy_price <= 0.0 or current_price <= 0.0:
        return 0.0
    forecast_candidates = []
    for field in ("predicted_close_p50_h1", "predicted_close_p50_h24", "predicted_high_p50_h1", "predicted_high_p50_h24"):
        value = _row_value(row, field)
        if value is None:
            continue
        numeric = _safe_float(value, default=0.0)
        if numeric > 0.0:
            forecast_candidates.append(numeric)
    if not forecast_candidates:
        return 0.0
    target_price = max(max(forecast_candidates), current_price)
    return float((target_price / buy_price) - 1.0)


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return float(result)


def _row_value(row: Mapping[str, object] | pd.Series | object, name: str) -> object:
    if isinstance(row, Mapping):
        return row.get(name)
    if isinstance(row, pd.Series):
        return row.get(name)
    return getattr(row, name, None)


__all__ = [
    "AMOUNT_REFORECAST_MODE",
    "BASELINE_REFORECAST_MODE",
    "GEMINI_AMOUNT_REFORECAST_MODE",
    "GEMINI_REFORECAST_MODE",
    "HourlyActionReforecastConfig",
    "apply_hourly_action_reforecasting",
    "normalize_reforecast_mode",
    "parse_reforecast_modes",
]
