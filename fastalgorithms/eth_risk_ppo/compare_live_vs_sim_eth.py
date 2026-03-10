#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import alpaca_wrapper

OPEN_ORDER_STATUSES = {"new", "accepted", "partially_filled", "pending_new", "accepted_for_bidding"}


def _normalize_symbol(symbol: str) -> tuple[str, str]:
    raw = symbol.strip().upper()
    if "/" in raw:
        compact = raw.replace("/", "")
        return raw, compact
    return f"{raw[:-3]}/{raw[-3:]}", raw


def _coerce_ts(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp value: {value!r}")
    return ts


def _resolve_window(
    *,
    hours: Optional[float],
    window_start: Optional[object],
    window_end: Optional[object],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = _coerce_ts(window_start) if window_start is not None else None
    end = _coerce_ts(window_end) if window_end is not None else None

    if start is None and end is None:
        if hours is None:
            raise ValueError("hours is required when window_start/window_end are omitted.")
        end = pd.Timestamp(datetime.now(timezone.utc))
        start = end - pd.Timedelta(hours=float(hours))
    elif start is None:
        if hours is None:
            raise ValueError("hours is required when only window_end is provided.")
        start = end - pd.Timedelta(hours=float(hours))
    elif end is None:
        if hours is None:
            raise ValueError("hours is required when only window_start is provided.")
        end = start + pd.Timedelta(hours=float(hours))

    if end <= start:
        raise ValueError(f"window_end must be after window_start: start={start}, end={end}")
    return start, end


def _load_sim_fills(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df

    if "timestamp" not in df.columns:
        raise ValueError(f"{path} must contain a timestamp column.")
    if "side" not in df.columns:
        raise ValueError(f"{path} must contain a side column.")

    qty_col = "quantity" if "quantity" in df.columns else "qty" if "qty" in df.columns else None
    if qty_col is None:
        raise ValueError(f"{path} must contain quantity or qty.")

    price_col = "price" if "price" in df.columns else "limit_price" if "limit_price" in df.columns else None
    if price_col is None:
        raise ValueError(f"{path} must contain price or limit_price.")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["timestamp"], utc=True),
            "side": df["side"].astype(str).str.lower(),
            "quantity": pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0),
            "price": pd.to_numeric(df[price_col], errors="coerce").fillna(0.0),
        }
    )
    if "symbol" in df.columns:
        out["symbol"] = df["symbol"].astype(str).str.upper().str.replace("/", "", regex=False)
    if "kind" in df.columns:
        out["kind"] = df["kind"].astype(str)
    return out.dropna(subset=["timestamp"]).reset_index(drop=True)


def _fetch_live_orders(
    *,
    symbol_slash: str,
    fetch_after: pd.Timestamp,
    fetch_until: pd.Timestamp,
) -> pd.DataFrame:
    from alpaca.trading.requests import GetOrdersRequest

    req = GetOrdersRequest(
        status="all",
        symbols=[symbol_slash],
        after=fetch_after.to_pydatetime(),
        until=fetch_until.to_pydatetime(),
        direction="asc",
        limit=500,
    )
    orders = alpaca_wrapper.alpaca_api.get_orders(filter=req)
    rows = []
    for order in orders:
        side = getattr(getattr(order, "side", None), "value", str(getattr(order, "side", ""))).lower()
        status = getattr(getattr(order, "status", None), "value", str(getattr(order, "status", ""))).lower()
        created_at = pd.to_datetime(getattr(order, "created_at", None), utc=True, errors="coerce")
        filled_at = pd.to_datetime(getattr(order, "filled_at", None), utc=True, errors="coerce")
        canceled_at = pd.to_datetime(getattr(order, "canceled_at", None), utc=True, errors="coerce")
        rows.append(
            {
                "id": str(getattr(order, "id", "")),
                "symbol": str(getattr(order, "symbol", symbol_slash)),
                "created_at": created_at,
                "filled_at": filled_at,
                "canceled_at": canceled_at,
                "side": side,
                "status": status,
                "qty": float(getattr(order, "qty", 0.0) or 0.0),
                "filled_qty": float(getattr(order, "filled_qty", 0.0) or 0.0),
                "limit_price": float(getattr(order, "limit_price", 0.0) or 0.0),
                "filled_price": float(getattr(order, "filled_avg_price", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def _filter_live_orders_to_window(
    live: pd.DataFrame,
    *,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    if live.empty:
        return live

    created_at = pd.to_datetime(live["created_at"], utc=True, errors="coerce")
    filled_at = pd.to_datetime(live["filled_at"], utc=True, errors="coerce")
    canceled_at = pd.to_datetime(live["canceled_at"], utc=True, errors="coerce")
    status = live["status"].astype(str).str.lower()

    created_in_window = created_at.between(window_start, window_end, inclusive="both")
    filled_in_window = filled_at.between(window_start, window_end, inclusive="both")
    canceled_in_window = canceled_at.between(window_start, window_end, inclusive="both")
    still_open_at_window_end = (
        status.isin(OPEN_ORDER_STATUSES)
        & created_at.le(window_end)
        & (canceled_at.isna() | canceled_at.gt(window_end))
    )

    relevant = created_in_window | filled_in_window | canceled_in_window | still_open_at_window_end
    return live.loc[relevant].copy().reset_index(drop=True)


def _match_fills(live_filled: pd.DataFrame, sim: pd.DataFrame, *, max_delay_minutes: float = 90.0) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    if live_filled.empty or sim.empty:
        return [], live_filled.copy(), sim.copy()

    sim_work = sim.copy().reset_index(drop=True)
    sim_work["_matched"] = False
    matches: list[dict] = []

    for _, live_row in live_filled.sort_values("filled_at").iterrows():
        live_ts = pd.to_datetime(live_row["filled_at"], utc=True)
        live_side = str(live_row["side"]).lower()
        candidates = sim_work[
            (~sim_work["_matched"])
            & (sim_work["side"].astype(str).str.lower() == live_side)
            & (abs((pd.to_datetime(sim_work["timestamp"], utc=True) - live_ts).dt.total_seconds()) <= max_delay_minutes * 60.0)
        ]
        if candidates.empty:
            continue

        diffs = abs((pd.to_datetime(candidates["timestamp"], utc=True) - live_ts).dt.total_seconds())
        best_idx = diffs.idxmin()
        best = sim_work.loc[best_idx]
        sim_work.at[best_idx, "_matched"] = True

        live_price = float(live_row["filled_price"])
        sim_price = float(best["price"])
        price_diff_bps = 0.0
        if live_price > 0:
            price_diff_bps = (sim_price - live_price) / live_price * 10_000.0

        matches.append(
            {
                "live_id": live_row.get("id"),
                "side": live_side,
                "live_ts": live_ts,
                "sim_ts": best["timestamp"],
                "minutes_delta": abs((pd.to_datetime(best["timestamp"], utc=True) - live_ts).total_seconds()) / 60.0,
                "live_qty": float(live_row["filled_qty"]),
                "sim_qty": float(best["quantity"]),
                "live_price": live_price,
                "sim_price": sim_price,
                "price_diff_bps": price_diff_bps,
            }
        )

    unmatched_live = live_filled[~live_filled["id"].isin({m["live_id"] for m in matches if m.get("live_id")})].copy()
    unmatched_sim = sim_work[~sim_work["_matched"]].drop(columns="_matched").copy()
    return matches, unmatched_live.reset_index(drop=True), unmatched_sim.reset_index(drop=True)


def _build_report(
    *,
    sim: pd.DataFrame,
    live: pd.DataFrame,
    window_start: Optional[object] = None,
    window_end: Optional[object] = None,
    hours: Optional[float] = None,
    symbol: Optional[str] = None,
) -> dict:
    resolved_start, resolved_end = _resolve_window(hours=hours, window_start=window_start, window_end=window_end)

    sim_window = sim.copy()
    if not sim_window.empty:
        sim_window = sim_window[
            pd.to_datetime(sim_window["timestamp"], utc=True).between(resolved_start, resolved_end, inclusive="both")
        ].copy()

    live_fetched_total = int(len(live))
    live_window = _filter_live_orders_to_window(live, window_start=resolved_start, window_end=resolved_end)

    live_filled = live_window[
        (live_window["filled_qty"] > 0.0)
        & (live_window["filled_price"] > 0.0)
        & pd.to_datetime(live_window["filled_at"], utc=True, errors="coerce").between(
            resolved_start, resolved_end, inclusive="both"
        )
    ].copy()
    live_open = live_window[
        live_window["status"].astype(str).str.lower().isin(OPEN_ORDER_STATUSES)
    ].copy()

    if not live_filled.empty:
        live_filled["hour"] = pd.to_datetime(live_filled["filled_at"], utc=True).dt.floor("h")
    else:
        live_filled["hour"] = pd.Series(dtype="datetime64[ns, UTC]")
    if not sim_window.empty:
        sim_window["hour"] = pd.to_datetime(sim_window["timestamp"], utc=True).dt.floor("h")
    else:
        sim_window["hour"] = pd.Series(dtype="datetime64[ns, UTC]")

    live_hourly = live_filled.groupby(["hour", "side"]).size().rename("live_count").reset_index()
    sim_hourly = sim_window.groupby(["hour", "side"]).size().rename("sim_count").reset_index()
    hourly_compare = (
        live_hourly.merge(sim_hourly, on=["hour", "side"], how="outer").fillna(0).sort_values(["hour", "side"]).reset_index(drop=True)
    )
    if not hourly_compare.empty:
        hourly_compare["delta"] = hourly_compare["sim_count"] - hourly_compare["live_count"]
    else:
        hourly_compare = pd.DataFrame(columns=["hour", "side", "live_count", "sim_count", "delta"])

    matches, unmatched_live, unmatched_sim = _match_fills(live_filled, sim_window)
    mean_abs_price_diff_bps = 0.0
    if matches:
        mean_abs_price_diff_bps = float(pd.Series([abs(m["price_diff_bps"]) for m in matches]).mean())

    summary = {
        "symbol": symbol,
        "window_start_utc": resolved_start.isoformat(),
        "window_end_utc": resolved_end.isoformat(),
        "window_hours": (resolved_end - resolved_start).total_seconds() / 3600.0,
        "live_orders_fetched_total": live_fetched_total,
        "live_orders_total": int(len(live_window)),
        "live_filled_total": int(len(live_filled)),
        "live_open_total": int(len(live_open)),
        "live_filled_buys": int((live_filled["side"] == "buy").sum()),
        "live_filled_sells": int((live_filled["side"] == "sell").sum()),
        "live_open_buys": int((live_open["side"] == "buy").sum()),
        "live_open_sells": int((live_open["side"] == "sell").sum()),
        "sim_fills_total": int(len(sim_window)),
        "sim_filled_buys": int((sim_window["side"] == "buy").sum()),
        "sim_filled_sells": int((sim_window["side"] == "sell").sum()),
        "hourly_abs_delta_total": float(hourly_compare["delta"].abs().sum()) if not hourly_compare.empty else 0.0,
        "exact_hourly_side_matches": int((hourly_compare["delta"] == 0).sum()) if not hourly_compare.empty else 0,
        "matched_fill_count": int(len(matches)),
        "unmatched_live_fill_count": int(len(unmatched_live)),
        "unmatched_sim_fill_count": int(len(unmatched_sim)),
        "matched_fill_mean_abs_price_diff_bps": mean_abs_price_diff_bps,
    }

    return {
        "summary": summary,
        "live_filled_orders": live_filled.to_dict(orient="records"),
        "live_open_orders": live_open.to_dict(orient="records"),
        "sim_fills": sim_window.to_dict(orient="records"),
        "hourly_side_compare": hourly_compare.to_dict(orient="records"),
        "fill_matches": matches,
        "unmatched_live_fills": unmatched_live.to_dict(orient="records"),
        "unmatched_sim_fills": unmatched_sim.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ETH live orders vs simulator fills over a trailing or explicit window.")
    parser.add_argument("--sim-fills", required=True, type=Path, help="Path to simulator fills CSV.")
    parser.add_argument("--hours", type=float, default=24.0, help="Comparison window in hours.")
    parser.add_argument("--symbol", default="ETHUSD", help="Trading symbol (ETHUSD or ETH/USD).")
    parser.add_argument("--output", type=Path, required=True, help="Path to write JSON report.")
    parser.add_argument("--window-start", default=None, help="Explicit UTC window start; overrides trailing start when combined with --window-end.")
    parser.add_argument("--window-end", default=None, help="Explicit UTC window end; defaults to now if omitted.")
    parser.add_argument(
        "--prefetch-hours",
        type=float,
        default=None,
        help="Extra hours of live-order lookback before window_start so pre-window fills/open orders are included.",
    )
    args = parser.parse_args()

    symbol_slash, symbol_compact = _normalize_symbol(args.symbol)
    resolved_start, resolved_end = _resolve_window(
        hours=args.hours,
        window_start=args.window_start,
        window_end=args.window_end,
    )

    sim = _load_sim_fills(args.sim_fills)
    if sim.empty:
        raise RuntimeError(f"Simulator fills are empty: {args.sim_fills}")

    if "symbol" in sim.columns:
        sim = sim[sim["symbol"].astype(str).str.upper().str.replace("/", "", regex=False) == symbol_compact].copy()

    prefetch_hours = float(args.prefetch_hours) if args.prefetch_hours is not None else max(float(args.hours or 0.0), 72.0)
    fetch_after = resolved_start - pd.Timedelta(hours=prefetch_hours)
    live = _fetch_live_orders(symbol_slash=symbol_slash, fetch_after=fetch_after, fetch_until=resolved_end)

    report = _build_report(
        sim=sim,
        live=live,
        window_start=resolved_start,
        window_end=resolved_end,
        symbol=symbol_compact,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
