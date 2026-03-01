#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.trading.requests import GetOrdersRequest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import alpaca_wrapper


def _normalize_symbol(symbol: str) -> tuple[str, str]:
    raw = symbol.strip().upper()
    if "/" in raw:
        compact = raw.replace("/", "")
        return raw, compact
    return f"{raw[:-3]}/{raw[-3:]}", raw


def _load_sim_fills(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df

    if "timestamp" not in df.columns:
        raise ValueError(f"{path} must contain a timestamp column.")
    if "side" not in df.columns:
        raise ValueError(f"{path} must contain a side column.")

    # Support both hourly fills.csv and selector_trades.csv schemas.
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
    if "kind" in df.columns:
        out["kind"] = df["kind"].astype(str)
    return out


def _fetch_live_orders(symbol_slash: str, hours: float) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    after = now - timedelta(hours=float(hours))
    req = GetOrdersRequest(
        status="all",
        symbols=[symbol_slash],
        after=after,
        direction="asc",
        limit=500,
    )
    orders = alpaca_wrapper.alpaca_api.get_orders(filter=req)
    rows = []
    for order in orders:
        side = getattr(getattr(order, "side", None), "value", str(getattr(order, "side", ""))).lower()
        status = getattr(getattr(order, "status", None), "value", str(getattr(order, "status", ""))).lower()
        created_at = pd.to_datetime(getattr(order, "created_at", None), utc=True)
        filled_at_raw = getattr(order, "filled_at", None)
        filled_at = pd.to_datetime(filled_at_raw, utc=True) if filled_at_raw else pd.NaT
        rows.append(
            {
                "created_at": created_at,
                "filled_at": filled_at,
                "side": side,
                "status": status,
                "qty": float(getattr(order, "qty", 0.0) or 0.0),
                "filled_qty": float(getattr(order, "filled_qty", 0.0) or 0.0),
                "limit_price": float(getattr(order, "limit_price", 0.0) or 0.0),
                "filled_price": float(getattr(order, "filled_avg_price", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def _build_report(sim: pd.DataFrame, live: pd.DataFrame, hours: float) -> dict:
    live_filled = live[
        (live["status"] == "filled") & (live["filled_qty"] > 0.0) & (live["filled_price"] > 0.0) & live["filled_at"].notna()
    ].copy()
    live_open = live[live["status"].isin(["new", "accepted", "partially_filled"])].copy()

    live_filled["hour"] = live_filled["filled_at"].dt.floor("h")
    sim["hour"] = sim["timestamp"].dt.floor("h")

    live_hourly = live_filled.groupby(["hour", "side"]).size().rename("live_count").reset_index()
    sim_hourly = sim.groupby(["hour", "side"]).size().rename("sim_count").reset_index()
    hourly_compare = (
        live_hourly.merge(sim_hourly, on=["hour", "side"], how="outer")
        .fillna(0)
        .sort_values(["hour", "side"])
        .reset_index(drop=True)
    )
    hourly_compare["delta"] = hourly_compare["sim_count"] - hourly_compare["live_count"]

    now = datetime.now(timezone.utc)
    summary = {
        "window_start_utc": (now - timedelta(hours=float(hours))).isoformat(),
        "window_end_utc": now.isoformat(),
        "live_orders_total": int(len(live)),
        "live_filled_total": int(len(live_filled)),
        "live_open_total": int(len(live_open)),
        "live_filled_buys": int((live_filled["side"] == "buy").sum()),
        "live_filled_sells": int((live_filled["side"] == "sell").sum()),
        "live_open_buys": int((live_open["side"] == "buy").sum()),
        "live_open_sells": int((live_open["side"] == "sell").sum()),
        "sim_fills_total": int(len(sim)),
        "sim_filled_buys": int((sim["side"] == "buy").sum()),
        "sim_filled_sells": int((sim["side"] == "sell").sum()),
        "hourly_abs_delta_total": float(hourly_compare["delta"].abs().sum()),
        "exact_hourly_side_matches": int((hourly_compare["delta"] == 0).sum()),
    }

    return {
        "summary": summary,
        "live_filled_orders": live_filled.to_dict(orient="records"),
        "live_open_orders": live_open.to_dict(orient="records"),
        "sim_fills": sim.to_dict(orient="records"),
        "hourly_side_compare": hourly_compare.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ETH live orders vs simulator fills over a trailing window.")
    parser.add_argument("--sim-fills", required=True, type=Path, help="Path to simulator fills CSV.")
    parser.add_argument("--hours", type=float, default=24.0, help="Trailing comparison window in hours.")
    parser.add_argument("--symbol", default="ETHUSD", help="Trading symbol (ETHUSD or ETH/USD).")
    parser.add_argument("--output", type=Path, required=True, help="Path to write JSON report.")
    args = parser.parse_args()

    symbol_slash, symbol_compact = _normalize_symbol(args.symbol)
    sim = _load_sim_fills(args.sim_fills)
    if sim.empty:
        raise RuntimeError(f"Simulator fills are empty: {args.sim_fills}")

    live = _fetch_live_orders(symbol_slash, hours=args.hours)
    if live.empty:
        raise RuntimeError(f"No live orders returned for {symbol_slash} in the last {args.hours}h.")

    # Keep only requested symbol format variations.
    if "symbol" in sim.columns:
        sim = sim[sim["symbol"].astype(str).str.upper().str.replace("/", "", regex=False) == symbol_compact]

    report = _build_report(sim=sim, live=live, hours=args.hours)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
