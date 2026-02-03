#!/usr/bin/env python3
"""Compare strategy trade PnL vs a no-trade baseline on Alpaca.

Computes PnL from executed fills in a time window (including fees) and
estimates what account equity would have been with no trades.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import alpaca_wrapper  # noqa: E402
from src.pnl_utils import compute_trade_pnl  # noqa: E402
from src.symbol_utils import is_crypto_symbol  # noqa: E402


def _parse_dt(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        # Try date-only
        dt = datetime.strptime(value, "%Y-%m-%d")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").upper()


def _activity_to_dict(activity: object) -> dict:
    if hasattr(activity, "to_dict"):
        return activity.to_dict()
    if isinstance(activity, dict):
        return activity
    return activity.__dict__


def _extract_items(response: object) -> List[dict]:
    if isinstance(response, list):
        return [
            _activity_to_dict(item) for item in response
        ]
    if isinstance(response, dict):
        for key in ("account_activities", "activities", "items", "data"):
            items = response.get(key)
            if items:
                return [_activity_to_dict(item) for item in items]
    return []


def _fetch_activities_after_until(
    *,
    client,
    after: datetime,
    until: datetime,
    activity_types: List[str],
    direction: str,
    page_size: int,
    max_pages: int,
) -> List[dict]:
    results: List[dict] = []
    page_token: Optional[str] = None

    for _ in range(max_pages):
        params: Dict[str, str] = {
            "direction": direction,
            "page_size": str(page_size),
            "activity_types": ",".join(activity_types),
            "after": after.isoformat(),
            "until": until.isoformat(),
        }
        if page_token:
            params["page_token"] = page_token

        response = client._request("GET", "/account/activities", data=params)
        items = _extract_items(response)
        results.extend(items)

        if isinstance(response, dict):
            page_token = (
                response.get("next_page_token")
                or response.get("next_page")
                or response.get("next")
            )
        else:
            page_token = None

        if not page_token or not items:
            break

    return results


def _fetch_activities_by_date(
    *,
    client,
    start: datetime,
    end: datetime,
    activity_types: List[str],
    direction: str,
) -> List[dict]:
    results: List[dict] = []
    day = start.date()
    end_date = end.date()
    while day <= end_date:
        params: Dict[str, str] = {
            "direction": direction,
            "activity_types": ",".join(activity_types),
            "date": day.strftime("%Y-%m-%d"),
        }
        response = client._request("GET", "/account/activities", data=params)
        items = _extract_items(response)
        results.extend(items)
        day += timedelta(days=1)
    return results


def fetch_account_activities(
    *,
    client,
    start: datetime,
    end: datetime,
    activity_types: List[str],
    direction: str,
    page_size: int,
    max_pages: int,
) -> List[dict]:
    try:
        return _fetch_activities_after_until(
            client=client,
            after=start,
            until=end,
            activity_types=activity_types,
            direction=direction,
            page_size=page_size,
            max_pages=max_pages,
        )
    except Exception:
        return _fetch_activities_by_date(
            client=client,
            start=start,
            end=end,
            activity_types=activity_types,
            direction=direction,
        )


def _filter_fills(
    activities: Iterable[dict],
    start: datetime,
    end: datetime,
) -> List[dict]:
    fills: List[dict] = []
    for item in activities:
        if item.get("activity_type") != "FILL":
            continue
        ts_raw = item.get("transaction_time") or item.get("transact_time")
        if not ts_raw:
            continue
        ts = _parse_dt(str(ts_raw))
        if ts < start or ts > end:
            continue
        symbol = _normalize_symbol(str(item.get("symbol", "")))
        side = str(item.get("side", "")).lower()
        price = float(item.get("price", 0) or 0)
        qty = float(item.get("qty", 0) or 0)
        if not symbol or price <= 0 or qty <= 0:
            continue
        fills.append(
            {
                "symbol": symbol,
                "side": side,
                "price": price,
                "qty": qty,
                "timestamp": ts,
            }
        )
    fills.sort(key=lambda x: x["timestamp"])
    return fills


def _current_prices() -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for pos in alpaca_wrapper.get_all_positions():
        symbol = _normalize_symbol(str(pos.symbol))
        try:
            current_price = float(pos.current_price)
        except (TypeError, ValueError):
            continue
        prices[symbol] = current_price
    return prices


def _fee_rates(symbols: Iterable[str], crypto_fee_bps: float, stock_fee_bps: float) -> Dict[str, float]:
    rates = {}
    for symbol in symbols:
        if is_crypto_symbol(symbol):
            rates[symbol] = crypto_fee_bps / 10000.0
        else:
            rates[symbol] = stock_fee_bps / 10000.0
    return rates


def _try_portfolio_history(start: datetime, end: datetime) -> Optional[dict]:
    client = alpaca_wrapper.alpaca_api
    params = {
        "timeframe": "1D",
        "date_start": _to_date_str(start),
        "date_end": _to_date_str(end),
    }
    try:
        response = client._request("GET", "/account/portfolio/history", data=params)
        if isinstance(response, dict) and response.get("equity"):
            return response
    except Exception:
        return None
    return None


def _compute_equity_change(history: dict) -> Optional[Tuple[float, float]]:
    equity = history.get("equity")
    if not equity:
        return None
    start_equity = float(equity[0])
    end_equity = float(equity[-1])
    return start_equity, end_equity


def _write_report(
    *,
    report_dir: Path,
    summary: dict,
    per_symbol_rows: List[dict],
) -> Tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"pnl_report_{timestamp}.json"
    csv_path = report_dir / f"pnl_report_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if per_symbol_rows:
        fieldnames = list(per_symbol_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_symbol_rows)

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Alpaca trade PnL vs baseline")
    parser.add_argument("--start", type=str, default=None, help="Start time (ISO or YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End time (ISO or YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=7, help="Lookback days if no start provided")
    parser.add_argument("--crypto-fee-bps", type=float, default=8.0, help="Crypto fee in bps")
    parser.add_argument("--stock-fee-bps", type=float, default=2.0, help="Stock fee in bps")
    parser.add_argument("--page-size", type=int, default=100, help="Activity page size")
    parser.add_argument("--max-pages", type=int, default=200, help="Max activity pages")
    parser.add_argument("--report-dir", type=str, default="reports/account_pnl", help="Report output directory")

    args = parser.parse_args()

    end = _parse_dt(args.end) if args.end else datetime.now(timezone.utc)
    start = _parse_dt(args.start) if args.start else end - timedelta(days=args.days)

    activities = fetch_account_activities(
        client=alpaca_wrapper.alpaca_api,
        start=start,
        end=end,
        activity_types=["FILL"],
        direction="asc",
        page_size=args.page_size,
        max_pages=args.max_pages,
    )

    fills = _filter_fills(activities, start, end)
    symbols = {f["symbol"] for f in fills}

    current_prices = _current_prices()
    fee_rates = _fee_rates(symbols, args.crypto_fee_bps, args.stock_fee_bps)

    summary = compute_trade_pnl(
        fills,
        current_prices=current_prices,
        fee_rate_by_symbol=fee_rates,
    )

    history = _try_portfolio_history(start, end)
    equity_change = _compute_equity_change(history) if history else None

    per_symbol_rows: List[dict] = []
    for symbol, pnl in summary.per_symbol.items():
        per_symbol_rows.append(
            {
                "symbol": symbol,
                "realized": pnl.realized,
                "unrealized": pnl.unrealized,
                "fees": pnl.fees,
                "gross": pnl.gross,
                "net": pnl.net,
                "net_qty": pnl.net_qty,
                "avg_price": pnl.avg_price,
                "current_price": pnl.current_price,
            }
        )

    output_summary = {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "fills": len(fills),
        "symbols": sorted(symbols),
        "realized": summary.realized,
        "unrealized": summary.unrealized,
        "fees": summary.fees,
        "gross": summary.gross,
        "net": summary.net,
        "portfolio_history_available": bool(history),
        "start_equity": equity_change[0] if equity_change else None,
        "end_equity": equity_change[1] if equity_change else None,
        "actual_equity_change": (equity_change[1] - equity_change[0]) if equity_change else None,
        "note": (
            "Trade PnL assumes flat starting positions within window. "
            "If positions existed before the window, results are approximate."
        ),
    }

    json_path, csv_path = _write_report(
        report_dir=Path(args.report_dir),
        summary=output_summary,
        per_symbol_rows=per_symbol_rows,
    )

    print("=" * 72)
    print("Account Trade PnL (fills-based)")
    print("=" * 72)
    print(f"Window: {output_summary['window_start']} -> {output_summary['window_end']}")
    print(f"Fills: {output_summary['fills']} | Symbols: {len(symbols)}")
    print(
        f"Realized: {summary.realized:.2f} | Unrealized: {summary.unrealized:.2f} | "
        f"Fees: {summary.fees:.2f} | Net: {summary.net:.2f}"
    )
    if equity_change:
        print(
            f"Actual equity change: {equity_change[1] - equity_change[0]:.2f} "
            f"(start {equity_change[0]:.2f} -> end {equity_change[1]:.2f})"
        )
    print(f"Report JSON: {json_path}")
    print(f"Report CSV: {csv_path}")


if __name__ == "__main__":
    main()
